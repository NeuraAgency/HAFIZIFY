"""
Real-Time Recitation Session Manager
-------------------------------------
Manages the lifecycle of a live recitation session:
  - **VAD-based chunking** (Silero VAD) for intelligent speech segmentation
  - Ring buffer for audio accumulation
  - Dual WAV recording (individual segments + full session)
  - Ayah progression tracking across chunks
  - Result accumulation and comparison
"""

import os
import time
import uuid
import json
import re
import difflib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set, Tuple

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile is required. Install with: pip install soundfile")

# ---------------------------------------------------------------------------
# VAD imports (graceful fallback)
# ---------------------------------------------------------------------------

_VAD_AVAILABLE = False
try:
    import torch
    # silero-vad >= 5.x exposes these at top-level
    try:
        # pyrefly: ignore [missing-import]
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError:
        _hub_utils = None
        def load_silero_vad():
            global _hub_utils
            _model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            _hub_utils = _utils
            return _model

        def get_speech_timestamps(audio, model, **kwargs):
            global _hub_utils
            if _hub_utils is None:
                load_silero_vad()
            _gst = _hub_utils[0]
            return _gst(audio, model, **kwargs)
    _VAD_AVAILABLE = True
except Exception as exc:
    print(f"[SessionManager] VAD not available ({exc}); falling back to fixed-time chunking.")


def _apply_fade(audio: np.ndarray, fade_ms: int = 5, sample_rate: int = 16000) -> np.ndarray:
    """Apply short fade-in and fade-out to avoid concatenation clicks."""
    fade_samples = int(fade_ms * sample_rate / 1000)  # 5ms = 80 samples
    if len(audio) < fade_samples * 2:
        return audio
    audio = audio.copy()
    # fade in
    audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
    # fade out
    audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return audio


SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# VAD tuning constants — optimised for Quran recitation
# ---------------------------------------------------------------------------

# ── VAD tuning for Quran recitation (long sessions) ─────────────────────────
# The critical value is _VAD_MIN_SILENCE_MS: Quran reciters pause between
# ayahs for roughly 50–200ms.  Setting this to 100ms ensures those pauses
# are detected as split points, so a 19-minute session gets broken into
# individual ayah-length chunks instead of one giant speech segment that
# exceeds Whisper's 30-second window.
#
# FIX: Increased from 180ms → 500ms to prevent mid-ayah choppiness.
# Quran recitation has natural word-boundary pauses of 200–400ms.
# A 180ms threshold was splitting mid-ayah, creating choppy ~4s chunks.
_VAD_MIN_SILENCE_MS      = 300     # only split on ayah-boundary pauses (500ms+)
_VAD_MIN_SPEECH_MS       = 150     # catch short ayahs promptly
_VAD_THRESHOLD           = 0.25    # sensitive — soft consonants miss na hon
_VAD_END_PAD_MS          = 100      # keep end padding
_VAD_MIN_CHUNK_DURATION  = 1.5     # 1.5s — matches the anti-hallucination gate in realtime_streamer.py; used only as documentation of that downstream floor, NOT to merge across real pauses
_VAD_NOISE_BLIP_DURATION = 0.6     # segments shorter than this are treated as VAD noise, not real ayahs — merge only these forward
_VAD_MIN_EMIT_DURATION   = 0.3     # absolute floor to avoid emitting near-empty slivers; realtime_streamer's 1.5s gate does the real quality filtering
_VAD_MAX_CHUNK_DURATION  = 15.0    # 15.0s safety net — VAD silence detection is the primary splitter
_VAD_RESCAN_INTERVAL     = 0.15    # rescan interval
_VAD_TAIL_CONFIRM        = 0.05    # 50ms tail confirmation for rapid emission


@dataclass
class ChunkResult:
    """Result from processing a single audio chunk."""
    chunk_index: int
    start_sample: int
    end_sample: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    raw_asr: str
    corrected_text: str
    matched_ayah: Optional[int]
    matched_surah_ayah_id: Optional[str]
    cer: Optional[float]
    wer: Optional[float]
    coverage: Optional[float]
    confidence: Optional[float]
    confidence_level: Optional[str]
    verdict: str
    chunk_wav_path: str
    matched_ayah_text: Optional[str] = None
    errors: List[Dict[str, str]] = field(default_factory=list)
    surah_lock_state: Optional[Dict[str, Any]] = None


@dataclass
class SessionSummary:
    """Final summary comparing chunk-based vs full-session decode."""
    session_id: str
    total_duration_s: float
    num_chunks: int
    chunk_merged_transcript: str
    full_session_transcript: str
    chunk_results: List[Dict[str, Any]]
    session_wav_path: str
    chunks_dir: str
    results_json_path: str


class RecitationSession:
    """Manages a single live recitation session with VAD-based chunk processing."""

    def __init__(
        self,
        session_dir: str,
        chunk_duration_s: float = 5.0,       # only used for fixed-time fallback
        overlap_duration_s: float = 1.5,     # only used for fixed-time fallback
        surah: Optional[int] = None,
        start_ayah: int = 1,
        use_vad: bool = True,
    ):
        self.session_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.session_dir = os.path.join(session_dir, self.session_id)
        self.chunks_dir = os.path.join(self.session_dir, "chunks")
        self.vad_segments_dir = os.path.join(self.session_dir, "vad_segments")
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.vad_segments_dir, exist_ok=True)

        # Fixed-time fallback parameters
        self.chunk_duration_s = chunk_duration_s
        self.overlap_duration_s = overlap_duration_s
        self.chunk_size = int(chunk_duration_s * SAMPLE_RATE)
        self.overlap_size = int(overlap_duration_s * SAMPLE_RATE)
        self.step_size = max(1, self.chunk_size - self.overlap_size)

        self.surah = surah
        self.current_ayah = start_ayah
        self.start_ayah = start_ayah

        # VAD state
        self.use_vad = use_vad and _VAD_AVAILABLE
        self._vad_model = None
        if self.use_vad:
            try:
                self._vad_model = load_silero_vad()
                print(f"[Session {self.session_id}] [OK] Silero VAD loaded")
            except Exception as e:
                print(f"[Session {self.session_id}] [WARNING] VAD init failed ({e}); falling back to fixed-time")
                self.use_vad = False



        # Audio accumulation
        self._buffer = np.array([], dtype=np.float32)
        self._total_samples_fed = 0
        self._next_chunk_start = 0
        self._chunk_index = 0

        # Full session recording
        self._all_audio = np.array([], dtype=np.float32)

        # Results
        self.chunk_results: List[ChunkResult] = []
        self.pending_chunks: List[Dict[str, Any]] = []
        self._next_pending_index = 0
        self.is_active = True
        self._pending_fragment = ""
        self.surah_lock_state: Optional[Dict[str, Any]] = None

        # Session WAV path
        self.session_wav_path = os.path.join(self.session_dir, "session_full.wav")

        # ---- Incremental VAD state ----
        self._vad_committed_up_to = 0
        self._vad_scanned_up_to = 0
        self._vad_segment_counter = 0
        # Re-run VAD every 150ms for fast ayah-boundary detection
        self._vad_min_new_samples = int(_VAD_RESCAN_INTERVAL * SAMPLE_RATE)

    # ----------------------------------------------------------------
    # VAD core — rewritten for aggressive ayah-level splitting
    # ----------------------------------------------------------------

    def _run_vad_on_window(
        self,
        audio_array: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> List[Dict[str, int]]:
        """Run Silero VAD with settings tuned for Quran recitation.

        Tuned for long sessions (19+ minutes):
          - threshold = 0.35          Sensitive enough for quiet recitation
          - min_silence_duration_ms   = 100ms   Split on short ayah-boundary pauses
          - min_speech_duration_ms    = 150ms   Catch short ayahs (e.g. Al-Fatiha)
        """
        if not self.use_vad or self._vad_model is None:
            return []

        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=-1)

        try:
            timestamps = get_speech_timestamps(
                audio_tensor,
                self._vad_model,
                sampling_rate=sr,
                threshold=_VAD_THRESHOLD,               # 0.35 — sensitive
                min_speech_duration_ms=_VAD_MIN_SPEECH_MS,   # 200ms
                min_silence_duration_ms=_VAD_MIN_SILENCE_MS, # 100ms — split here!
                return_seconds=False,
            )
        except Exception as e:
            print(f"[VAD] Error: {e}")
            return []

        print(f"[VAD] Found {len(timestamps)} speech segment(s) in "
              f"{len(audio_array)/sr:.1f}s window")
        return timestamps

    def _split_long_segment(
        self,
        audio: np.ndarray,
        abs_start: int,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Split audio that exceeds _VAD_MAX_CHUNK_DURATION into smaller pieces.

        Uses energy-based splitting to find the best split point (lowest energy
        region) rather than blindly cutting at the midpoint, which would
        truncate words mid-syllable.
        """
        max_samples = int(_VAD_MAX_CHUNK_DURATION * SAMPLE_RATE)
        if len(audio) <= max_samples:
            return [(audio, abs_start, abs_start + len(audio))]

        pieces = []
        pos = 0
        while pos < len(audio):
            end = min(pos + max_samples, len(audio))
            if end < len(audio):
                # Find the lowest-energy 50ms region around the split boundary
                # to avoid cutting mid-word
                search_start = max(pos, end - int(0.3 * SAMPLE_RATE))
                search_end = min(len(audio), end + int(0.2 * SAMPLE_RATE))
                window = int(0.05 * SAMPLE_RATE)  # 50ms window
                best_split = end
                best_energy = float("inf")
                for candidate in range(search_start, search_end - window):
                    energy = float(np.mean(audio[candidate:candidate + window] ** 2))
                    if energy < best_energy:
                        best_energy = energy
                        best_split = candidate
                end = best_split
            piece = audio[pos:end]
            if len(piece) >= int(_VAD_MIN_CHUNK_DURATION * SAMPLE_RATE):
                pieces.append((piece, abs_start + pos, abs_start + end))
            pos = end
        return pieces if pieces else [(audio, abs_start, abs_start + len(audio))]

    def _emit_segment(
        self,
        abs_start: int,
        abs_end: int,
        total_len: int,
        label: str = "vad",
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Extract, amplify, and save a confirmed speech segment.

        Returns (chunk_audio, padded_start, padded_end) or None if too short.
        """
        end_pad = int(_VAD_END_PAD_MS / 1000 * SAMPLE_RATE)
        padded_end = min(total_len, abs_end + end_pad)
        chunk_audio = self._all_audio[abs_start:padded_end].copy()
        chunk_audio = _apply_fade(chunk_audio, fade_ms=10)

        if len(chunk_audio) < int(_VAD_MIN_EMIT_DURATION * SAMPLE_RATE):
            return None

        # High-pass filter — remove DC offset & low-freq rumble (safe on full chunk)
        if len(chunk_audio) > 20:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 80.0 / (SAMPLE_RATE / 2), btype='high')
            chunk_audio = filtfilt(b, a, chunk_audio).astype(np.float32)

        # Gentle RMS-based normalisation to a consistent level.
        # Uses RMS (not peak) to avoid volume pumping between chunks.
        # Target RMS = 0.12 (~-18dBFS) — comfortable listening level.
        # Clamp gain to ±6dB to prevent extreme boosting/attenuation.
        rms = float(np.sqrt(np.mean(chunk_audio ** 2)))
        if rms > 1e-6:
            target_rms = 0.12
            gain = target_rms / rms
            gain = max(0.5, min(2.0, gain))  # clamp to ±6dB
            chunk_audio = chunk_audio * gain

        # Soft limiter — smoothly compresses peaks above the knee instead of
        # hard-clipping them flat. A hard np.clip introduces audible
        # harshness/distortion on loud consonants when the RMS gain above
        # pushes a transient over 1.0; this keeps everything below the knee
        # untouched and only rounds off the very top of the waveform.
        limiter_knee = 0.9
        headroom = 1.0 - limiter_knee
        over = np.abs(chunk_audio) > limiter_knee
        if np.any(over):
            sign = np.sign(chunk_audio[over])
            excess = np.abs(chunk_audio[over]) - limiter_knee
            chunk_audio[over] = sign * (limiter_knee + headroom * np.tanh(excess / headroom))
        # Final safety net (tanh asymptotically approaches but never exceeds
        # 1.0, so this should be a no-op in practice)
        np.clip(chunk_audio, -1.0, 1.0, out=chunk_audio)

        # Save WAV for debugging
        seg_wav = os.path.join(
            self.vad_segments_dir,
            f"{label}_{self._vad_segment_counter:04d}_{abs_start}_{abs_end}.wav",
        )
        try:
            sf.write(seg_wav, chunk_audio, SAMPLE_RATE)
        except Exception:
            pass

        return chunk_audio, abs_start, padded_end

    def _merge_and_emit(
        self,
        timestamps: List[Dict[str, int]],
        window_start: int,
        total_len: int,
        tail_confirm: int,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Merge adjacent VAD segments into chunks and emit confirmed ones.

        Adjacent segments are merged until the group meets
        _VAD_MIN_CHUNK_DURATION.  This prevents short speech bursts
        from being discarded, which was the main cause of choppy audio.
        """
        noise_blip = int(_VAD_NOISE_BLIP_DURATION * SAMPLE_RATE)
        pre_pad = int(0.15 * SAMPLE_RATE)

        # ── Build groups ──
        # IMPORTANT: every pair of consecutive VAD timestamps is, by construction,
        # already separated by >= _VAD_MIN_SILENCE_MS of real silence (that's what
        # min_silence_duration_ms means). Merging across that gap just to hit a
        # duration floor re-splices legitimate ayah-boundary pauses back together —
        # that was the source of the choppy/garbled chunks. We only merge a segment
        # forward when IT ITSELF is a tiny noise blip (shorter than
        # _VAD_NOISE_BLIP_DURATION), never just because the running group is short.
        groups: List[Tuple[int, int]] = []
        g_start = timestamps[0]["start"]
        g_end   = timestamps[0]["end"]

        for seg in timestamps[1:]:
            if (g_end - g_start) >= noise_blip:
                # Current group is a real segment on its own → finalise it
                groups.append((g_start, g_end))
                g_start = seg["start"]
            # Otherwise the running group is just a noise blip — fold the next
            # segment into it instead of emitting the blip alone.
            g_end = seg["end"]
        groups.append((g_start, g_end))  # last group

        # ── Emit groups whose trailing silence is confirmed ───────────
        ready: List[Tuple[np.ndarray, int, int]] = []

        for g_s, g_e in groups:
            abs_end = window_start + g_e

            # Still speaking or not enough silence yet → stop here
            if abs_end + tail_confirm > total_len:
                break

            abs_start = max(self._vad_committed_up_to,
                            window_start + g_s - pre_pad)

            result = self._emit_segment(abs_start, abs_end, total_len, "vad")
            self._vad_committed_up_to = abs_end
            self._vad_segment_counter += 1

            if result is None:
                continue  # too short even after merge — skip, audio consumed

            chunk_audio, padded_start, padded_end = result
            pieces = self._split_long_segment(chunk_audio, padded_start)
            for p_audio, p_start, p_end in pieces:
                ready.append((p_audio, p_start, p_end))
                print(f"[VAD] Emitting: "
                      f"{p_start/SAMPLE_RATE:.1f}s–{p_end/SAMPLE_RATE:.1f}s "
                      f"({len(p_audio)/SAMPLE_RATE:.2f}s)")

        return ready

    # ----------------------------------------------------------------
    # Audio feeding
    # ----------------------------------------------------------------

    def feed_audio(self, audio_fragment: np.ndarray) -> List[tuple]:
        """Feed a small audio fragment from the microphone.

        Returns a list of (audio_np, start_sample, end_sample) tuples —
        one per confirmed speech segment detected by VAD.
        """
        if not self.is_active:
            return []

        # Ensure float32, mono
        if audio_fragment.ndim > 1:
            audio_fragment = audio_fragment.mean(axis=1)
        audio_fragment = audio_fragment.astype(np.float32)

        # Append to buffer and full session recording
        self._buffer = np.concatenate([self._buffer, audio_fragment])
        self._all_audio = np.concatenate([self._all_audio, audio_fragment])
        self._total_samples_fed += len(audio_fragment)

        if self.use_vad and self._vad_model is not None:
            return self._feed_audio_vad()

        return self._feed_audio_fixed()

    def _feed_audio_vad(self) -> List[tuple]:
        """VAD-based chunking with automatic segment merging.

        Runs VAD on the uncommitted audio window, merges adjacent short
        segments so that each emitted chunk meets _VAD_MIN_CHUNK_DURATION,
        and only emits chunks whose trailing silence is confirmed.
        """
        total_len = len(self._all_audio)

        # Throttle: don't re-run VAD unless enough new audio arrived
        if total_len - self._vad_scanned_up_to < self._vad_min_new_samples:
            return []
        self._vad_scanned_up_to = total_len

        window_start = self._vad_committed_up_to
        window = self._all_audio[window_start:]

        if len(window) < int(0.5 * SAMPLE_RATE):
            return []

        timestamps = self._run_vad_on_window(window)
        if not timestamps:
            return []

        return self._merge_and_emit(timestamps, window_start, total_len,
                                    tail_confirm=int(_VAD_TAIL_CONFIRM * SAMPLE_RATE))

    def _feed_audio_fixed(self) -> List[tuple]:
        """Fixed-time chunking fallback (used when VAD is unavailable)."""
        ready_chunks = []
        while self._next_chunk_start + self.chunk_size <= len(self._all_audio):
            chunk_start = self._next_chunk_start
            chunk_end = chunk_start + self.chunk_size
            chunk_audio = self._all_audio[chunk_start:chunk_end].copy()
            ready_chunks.append((chunk_audio, chunk_start, chunk_end))
            self._next_chunk_start += self.step_size
        return ready_chunks

    def skip_paused_audio(self):
        """Commit/skip any audio accumulated while paused so it won't be chunked upon resuming."""
        total_len = len(self._all_audio)
        self._vad_committed_up_to = total_len
        self._vad_scanned_up_to = total_len
        self._next_chunk_start = total_len

    def flush_remaining_audio(self) -> List[tuple]:
        """Extract any remaining unprocessed audio when stopping the session.

        Runs a final VAD pass on the uncommitted tail, then falls back to a
        safety-net that emits everything not yet processed.
        """
        if not self.is_active:
            return []

        remaining_chunks = []

        if self.use_vad and self._vad_model is not None:
            window_start = self._vad_committed_up_to
            window_audio = self._all_audio[window_start:]
            total_len = len(self._all_audio)

            if len(window_audio) >= int(0.2 * SAMPLE_RATE):
                timestamps = self._run_vad_on_window(window_audio)
                if timestamps:
                    # On flush, no tail confirmation needed — session ending
                    remaining_chunks.extend(
                        self._merge_and_emit(
                            timestamps, window_start, total_len,
                            tail_confirm=0,
                        )
                    )

            # Safety net: emit anything VAD missed
            leftover_start = self._vad_committed_up_to
            leftover_audio = self._all_audio[leftover_start:]

            if len(leftover_audio) >= int(0.3 * SAMPLE_RATE):
                result = self._emit_segment(
                    leftover_start,
                    len(self._all_audio),
                    len(self._all_audio),
                    label="vad_tail",
                )
                self._vad_segment_counter += 1
                self._vad_committed_up_to = len(self._all_audio)

                if result is not None:
                    chunk_audio, padded_start, padded_end = result
                    pieces = self._split_long_segment(chunk_audio, padded_start)
                    remaining_chunks.extend(pieces)
                    print(f"[Session] Safety-net: flushed "
                          f"{len(leftover_audio)/SAMPLE_RATE:.1f}s → "
                          f"{len(pieces)} piece(s)")
        else:
            # Fixed-time: flush remainder
            remaining_start = self._next_chunk_start
            if remaining_start < len(self._all_audio):
                remaining_audio = self._all_audio[remaining_start:].copy()
                if len(remaining_audio) >= SAMPLE_RATE:
                    remaining_chunks.append(
                        (remaining_audio, remaining_start, len(self._all_audio))
                    )
                    self._next_chunk_start = len(self._all_audio)

        return remaining_chunks

    # ----------------------------------------------------------------
    # Chunk registration and session bookkeeping (unchanged)
    # ----------------------------------------------------------------

    def save_chunk_wav(self, chunk_audio: np.ndarray) -> str:
        """Save an individual chunk as a WAV file. Returns the path."""
        filename = f"chunk_{self._chunk_index:04d}.wav"
        filepath = os.path.join(self.chunks_dir, filename)
        sf.write(filepath, chunk_audio, SAMPLE_RATE)
        return filepath

    def register_pending_chunk(self, start_sample: int, end_sample: int) -> int:
        """Register a chunk that has been sent to the queue but not yet processed."""
        idx = self._next_pending_index
        self._next_pending_index += 1
        self.pending_chunks.append({
            "chunk_index": idx,
            "start_time_s": start_sample / SAMPLE_RATE,
            "end_time_s": end_sample / SAMPLE_RATE,
        })
        return idx

    def discard_pending_chunk(self, start_sample: int, end_sample: int) -> None:
        """Remove a pending chunk entry without registering a result.
        Used by the worker to drop stale chunks that were queued before
        the Qari correction engine paused, so they don't corrupt state
        after a correction has already started."""
        target = start_sample / SAMPLE_RATE
        for i, p in enumerate(self.pending_chunks):
            if abs(p["start_time_s"] - target) < 0.05:
                self.pending_chunks.pop(i)
                return

    def register_chunk_result(
        self,
        chunk_audio: np.ndarray,
        guard_result: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
        surah_lock_state: Optional[Dict[str, Any]] = None,
        chunk_start_sample: Optional[int] = None,
        chunk_end_sample: Optional[int] = None,
    ) -> ChunkResult:
        """Register the result of processing a chunk."""
        chunk_wav_path = self.save_chunk_wav(chunk_audio)

        if chunk_start_sample is not None and chunk_end_sample is not None:
            actual_start = chunk_start_sample
            actual_end = chunk_end_sample
        else:
            actual_start = self._chunk_index * self.step_size
            actual_end = actual_start + self.chunk_size

        # Remove from pending chunks if it exists (FIFO)
        actual_index = self._chunk_index
        if self.pending_chunks:
            pending = self.pending_chunks.pop(0)
            actual_index = pending["chunk_index"]

        start_time = actual_start / SAMPLE_RATE
        end_time = actual_end / SAMPLE_RATE
        duration = round((actual_end - actual_start) / SAMPLE_RATE, 2)

        errors = []
        if eval_result and isinstance(eval_result, dict):
            errors = eval_result.get("errors", [])

        corrected_text = guard_result.get("corrected_text", "")
        if self._pending_fragment:
            corrected_text = f"{self._pending_fragment} {corrected_text}".strip()

        fragment, corrected_text = self._extract_trailing_fragment(
            corrected_text,
            guard_result.get("matched_ayah_text", ""),
        )
        self._pending_fragment = fragment
        guard_result["corrected_text"] = corrected_text

        # Update current ayah tracking based on matched ayah
        matched_key = guard_result.get("matched_key")
        if matched_key and isinstance(matched_key, (tuple, list)):
            matched_surah = matched_key[0]
            matched_ayah = matched_key[1]
            if self.surah is None:
                self.surah = matched_surah
                self.start_ayah = matched_ayah
                self.current_ayah = matched_ayah
            elif matched_surah == self.surah:
                if matched_ayah > self.current_ayah:
                    self.current_ayah = matched_ayah

        result = ChunkResult(
            chunk_index=actual_index,
            start_sample=actual_start,
            end_sample=actual_end,
            start_time_s=round(start_time, 2),
            end_time_s=round(end_time, 2),
            duration_s=round(duration, 2),
            raw_asr=guard_result.get("raw_asr", ""),
            corrected_text=guard_result.get("corrected_text", ""),
            matched_ayah=guard_result.get("matched_ayah"),
            matched_surah_ayah_id=(
                f"{self.surah}:{guard_result.get('matched_ayah')}"
                if self.surah and guard_result.get("matched_ayah")
                else None
            ),
            matched_ayah_text=guard_result.get("matched_ayah_text", None),
            cer=guard_result.get("cer"),
            wer=guard_result.get("wer"),
            coverage=guard_result.get("coverage"),
            confidence=guard_result.get("confidence"),
            confidence_level=guard_result.get("confidence_level"),
            verdict=guard_result.get("verdict", "unknown"),
            chunk_wav_path=chunk_wav_path,
            errors=errors,
            surah_lock_state=surah_lock_state,
        )

        self.chunk_results.append(result)
        self._chunk_index += 1
        return result

    @staticmethod
    def _normalize_for_fragment(text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(
            r"[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED"
            r"\u00AB\u00BB\u200F\u200E\u202A-\u202E\uFEFF\u06DD\u06DE"
            r"۩۞۝]+",
            "",
            text,
        )
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def _extract_trailing_fragment(self, corrected_text: str, ref_text: str) -> tuple:
        if not corrected_text or not ref_text:
            return "", corrected_text

        corrected_tokens = self._normalize_for_fragment(corrected_text)
        ref_tokens = self._normalize_for_fragment(ref_text)
        if not corrected_tokens or not ref_tokens:
            return "", corrected_text

        last_token = corrected_tokens[-1]
        if len(last_token) < 2 or last_token in ref_tokens:
            return "", corrected_text

        for ref_token in ref_tokens:
            if ref_token.startswith(last_token) and len(last_token) < len(ref_token):
                original_tokens = corrected_text.split()
                if original_tokens:
                    original_tokens = original_tokens[:-1]
                return last_token, " ".join(original_tokens).strip()

        return "", corrected_text

    def get_full_session_audio(self) -> np.ndarray:
        """Return the complete session audio as a single numpy array."""
        return self._all_audio.copy()

    def save_session_wav(self) -> str:
        """Save the full session recording to disk. Returns the path."""
        if len(self._all_audio) > 0:
            sf.write(self.session_wav_path, self._all_audio, SAMPLE_RATE)
        return self.session_wav_path

    def get_merged_transcript(self) -> str:
        """Merge all chunk corrected texts into a single transcript."""
        if not self.chunk_results:
            return ""

        merged_parts = []
        prev_words = []
        prev_norm_words = []

        def _normalize_tokens(text: str) -> list:
            if not text:
                return []
            text = re.sub(
                r"[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED"
                r"\u00AB\u00BB\u200F\u200E\u202A-\u202E\uFEFF\u06DD\u06DE"
                r"۩۞۝]+",
                "",
                text,
            )
            text = re.sub(r"\s+", " ", text).strip()
            return text.split()

        def _best_overlap_len(prev_tokens: list, curr_tokens: list, max_check: int = 10, min_ratio: float = 0.85) -> int:
            if not prev_tokens or not curr_tokens:
                return 0
            best_len = 0
            best_ratio = 0.0
            max_len = min(len(prev_tokens), len(curr_tokens), max_check)
            for overlap_len in range(1, max_len + 1):
                prev_tail = " ".join(prev_tokens[-overlap_len:])
                curr_head = " ".join(curr_tokens[:overlap_len])
                ratio = difflib.SequenceMatcher(None, prev_tail, curr_head).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_len = overlap_len
            return best_len if best_ratio >= min_ratio else 0

        for result in self.chunk_results:
            current_words = result.corrected_text.split()
            current_norm_words = _normalize_tokens(result.corrected_text)
            if not current_words:
                continue

            if prev_words:
                best_overlap = _best_overlap_len(prev_norm_words, current_norm_words)
                if best_overlap > 0:
                    current_words = current_words[best_overlap:]
                    current_norm_words = current_norm_words[best_overlap:]

            if current_words:
                merged_parts.extend(current_words)
            prev_words = result.corrected_text.split()
            prev_norm_words = _normalize_tokens(result.corrected_text)

        return " ".join(merged_parts)

    def get_detected_surah(self) -> Optional[int]:
        """Return the most frequently matched surah across chunk results."""
        locked = self.get_final_locked_surah()
        if locked is not None:
            return locked
        if not self.chunk_results:
            return self.surah
        counts: Dict[Optional[int], int] = {}
        for r in self.chunk_results:
            key = r.matched_surah_ayah_id
            if key:
                surah_num = int(key.split(":")[0])
                counts[surah_num] = counts.get(surah_num, 0) + 1
        if not counts:
            return self.surah
        return max(counts, key=counts.get)

    def get_chunk_results_as_dicts(self) -> List[Dict[str, Any]]:
        """Return chunk results as dicts suitable for the LiveDisplayFormatter."""
        results = []
        for r in self.chunk_results:
            results.append({
                "corrected_text": r.corrected_text,
                "raw_asr": r.raw_asr,
                "matched_ayah": r.matched_ayah,
                "matched_surah_ayah_id": r.matched_surah_ayah_id,
                "matched_ayah_text": r.matched_ayah_text,
                "surah": self.surah,
                "surah_lock_state": r.surah_lock_state,
                "confidence": r.confidence,
                "confidence_level": r.confidence_level,
                "cer": r.cer,
                "wer": r.wer,
                "verdict": r.verdict,
            })
        return results

    def _get_final_locked_surah(self) -> Optional[int]:
        for r in reversed(self.chunk_results):
            state = r.surah_lock_state or {}
            if state.get("lock_state") == "ACTIVE" and state.get("locked_surah") is not None:
                return int(state["locked_surah"])
        return self.surah

    def get_final_locked_surah(self) -> Optional[int]:
        return self._get_final_locked_surah()

    def finalize(self) -> str:
        """Finalize the session: save session WAV, save results JSON."""
        self.is_active = False
        self.save_session_wav()

        results_path = os.path.join(self.session_dir, "results.json")
        payload = {
            "session_id": self.session_id,
            "surah": self.surah,
            "locked_surah": self._get_final_locked_surah(),
            "start_ayah": self.start_ayah,
            "final_ayah": self.current_ayah,
            "chunk_duration_s": self.chunk_duration_s,
            "overlap_duration_s": self.overlap_duration_s,
            "total_duration_s": round(len(self._all_audio) / SAMPLE_RATE, 2),
            "num_chunks": len(self.chunk_results),
            "vad_enabled": self.use_vad,
            "merged_transcript": self.get_merged_transcript(),
            "session_wav_path": self.session_wav_path,
            "chunks_dir": self.chunks_dir,
            "vad_segments_dir": self.vad_segments_dir,
            "chunks": [
                {
                    "index": r.chunk_index,
                    "time_range": f"{r.start_time_s:.2f}s - {r.end_time_s:.2f}s",
                    "raw_asr": r.raw_asr,
                    "corrected_text": r.corrected_text,
                    "matched_surah_ayah_id": r.matched_surah_ayah_id,
                    "cer": r.cer,
                    "wer": r.wer,
                    "confidence": r.confidence,
                    "verdict": r.verdict,
                    "chunk_wav": r.chunk_wav_path,
                    "errors": r.errors,
                    "surah_lock_state": r.surah_lock_state,
                }
                for r in self.chunk_results
            ],
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return results_path

    @property
    def total_duration_s(self) -> float:
        return len(self._all_audio) / SAMPLE_RATE
