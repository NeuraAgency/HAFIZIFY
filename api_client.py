"""
api_client.py — Thin HTTP client for Combined Mode's optional remote API path
================================================================================
Lets realtime_streamer.py's process_chunk_api() delegate a single audio
chunk to a running api/main.py server's POST /transcribe endpoint instead of
running Groq + the local turbo model in-process (masterplan.md §9 — "Use
Remote API for Combined Mode"). Standard Mode and local Combined Mode
(process_chunk_combined()) are completely independent of this file.

Usage
-----
    from api_client import transcribe_via_api
    result = transcribe_via_api(
        chunk_audio_np, sample_rate=16000,
        surah=1, expected_ayah=1,
        correction_mode="balanced", asr_engine="combined",
        model_choice="groq-whisper-large-v3",
        api_base_url="http://127.0.0.1:8000",
    )
    # result is a dict matching api/main.py's /transcribe JSON response,
    # or None if the request failed (server down, timeout, non-2xx, etc.)
"""

import io
import wave
from typing import Optional, Dict, Any

import numpy as np

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None
    _HAS_HTTPX = False

# Matches app.py's default Advanced Settings value and api/main.py's own
# default bind address (uvicorn api.main:app --host 0.0.0.0 --port 8000).
_DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT_S = 120.0  # TEMP: bumped from 45s so a single chunk can be
                            # seen through end-to-end on CPU-only hardware
                            # (both app.py and uvicorn sharing the same 6
                            # physical cores easily pushes one chunk past
                            # 60-90s here). Once Combined Mode actually runs
                            # on a GPU server, drop this back down — a real
                            # 2-minute stall per chunk is not usable for live
                            # reciting, this is only for a one-time "does it
                            # work end to end" check.


def _write_wav_bytes(audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert a float32 numpy array to WAV bytes (PCM 16-bit).

    Same technique as groq_transcriber.py's _write_wav_bytes() — duplicated
    here rather than imported so api_client.py has no dependency on the
    Groq SDK. This module exists specifically for the case where this
    machine may NOT have Groq/the local turbo model installed at all
    (masterplan.md §9 — that's the point of delegating to a remote server).
    """
    audio_np = np.clip(audio_np.astype(np.float32), -1.0, 1.0)
    pcm = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe_via_api(
    audio_np: np.ndarray,
    sample_rate: int,
    surah: Optional[int],
    expected_ayah: Optional[int],
    correction_mode: str = "balanced",
    asr_engine: str = "combined",
    model_choice: str = "groq-whisper-large-v3",
    api_base_url: str = _DEFAULT_API_BASE_URL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    """POST one audio chunk to a running api/main.py server's /transcribe
    endpoint and return its parsed JSON response.

    Mirrors api/main.py's own request contract exactly (see its
    @app.post("/transcribe") handler): a multipart `file` plus the
    `surah` (required), `model_choice`, `correction_mode`, `expected_ayah`,
    and `asr_engine` form fields. The returned dict has the same keys as
    that endpoint's JSON response (raw_asr, corrected_text, matched_ayah,
    matched_ayah_text, confidence, confidence_level, verdict, cer, wer,
    harakaat_errors, harakaat_error_count, word_errors).

    Returns None (never raises) on any failure — httpx not installed,
    connection refused, timeout, non-2xx status, or malformed JSON — so
    realtime_streamer.py's process_chunk_api() can degrade gracefully to a
    low-confidence "error" chunk instead of crashing the session
    (masterplan.md §9).
    """
    if not _HAS_HTTPX:
        print(
            "[api_client] httpx not installed — cannot reach remote API. "
            "Run: pip install httpx"
        )
        return None

    if surah is None:
        # api/main.py's /transcribe requires surah (Form(...), no default).
        # A session whose auto-detection hasn't locked a surah yet has
        # nothing valid to send — fail this one chunk rather than guess.
        print("[api_client] No surah known yet — skipping remote API call for this chunk")
        return None

    try:
        wav_bytes = _write_wav_bytes(audio_np, sample_rate)
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {
            "surah": str(int(surah)),
            "model_choice": model_choice,
            "correction_mode": correction_mode,
            "expected_ayah": str(int(expected_ayah)) if expected_ayah is not None else "1",
            "asr_engine": asr_engine,
        }
        url = f"{api_base_url.rstrip('/')}/transcribe"
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"[api_client] /transcribe call failed ({api_base_url}): {exc}")
        return None
