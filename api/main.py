"""
Hafizify Mobile API
====================
FastAPI layer for the Expo/React Native mobile client. This is purely
additive — it imports the exact same engine modules `app.py` (the Gradio
desktop app) uses, and does not modify any of them. The desktop app keeps
working completely independently; this process can run at the same time
or separately.

Run from the `hafizify/` project root (same venv as the desktop app):

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Two capabilities:
  1. POST /transcribe        - stateless file-upload transcription
                                (mirrors the Gradio "File Upload" tab)
  2. Live session lifecycle  - mirrors the Gradio "Live Recitation" tab:
       POST /session/start
       WS   /session/{session_id}/stream
       POST /session/{session_id}/stop

IMPORTANT CONSTRAINT (intentional, not a bug):
RealtimeStreamer holds session-shaped state on itself (correction_engine,
_decode_context, _vad_paused) — exactly like the desktop app's global
`rt_streamer`. The desktop app only ever supports ONE active recitation
session at a time (see `_active_session` in app.py). This API preserves
that same constraint rather than touching realtime_streamer.py to make it
multi-session-safe. A second /session/start while one is active returns
409. The /transcribe endpoint has no such limit — it's a stateless decode
and is safe under concurrent calls (serialized internally by a lock, same
as the model's own thread-safety requirement on desktop).
"""
import os
import sys
import tempfile
import threading
import uuid
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Make the hafizify project root (parent of this api/ folder) importable,
# exactly the way app.py adds fyp_model to sys.path relative to itself.
# ---------------------------------------------------------------------------
_API_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_API_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FYP_DIR = os.path.join(_PROJECT_ROOT, "fyp_model")
if _FYP_DIR not in sys.path:
    sys.path.insert(0, _FYP_DIR)

import librosa  # noqa: E402

from realtime_streamer import RealtimeStreamer, resample_to_16k  # noqa: E402
from session_manager import (  # noqa: E402
    RecitationSession,
    SAMPLE_RATE,
    _VAD_AVAILABLE,
    load_silero_vad,
    get_speech_timestamps,
)
from fyp_model.quran_guard import guard_inference, get_word_error_annotations, correct_text_rules  # noqa: E402

from api.formatters import (  # noqa: E402
    chunk_result_to_json,
    error_to_json,
    qari_action_to_json,
    session_summary_to_json,
)

app = FastAPI(title="Hafizify Mobile API")

# CORS: needed for the web client (browser fetch/WS is origin-restricted;
# React Native / Expo isn't, so this is a no-op for mobile). Wide open for
# now since this is a pre-deployment test build behind a tunnel URL that
# changes anyway — tighten allow_origins to your real web domain(s) once
# you deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_combined_mode_model():
    """Preload the local turbo model at server BOOT, not on the first
    /transcribe or WS chunk that needs it.

    Without this, the very first Combined Mode request pays the full cold
    weights-load cost (several seconds — see 'Loading weights: 100%...' in
    the server log) on top of Groq + decode + merge time, all within one
    request. That routinely exceeds api_client.py's client-side timeout,
    so the caller (app.py, via process_chunk_api()) sees 'timed out' at the
    exact moment the server is still finishing up and about to return 200
    OK a few seconds later — it looks broken but isn't. This mirrors what
    app.py's start_live_session() and this file's own /session/start
    handler already do for the desktop app / live-session path; /transcribe
    never went through either of those, so it never got the same warm-up.

    Runs in a background thread so a slow model load never delays the
    server actually accepting connections; the first real request just
    waits on _local_pipeline being ready if it beats the thread.
    """
    import threading
    from hybrid_diacritic_pipeline import preload_local_pipeline

    def _preload():
        try:
            preload_local_pipeline()
            print("[API] Combined Mode local model preloaded at server startup.")
        except Exception as e:
            print(f"[API] Combined Mode preload at startup failed, will lazy-load on first request: {e}")

    threading.Thread(target=_preload, daemon=True).start()

# ---------------------------------------------------------------------------
# Shared engine state — one process, one model in memory, one live session
# at a time (see module docstring for why).
# ---------------------------------------------------------------------------
rt_streamer = RealtimeStreamer()

# Guards concurrent calls into the model (generate() / transcribe() are not
# thread-safe to call from two requests at once — same limitation the
# desktop app avoids by funnelling everything through one worker thread).
_INFERENCE_LOCK = threading.Lock()

# Guards the "only one live session at a time" constraint.
_SESSION_LOCK = threading.Lock()
_active_session: Optional[RecitationSession] = None
_active_session_qari_mode: bool = False
_active_session_correction_mode: str = "balanced"
# "standard" | "combined" (masterplan.md §4.2/§4.3) — set once at
# /session/start and read by every chunk in the WS stream + by the flush
# loop in _finalize_active_session(), same dispatch app.py's queue worker
# does via _parse_asr_engine(). "standard" preserves today's exact behavior.
_active_session_asr_engine: str = "standard"

_vad_model_singleton = None


def _trim_leading_trailing_silence(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim leading/trailing silence from a one-shot uploaded file before
    decoding.

    The live /session/*/stream path never has this problem: session_manager.py's
    VAD-based chunking only ever hands the ASR engines already speech-trimmed
    segments (see RecitationSession._emit_segment). /transcribe, being a
    stateless one-shot decode of whatever file was uploaded, has no such
    trimming — a test clip with a second or two of silence padding around
    the actual recitation goes straight into Groq/local as-is. Whisper
    (both Groq's hosted model and the local model) has a well-known
    tendency to hallucinate a repeat of the last phrase into that
    unexplained silence, which is exactly the "same ayah twice" symptom
    seen via Postman but never through the live app. This brings
    /transcribe's input in line with what the live path already guarantees,
    without touching the live path itself.
    """
    global _vad_model_singleton
    if not _VAD_AVAILABLE:
        return audio_np
    try:
        import torch
        if _vad_model_singleton is None:
            _vad_model_singleton = load_silero_vad()
        audio_tensor = torch.from_numpy(audio_np).float()
        timestamps = get_speech_timestamps(
            audio_tensor,
            _vad_model_singleton,
            sampling_rate=sample_rate,
            threshold=0.25,
            min_speech_duration_ms=150,
            min_silence_duration_ms=300,
        )
        if not timestamps:
            return audio_np
        pad = int(0.2 * sample_rate)
        start = max(0, timestamps[0]["start"] - pad)
        end = min(len(audio_np), timestamps[-1]["end"] + pad)
        return audio_np[start:end]
    except Exception as e:
        print(f"[API] Silence trim skipped ({e}); using untrimmed audio")
        return audio_np


# ---------------------------------------------------------------------------
# /transcribe — stateless file upload (mirrors the Gradio "File Upload" tab)
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    surah: int = Form(...),
    model_choice: str = Form("groq-whisper-large-v3"),
    correction_mode: str = Form("balanced"),
    expected_ayah: int = Form(1),
    asr_engine: str = Form("standard"),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio_np = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)[0].astype(np.float32)
        audio_np = _trim_leading_trailing_silence(audio_np, SAMPLE_RATE)

        harakaat_errors = None
        harakaat_error_count = 0

        if asr_engine == "combined":
            # Combined Mode (masterplan.md §1/§4.2) — Groq consonant backbone
            # + local turbo diacritics, same pipeline the live WS path uses,
            # just as a one-shot stateless decode instead of per-VAD-chunk.
            # Lazy imports: the local turbo model only loads on first actual
            # use of this path, never on server startup or standard calls.
            from hybrid_diacritic_pipeline import run_combined_transcription
            from harakaat_error_detector import detect_harakaat_errors

            with _INFERENCE_LOCK:
                rt_streamer.ensure_ayah_map_loaded()
                decode_result = run_combined_transcription(audio_np, SAMPLE_RATE)
                raw_text = decode_result["combined_text"]

                guard_result = guard_inference(
                    raw_text=raw_text,
                    ayah_map=rt_streamer._ayah_map,  # noqa: SLF001
                    surah=surah,
                    expected_ayah=expected_ayah,
                    lookahead=10,
                    window_back=5,
                    correction_mode=correction_mode,
                    allow_auto_correct=True,
                    allow_reference_replacement=True,
                    preserve_reciter=True,
                    use_sequence_match=True,
                    sequence_max_ayahs=8,
                )

                matched_key = guard_result.get("matched_key")
                if isinstance(matched_key, (tuple, list)) and len(matched_key) >= 2:
                    harakaat_result = detect_harakaat_errors(raw_text, int(matched_key[0]), int(matched_key[1]))
                    harakaat_errors = [
                        {"index": w.index, "predicted": w.predicted_word, "reference": w.reference_word, "status": w.status}
                        for w in harakaat_result.words if w.status == "harakaat_error"
                    ]
                    harakaat_error_count = harakaat_result.harakaat_error_count
        else:
            with _INFERENCE_LOCK:
                rt_streamer.set_model_choice(model_choice)
                rt_streamer._ensure_model_loaded()  # noqa: SLF001 (internal, same as app.py's load_models_once)
                raw_text = rt_streamer._decode_raw(audio_np)  # noqa: SLF001

                guard_result = guard_inference(
                    raw_text=raw_text,
                    ayah_map=rt_streamer._ayah_map,  # noqa: SLF001
                    surah=surah,
                    expected_ayah=expected_ayah,
                    lookahead=10,
                    window_back=5,
                    correction_mode=correction_mode,
                    allow_auto_correct=True,
                    allow_reference_replacement=True,
                    preserve_reciter=True,
                    use_sequence_match=True,
                    sequence_max_ayahs=8,
                )

        word_errors = []
        ref_text_for_words = guard_result.get("matched_ayah_text") or ""
        if ref_text_for_words:
            # Same word-level classifier (correct/minor/major/missing/extra)
            # your Qari Mode's strict correction logic already uses
            # internally (realtime_streamer.py's _apply_qari_word_scoring) —
            # reused here as-is, not reimplemented, so the client gets the
            # exact same per-word verdicts without duplicating that logic.
            hyp_for_annotation = correct_text_rules(raw_text, mode="balanced")
            word_errors = get_word_error_annotations(
                hyp_for_annotation,
                ref_text_for_words,
                confidence=guard_result.get("confidence"),
            )

        return {
            "raw_asr": raw_text,
            "corrected_text": guard_result.get("corrected_text", ""),
            "matched_ayah": guard_result.get("matched_ayah"),
            "matched_surah_ayah_id": (
                f"{surah}:{guard_result.get('matched_ayah')}"
                if surah and guard_result.get("matched_ayah")
                else None
            ),
            "matched_ayah_text": guard_result.get("matched_ayah_text"),
            "confidence": guard_result.get("confidence"),
            "confidence_level": guard_result.get("confidence_level"),
            "verdict": guard_result.get("verdict"),
            "cer": guard_result.get("cer"),
            "wer": guard_result.get("wer"),
            "harakaat_errors": harakaat_errors or [],
            "harakaat_error_count": harakaat_error_count,
            # Per-word verdicts against the matched ayah — status is one of
            # correct/minor/major/missing/extra/uncertain. Lets the client
            # drive TTS/correction UI directly without its own matching logic.
            "word_errors": word_errors,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Live session lifecycle
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    surah: int
    start_ayah: int = 1
    use_vad: bool = True
    model_choice: str = "groq-whisper-large-v3"
    correction_mode: str = "balanced"
    qari_mode: bool = False
    # "standard" (default, byte-identical to today) or "combined" (masterplan.md
    # §4.2/§4.3) — Groq + local-turbo harakaat-aware pipeline. Any other value
    # falls back to "standard" in start_session(), same tolerance app.py's
    # _parse_asr_engine() has for an unrecognized UI value.
    asr_engine: str = "standard"


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    global _active_session, _active_session_qari_mode, _active_session_correction_mode, _active_session_asr_engine

    with _SESSION_LOCK:
        if _active_session is not None:
            raise HTTPException(
                status_code=409,
                detail="A live session is already active. Stop it before starting a new one.",
            )

        engine = req.asr_engine if req.asr_engine in ("standard", "combined") else "standard"

        with _INFERENCE_LOCK:
            rt_streamer.set_model_choice(req.model_choice)
            session = rt_streamer.create_session(
                surah=req.surah,
                start_ayah=req.start_ayah,
                use_vad=req.use_vad,
            )

            if engine == "combined":
                # Force the local turbo model to load NOW, synchronously,
                # instead of on the first streamed audio chunk — same reason
                # app.py's start_live_session() preloads it (masterplan.md
                # §1): without this the first chunk after start eats the
                # multi-second HF model load on top of inference.
                from hybrid_diacritic_pipeline import preload_local_pipeline
                try:
                    preload_local_pipeline()
                except Exception as e:
                    print(f"[API] Combined Mode local model preload failed, will lazy-load on first chunk: {e}")

        if req.qari_mode:
            rt_streamer.correction_engine.reset()
            rt_streamer._vad_paused = False  # noqa: SLF001
            rt_streamer._last_qari_action = None  # noqa: SLF001

        _active_session = session
        _active_session_qari_mode = req.qari_mode
        _active_session_correction_mode = req.correction_mode
        _active_session_asr_engine = engine

        return {
            "session_id": session.session_id,
            "surah": session.surah,
            "start_ayah": session.start_ayah,
            "use_vad": session.use_vad,
            "asr_engine": engine,
        }


def _process_one_chunk(session: RecitationSession, chunk_audio, start: int, end: int):
    """Dispatch a single chunk to the standard or Combined Mode decode path,
    based on the active session's asr_engine — same branch app.py's queue
    worker does. Centralized here so the flush loop and the WS loop below
    can't drift out of sync with each other."""
    if _active_session_asr_engine == "combined":
        return rt_streamer.process_chunk_combined(
            session,
            chunk_audio,
            correction_mode=_active_session_correction_mode,
            qari_mode=_active_session_qari_mode,
            chunk_start_sample=start,
            chunk_end_sample=end,
        )
    return rt_streamer.process_chunk(
        session,
        chunk_audio,
        correction_mode=_active_session_correction_mode,
        qari_mode=_active_session_qari_mode,
        chunk_start_sample=start,
        chunk_end_sample=end,
    )


def _finalize_active_session() -> Optional[Dict[str, Any]]:
    """Flush + finalize whatever the active session is, and release the lock.
    Returns the summary JSON, or None if there was no active session."""
    global _active_session

    with _SESSION_LOCK:
        session = _active_session
        if session is None:
            return None

        # Flush any tail audio that hasn't been VAD-emitted yet.
        with _INFERENCE_LOCK:
            for chunk_audio, start, end in session.flush_remaining_audio():
                _process_one_chunk(session, chunk_audio, start, end)
            results_path = session.finalize()

        summary = session_summary_to_json(results_path, session)
        _active_session = None
        return summary


@app.post("/session/{session_id}/stop")
def stop_session(session_id: str):
    if _active_session is None or _active_session.session_id != session_id:
        raise HTTPException(status_code=404, detail="No active session with that id.")
    summary = _finalize_active_session()
    return summary


@app.websocket("/session/{session_id}/stream")
async def stream_session(websocket: WebSocket, session_id: str):
    """
    Protocol (JSON text frames — friendlier than binary for React Native's
    WebSocket implementation):

      Client -> Server:
        {"type": "audio", "pcm16_base64": "<base64-encoded little-endian
                 int16 PCM, mono, 16kHz>"}
        {"type": "stop"}

      Server -> Client:
        {"type": "chunk", ...}          (one per VAD-detected ayah segment)
        {"type": "qari_action", ...}    (only when qari_mode is on)
        {"type": "session_summary", ...} (once, right before closing)
        {"type": "error", "message": "..."}
    """
    await websocket.accept()

    if _active_session is None or _active_session.session_id != session_id:
        await websocket.send_json(error_to_json("No active session with that id."))
        await websocket.close()
        return

    session = _active_session

    try:
        import base64

        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "stop":
                break

            if msg_type != "audio":
                await websocket.send_json(error_to_json(f"Unknown message type: {msg_type}"))
                continue

            raw_bytes = base64.b64decode(msg["pcm16_base64"])
            pcm16 = np.frombuffer(raw_bytes, dtype="<i2")
            audio_fragment = (pcm16.astype(np.float32) / 32768.0)

            # DIAGNOSTIC: confirms whether real audio signal is arriving at
            # all, before VAD/ASR even get involved. peak_amp near 0.0 means
            # the client is sending silence/empty buffers — not a VAD or
            # server bug. A real recorded voice typically peaks well above
            # 0.02–0.05 even in quiet rooms.
            peak_amp = float(np.abs(audio_fragment).max()) if len(audio_fragment) else 0.0
            print(f"[WS audio] {len(pcm16)} samples ({len(pcm16)/16000:.2f}s) | peak_amp={peak_amp:.5f}"
                  + ("  <-- SILENCE, check client audio capture/encoding" if peak_amp < 0.01 else ""))

            ready_chunks = session.feed_audio(audio_fragment)

            for chunk_audio, start, end in ready_chunks:
                with _INFERENCE_LOCK:
                    result = _process_one_chunk(session, chunk_audio, start, end)
                await websocket.send_json(chunk_result_to_json(result, session))

                if _active_session_qari_mode:
                    if _active_session_asr_engine == "combined":
                        # Combined Mode: rt_streamer._last_qari_action is the
                        # actual dict correction_engine.process_verdict()
                        # returned for this chunk — covers "pause" (word
                        # error) AND "hint" (harakaat-only slip,
                        # masterplan.md §4.4/Phase 5), which the pending-
                        # corrections check below can't see since a harakaat
                        # hint never touches _pending_wrong_words.
                        action_json = qari_action_to_json(rt_streamer._last_qari_action)  # noqa: SLF001
                        if action_json:
                            # Relay every action (pause/hint/continue/retry/
                            # skip), not just pause/hint. correction_engine.py
                            # legitimately returns "continue" (correction
                            # verified, resume listening) and "retry" (still
                            # wrong, replay correction) too — a client that
                            # paused its mic on "pause" needs one of those to
                            # know when it's safe to resume, or it stays
                            # paused forever.
                            await websocket.send_json(action_json)
                    else:
                        pending = rt_streamer.correction_engine.get_pending_corrections()
                        action_json = qari_action_to_json(
                            {"action": "pause", "wrong_words": pending} if pending else None
                        )
                        if action_json:
                            await websocket.send_json(action_json)

        summary = _finalize_active_session()
        if summary:
            await websocket.send_json(summary)
        await websocket.close()

    except WebSocketDisconnect:
        # Client dropped without sending "stop" (e.g. call ended, app
        # backgrounded). Finalize server-side so the session lock isn't
        # left stuck forever.
        _finalize_active_session()

    except Exception as exc:
        try:
            await websocket.send_json(error_to_json(str(exc)))
        finally:
            _finalize_active_session()
            await websocket.close()


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "active_session": _active_session.session_id if _active_session else None})
