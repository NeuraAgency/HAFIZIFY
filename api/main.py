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
from session_manager import RecitationSession, SAMPLE_RATE  # noqa: E402
from fyp_model.quran_guard import guard_inference  # noqa: E402

from api.formatters import (  # noqa: E402
    chunk_result_to_json,
    error_to_json,
    qari_action_to_json,
    session_summary_to_json,
)

app = FastAPI(title="Hafizify Mobile API")

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
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_path = None
    try:
        data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        audio_np = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)[0].astype(np.float32)

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


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    global _active_session, _active_session_qari_mode, _active_session_correction_mode

    with _SESSION_LOCK:
        if _active_session is not None:
            raise HTTPException(
                status_code=409,
                detail="A live session is already active. Stop it before starting a new one.",
            )

        with _INFERENCE_LOCK:
            rt_streamer.set_model_choice(req.model_choice)
            session = rt_streamer.create_session(
                surah=req.surah,
                start_ayah=req.start_ayah,
                use_vad=req.use_vad,
            )

        _active_session = session
        _active_session_qari_mode = req.qari_mode
        _active_session_correction_mode = req.correction_mode

        return {
            "session_id": session.session_id,
            "surah": session.surah,
            "start_ayah": session.start_ayah,
            "use_vad": session.use_vad,
        }


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
                rt_streamer.process_chunk(
                    session,
                    chunk_audio,
                    correction_mode=_active_session_correction_mode,
                    qari_mode=_active_session_qari_mode,
                    chunk_start_sample=start,
                    chunk_end_sample=end,
                )
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

            ready_chunks = session.feed_audio(audio_fragment)

            for chunk_audio, start, end in ready_chunks:
                with _INFERENCE_LOCK:
                    result = rt_streamer.process_chunk(
                        session,
                        chunk_audio,
                        correction_mode=_active_session_correction_mode,
                        qari_mode=_active_session_qari_mode,
                        chunk_start_sample=start,
                        chunk_end_sample=end,
                    )
                await websocket.send_json(chunk_result_to_json(result, session))

                if _active_session_qari_mode:
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
