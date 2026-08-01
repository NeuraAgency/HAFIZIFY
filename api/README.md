# Hafizify Mobile API

Additive FastAPI layer for the Expo/React Native app. Imports the same
engine modules `app.py` (desktop Gradio) uses — `realtime_streamer.py`,
`session_manager.py`, `fyp_model/quran_guard.py` — unmodified. Nothing in
this folder is imported by `app.py`, and nothing in `app.py` is imported
by this folder except read-only engine modules. The desktop app is
unaffected whether this process is running or not.

## Run

From the `hafizify/` project root, same venv as the desktop app:

```bash
pip install -r requirements.txt   # picks up fastapi/uvicorn (additive lines at the bottom)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Constraint: one live session at a time

`RealtimeStreamer` (in `realtime_streamer.py`) holds session-shaped state
on itself — `correction_engine`, `_decode_context`, `_vad_paused` — not
just on `RecitationSession`. That's true today for the desktop app too
(`_active_session` in `app.py` is a single global). This API keeps that
same constraint rather than editing `realtime_streamer.py` to make it
multi-session-safe, since that file is core to what's already working.

Practically: `POST /session/start` returns `409` if a session is already
active. `POST /transcribe` (file upload) has no such limit — it's a
stateless decode.

If concurrent mobile users become a real requirement later, the fix is to
split `RealtimeStreamer` into a shared read-only model-holder + a
per-session `CorrectionEngine`/`_decode_context` — that's a deliberate,
separate change to make later, not something to bolt on here.

## Endpoints

### `POST /transcribe`
multipart form: `file`, `surah` (required — the user picks this in-app),
`model_choice` (default `groq-whisper-large-v3`), `correction_mode`,
`expected_ayah`.
Stateless — safe to call concurrently. Mirrors the Gradio "File Upload" tab.

### `POST /session/start`
JSON: `{surah, start_ayah, use_vad, model_choice, correction_mode, qari_mode}`
`surah` is required — the app's surah picker is the source of truth, so the
server no longer runs surah auto-detection for API sessions. `model_choice`
defaults to `groq-whisper-large-v3`.
Returns `{session_id, surah, start_ayah, use_vad}`. `409` if a session is already active.

### `WS /session/{session_id}/stream`
JSON text frames (not binary — more reliable across React Native's WebSocket implementation):

Client → Server:
```json
{"type": "audio", "pcm16_base64": "<base64 little-endian int16 PCM, mono, 16kHz>"}
{"type": "stop"}
```

Server → Client:
```json
{"type": "chunk", "raw_asr": "...", "corrected_text": "...", "matched_ayah": 5, "verdict": "ok", ...}
{"type": "qari_action", "action": "pause", "wrong_words": [...]}
{"type": "session_summary", "merged_transcript": "...", ...}
{"type": "error", "message": "..."}
```

A chunk is emitted whenever the server-side Silero VAD (already built into
`session_manager.py`) detects an ayah-boundary pause — not on a fixed
timer, and not something the client controls. Client audio-chunking
interval (how often you flush from the mic library) is irrelevant to this
— send whatever interval is convenient, e.g. every 200-300ms.

If the socket disconnects without a `"stop"` message, the server finalizes
the session anyway so the single-session lock doesn't get stuck.

### `POST /session/{session_id}/stop`
REST fallback to finalize a session (in case you're not using the WS for
teardown, or need to force-close after a bad disconnect).

### `GET /health`
`{"status": "ok", "active_session": "<id or null>"}`

## Mobile-side pipeline (Expo)

1. `react-native-live-audio-stream` opens the mic and emits base64 PCM16
   chunks on an interval. Configure it for 16kHz mono if the library
   supports specifying sample rate directly — saves a resample step
   either side. Requires a dev client / prebuild, not Expo Go.
2. `POST /session/start` with the surah the user picked in-app, get `session_id`.
3. Open `WS /session/{session_id}/stream`, forward every audio chunk from
   the mic library straight through as `{"type":"audio","pcm16_base64":...}`
   — don't buffer or try to do VAD/silence-detection client-side, the
   server already does that.
4. Render `chunk` messages as they arrive (raw_asr / corrected_text /
   matched_ayah / verdict).
5. On stop button: send `{"type":"stop"}`, wait for `session_summary`,
   close the mic stream.
