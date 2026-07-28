"""
Groq Whisper Transcriber
-------------------------
Thin wrapper around the Groq REST API for the whisper-large-v3-turbo model.
Handles both file-path inputs (upload tab) and raw in-memory numpy audio
arrays (live streaming tab) by writing a temporary WAV to disk before
sending to the API.

Usage
-----
    from groq_transcriber import GroqTranscriber
    tr = GroqTranscriber()               # uses GROQ_API_KEY env var
    text = tr.transcribe_file("path/to/audio.wav")
    text = tr.transcribe_array(audio_np, sample_rate=16000)
"""

import io
import os
import tempfile
import wave
import numpy as np

# No hardcoded default — API key must be set via GROQ_API_KEY env var
_MODEL = "whisper-large-v3-turbo"
_LANGUAGE = "ar"  # Arabic


def _write_wav_bytes(audio_np: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert a float32 numpy array to WAV bytes (PCM 16-bit)."""
    audio_np = np.clip(audio_np.astype(np.float32), -1.0, 1.0)
    pcm = (audio_np * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class GroqTranscriber:
    """Wraps the Groq Audio Transcription API for whisper-large-v3-turbo."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY env var or pass api_key=."
            )
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )
        self._client = Groq(api_key=key)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file on disk.
        Returns the raw Arabic transcript string.
        """
        with open(audio_path, "rb") as f:
            result = self._client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model=_MODEL,
                language=_LANGUAGE,
                response_format="text",
                temperature=0.0,
            )
        return str(result).strip()

    def transcribe_array(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe a 16 kHz float32 numpy audio array.
        Writes a temporary WAV, uploads to Groq, then cleans up.
        Returns the raw Arabic transcript string.
        """
        wav_bytes = _write_wav_bytes(audio_np, sample_rate)
        result = self._client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=_MODEL,
            language=_LANGUAGE,
            response_format="text",
            temperature=0.0,
        )
        return str(result).strip()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised)
# ---------------------------------------------------------------------------
_instance: GroqTranscriber | None = None


def get_groq_transcriber() -> GroqTranscriber:
    """Return a cached GroqTranscriber instance (created on first call)."""
    global _instance
    if _instance is None:
        _instance = GroqTranscriber()
    return _instance
