"""
JSON formatters for the mobile API.

This is the mobile-side twin of `live_display_formatter.py`. That module
formats ChunkResult objects as HTML for the Gradio UI; this module formats
the exact same ChunkResult objects as plain JSON-serialisable dicts for a
mobile client. Nothing here mutates or depends on Gradio-specific state —
it only reads fields off ChunkResult / RecitationSession.
"""
from typing import Any, Dict, Optional

from session_manager import ChunkResult, RecitationSession


def chunk_result_to_json(result: ChunkResult, session: RecitationSession) -> Dict[str, Any]:
    """Convert a single ChunkResult into a JSON-safe dict for the mobile client."""
    return {
        "type": "chunk",
        "chunk_index": result.chunk_index,
        "start_time_s": result.start_time_s,
        "end_time_s": result.end_time_s,
        "duration_s": result.duration_s,
        "raw_asr": result.raw_asr,
        "corrected_text": result.corrected_text,
        "matched_ayah": result.matched_ayah,
        "matched_surah_ayah_id": result.matched_surah_ayah_id,
        "matched_ayah_text": result.matched_ayah_text,
        "cer": result.cer,
        "wer": result.wer,
        "coverage": result.coverage,
        "confidence": result.confidence,
        "confidence_level": result.confidence_level,
        "verdict": result.verdict,
        "errors": result.errors,
        "surah_lock_state": result.surah_lock_state,
        # session-level tracking, echoed per chunk so the client never has
        # to reconstruct it from a stream of partial updates
        "session": {
            "surah": session.surah,
            "current_ayah": session.current_ayah,
        },
    }


def qari_action_to_json(correction_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pass through the Qari-mode correction engine's action, if any, as JSON."""
    if not correction_result:
        return None
    return {
        "type": "qari_action",
        "action": correction_result.get("action"),
        "message": correction_result.get("message"),
        "wrong_words": correction_result.get("wrong_words"),
    }


def session_summary_to_json(results_path: str, session: RecitationSession) -> Dict[str, Any]:
    """Final payload sent when a session is stopped/finalized."""
    return {
        "type": "session_summary",
        "session_id": session.session_id,
        "surah": session.get_detected_surah(),
        "start_ayah": session.start_ayah,
        "final_ayah": session.current_ayah,
        "total_duration_s": round(session.total_duration_s, 2),
        "num_chunks": len(session.chunk_results),
        "merged_transcript": session.get_merged_transcript(),
        "results_json_path": results_path,
    }


def error_to_json(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": message}
