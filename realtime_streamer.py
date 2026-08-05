"""
Real-Time Streamer — Gradio ↔ Model Bridge
--------------------------------------------
Bridges Gradio's streaming microphone input with the Whisper model
and correction pipelines. Handles:
  - Audio resampling from browser sample rate to 16 kHz
  - Feeding audio into the session manager's ring buffer
  - Running inference on each extracted chunk
  - Formatting results for the live Gradio UI
  - Optional beam search for real-time mode

Supported models:
  - whisper-base-quran-lora      : LoRA fine-tuned on top of tarteel-ai base
  - whisper-medium-quran-full    : tarbiyah-ai fully merged Quran-trained whisper-medium
  - groq-whisper-large-v3        : Cloud ASR via Groq API (fastest, no local GPU needed)
"""

import os
import sys
import time
import difflib
try:
    from rapidfuzz import fuzz as _rfuzz_rt
    _RT_HAS_RAPIDFUZZ = True
except ImportError:
    _rfuzz_rt = None
    _RT_HAS_RAPIDFUZZ = False
import numpy as np
import torch
import torchaudio
from typing import Optional, Dict, Any, List

from quran_trie import QuranTrie

# Ensure fyp_model is importable
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FYP_DIR = os.path.join(_BASE_DIR, "fyp_model")
if _FYP_DIR not in sys.path:
    sys.path.insert(0, _FYP_DIR)

from peft import PeftModel
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
from faster_whisper import WhisperModel
from session_manager import RecitationSession, ChunkResult, SAMPLE_RATE

from fyp_model.quran_guard import (
    correct_text_rules,
    guard_inference,
    get_word_error_annotations,
    get_word_error_annotations_best_of,
    load_all_ayat_json,
    normalize_arabic,
)
from surah_detector import SurahCandidate, SurahDetector, SurahLockManager
from correction_engine import CorrectionEngine
from api_client import transcribe_via_api

# Whisper hard limit: 30 seconds at 16kHz
_WHISPER_MAX_SAMPLES = 30 * SAMPLE_RATE  # 480,000 samples

# Anti-hallucination gates
_MIN_CHUNK_SAMPLES = int(1.5 * SAMPLE_RATE)   # skip chunks shorter than 1.5s
_MIN_SPEECH_RMS    = 0.005                     # skip near-silent chunks (RMS threshold)

_TAWWUZ_TEXT = "أعوذ بالله من الشيطان الرجيم"
_BASMALA_TEXT = "بسم الله الرحمن الرحيم"


def _fold_invocation_token(token: str) -> str:
    token = normalize_arabic(token)
    token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    token = token.replace("ى", "ي").replace("ة", "ه")
    return token


_TAWWUZ_TOKENS = [_fold_invocation_token(w) for w in _TAWWUZ_TEXT.split()]
_BASMALA_TOKENS = [_fold_invocation_token(w) for w in _BASMALA_TEXT.split()]


def _token_close(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a)):
        return True
    if _RT_HAS_RAPIDFUZZ:
        return _rfuzz_rt.ratio(a, b) >= 72.0
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.72


def _leading_invocation_len(norm_tokens: list[str], expected: list[str]) -> int:
    if not norm_tokens or not expected:
        return 0

    max_len = min(len(norm_tokens), len(expected))
    matched = 0
    for idx in range(max_len):
        if _token_close(norm_tokens[idx], expected[idx]):
            matched += 1
            continue
        break

    if matched == len(expected):
        return matched
    if matched >= 2 and matched == len(norm_tokens):
        return matched
    return 0


def _strip_leading_invocations(text: str, strip_basmala: bool = True) -> str:
    tokens = str(text or "").split()
    norm_tokens = [_fold_invocation_token(t) for t in tokens]

    tawwuz_len = _leading_invocation_len(norm_tokens, _TAWWUZ_TOKENS)
    if tawwuz_len:
        tokens = tokens[tawwuz_len:]
        norm_tokens = norm_tokens[tawwuz_len:]

    if strip_basmala:
        basmala_len = _leading_invocation_len(norm_tokens, _BASMALA_TOKENS)
        if basmala_len:
            tokens = tokens[basmala_len:]

    return " ".join(tokens).strip()


def _strip_trailing_partial_token(hyp_text: str, ref_text: str) -> str:
    """Drop a likely incomplete final ASR token before realtime correction checks."""
    hyp_tokens = str(hyp_text or "").split()
    if not hyp_tokens or not ref_text:
        return str(hyp_text or "").strip()

    last_norm = normalize_arabic(hyp_tokens[-1])
    if not last_norm:
        return " ".join(hyp_tokens[:-1]).strip()

    ref_norm_tokens = normalize_arabic(ref_text).split()
    if last_norm in ref_norm_tokens:
        return " ".join(hyp_tokens).strip()

    is_prefix = any(
        ref_tok.startswith(last_norm) and len(last_norm) < len(ref_tok)
        for ref_tok in ref_norm_tokens
    )
    if is_prefix and len(last_norm) <= 4:
        return " ".join(hyp_tokens[:-1]).strip()

    return " ".join(hyp_tokens).strip()


def _drop_trailing_future_missing(annotations):
    """Ignore reference words not reached yet by the current realtime chunk."""
    trimmed = list(annotations or [])
    while trimmed and trimmed[-1].get("status") == "missing" and not trimmed[-1].get("word"):
        trimmed.pop()
    return trimmed


def _apply_minor_reference_fixes(annotations) -> str:
    """Build display text with close ASR variants replaced by reference words."""
    words = []
    for ann in annotations or []:
        status = ann.get("status")
        word = ann.get("word") or ""
        ref_word = ann.get("reference") or ""

        if status == "minor" and ref_word:
            words.append(ref_word)
        elif word:
            words.append(word)

    return " ".join(words).strip()


def _correction_span_from_annotations(annotations, start_idx: int, end_idx: int) -> dict:
    selected = list(annotations[start_idx : end_idx + 1])
    refs = [a.get("reference") for a in selected if a.get("reference")]
    ref_indexes = [
        int(a["ref_index"])
        for a in selected
        if a.get("ref_index") is not None and a.get("reference")
    ]
    text = " ".join(str(ref) for ref in refs).strip()
    return {
        "text": text,
        "ref_word_start": min(ref_indexes) if ref_indexes else None,
        "ref_word_end": max(ref_indexes) if ref_indexes else None,
    }


def _wrong_words_and_spans_from_word_errors(annotations: list) -> tuple:
    """Group consecutive major/missing word-error annotations into teachable
    phrases. Shared by _apply_qari_word_scoring() (Combined Mode, local
    decode) and process_chunk_api() (masterplan.md §9 — remote /transcribe)
    so this grouping logic exists in exactly one place rather than being
    copied a third time. Returns (wrong_words, correction_spans).
    """
    wrong_words = []
    correction_spans = []
    if not annotations:
        return wrong_words, correction_spans

    first_teachable_idx = next(
        (
            i
            for i, a in enumerate(annotations)
            if a.get("status") in ("major", "missing")
            and a.get("reference")
        ),
        None,
    )
    if first_teachable_idx is None:
        return wrong_words, correction_spans

    phrase_parts = []
    phrase_start_idx = first_teachable_idx
    for i, ann in enumerate(annotations[first_teachable_idx:], first_teachable_idx):
        if i > first_teachable_idx and ann.get("status") == "correct":
            break
        status = ann.get("status")
        ref_word = ann.get("reference")
        if status in ("major", "missing") and ref_word:
            phrase_parts.append(ref_word)
            continue
        if status not in ("extra", "uncertain") and phrase_parts:
            wrong_words.append(" ".join(phrase_parts))
            correction_spans.append(
                _correction_span_from_annotations(annotations, phrase_start_idx, i - 1)
            )
            phrase_parts = []
    if phrase_parts:
        wrong_words.append(" ".join(phrase_parts))
        correction_spans.append(
            _correction_span_from_annotations(
                annotations,
                phrase_start_idx,
                min(len(annotations) - 1, phrase_start_idx + len(phrase_parts) - 1),
            )
        )
    return wrong_words, correction_spans


def _apply_qari_word_scoring(guard_result: dict, analysis_text: str) -> None:
    """Qari Mode word-level teachable-phrase scoring for Combined Mode
    (masterplan.md changelog, 2026-08-02 — wiring qari_mode into
    process_chunk_combined()).

    Mirrors process_chunk()'s qari_mode branch so Combined Mode gets the
    same wrong-word/correction-span behavior Standard Qari Mode already
    has. Lives as its own function rather than being extracted out of
    process_chunk() because process_chunk()'s body is on masterplan.md's
    "do not touch" list (§5) — this is a fresh copy for the new caller,
    not a refactor of the existing one. Mutates guard_result in place.
    """
    ref_text = guard_result.get("matched_ayah_text") or ""
    wrong_words = []
    correction_spans = []
    if ref_text:
        hyp_text = correct_text_rules(analysis_text, mode="balanced")
        detection_text = _strip_trailing_partial_token(hyp_text, ref_text)
        annotations = get_word_error_annotations(detection_text, ref_text, confidence=None)
        annotations = _drop_trailing_future_missing(annotations)
        statuses = {a.get("status") for a in annotations}
        minor_fixed_text = _apply_minor_reference_fixes(annotations)
        if minor_fixed_text:
            guard_result["corrected_text"] = minor_fixed_text
            guard_result["display_text"] = minor_fixed_text
        elif hyp_text:
            guard_result["corrected_text"] = hyp_text
            guard_result["display_text"] = hyp_text

        wrong_words, correction_spans = _wrong_words_and_spans_from_word_errors(annotations)

        if statuses - {"correct", "minor", None}:
            guard_result["verdict"] = "error"
        elif "minor" in statuses:
            guard_result["verdict"] = "minor"
        elif statuses:
            guard_result["verdict"] = "ok"

    guard_result["wrong_words"] = wrong_words
    guard_result["correction_spans"] = correction_spans


def _force_expected_ayah_if_behind(guard_result: dict, session: RecitationSession, ayah_map) -> None:
    expected_ayah = getattr(session, "current_ayah", None)
    surah = getattr(session, "surah", None)
    matched_ayah = guard_result.get("matched_ayah")

    if expected_ayah is None or matched_ayah is None or surah is None:
        return

    try:
        matched_ayah = int(matched_ayah)
        expected_ayah = int(expected_ayah)
        surah = int(surah)
    except Exception:
        return

    if matched_ayah >= expected_ayah:
        return

    fallback_text = ayah_map.get((surah, expected_ayah)) if ayah_map else None
    if not fallback_text:
        return

    guard_result["matched_ayah"] = expected_ayah
    guard_result["matched_key"] = (surah, expected_ayah)
    guard_result["matched_start_ayah"] = expected_ayah
    guard_result["matched_end_ayah"] = expected_ayah
    guard_result["matched_ayah_text"] = fallback_text
    guard_result["is_sequence_match"] = False


def _flag_skipped_ayah_if_forward_jump(guard_result: dict, session: RecitationSession, ayah_map, correction_engine=None) -> None:
    """Symmetric to _force_expected_ayah_if_behind(): if the matched ayah
    jumped forward by 2 or more (the reciter skipped at least one full ayah
    without saying it), force the expected ayah back into the result and
    mark it an error — same override style the "behind" case already uses.
    A jump of exactly +1 is normal forward progress and is left untouched.
    """
    expected_ayah = getattr(session, "current_ayah", None)
    surah = getattr(session, "surah", None)
    matched_ayah = guard_result.get("matched_ayah")

    if expected_ayah is None or matched_ayah is None or surah is None:
        return

    try:
        matched_ayah = int(matched_ayah)
        expected_ayah = int(expected_ayah)
        surah = int(surah)
    except Exception:
        return

    if matched_ayah < expected_ayah + 2:
        return  # same ayah or normal +1 progress — not a skip

    fallback_text = ayah_map.get((surah, expected_ayah)) if ayah_map else None
    if not fallback_text:
        return

    if correction_engine is not None and expected_ayah not in correction_engine.skipped_ayahs:
        correction_engine.skipped_ayahs.append(expected_ayah)

    guard_result["matched_ayah"] = expected_ayah
    guard_result["matched_key"] = (surah, expected_ayah)
    guard_result["matched_start_ayah"] = expected_ayah
    guard_result["matched_end_ayah"] = expected_ayah
    guard_result["matched_ayah_text"] = fallback_text
    guard_result["is_sequence_match"] = False
    guard_result["verdict"] = "error"


def _is_repetition_loop(text: str, repeat_threshold: int = 4) -> bool:
    tokens = normalize_arabic(text).split()
    if len(tokens) < repeat_threshold:
        return False
    streak = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            streak += 1
            if streak >= repeat_threshold:
                return True
        else:
            streak = 1
    return False


class RealtimeStreamer:
    """Manages model loading + real-time chunk inference for one app lifecycle."""

    def __init__(self, surah_detector: Optional[SurahDetector] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ayah_json_path = os.path.join(_BASE_DIR, "fyp_model", "all_ayat.json")
        self.recordings_dir = os.path.join(_BASE_DIR, "recordings")
        os.makedirs(self.recordings_dir, exist_ok=True)

        # Model selection:
        # "whisper-base-quran-lora"    → LoRA adapter on tarteel-ai base
        # "whisper-medium-quran-full"  → tarbiyah-ai fully merged whisper-medium
        self._model_choice = "whisper-base-quran-lora"  # default

        # Lazy-loaded model state
        self._processor = None
        self._model = None
        self._model_type = "whisper"
        self._ayah_map = None
        self._surah_detector = surah_detector
        self._forced_decoder_ids = None

        # Optional viterbi pipeline for final session re-decode
        self._viterbi_pipeline = None
        self._quran_trie: Optional[QuranTrie] = None
        self._decode_context = {"surah": None, "ayah": None}  # updated before each decode

        # Beam search setting
        self._use_beam_realtime = False
        self._beam_width = 5
        self._is_faster_whisper = False
        
        self._vad_paused = False
        self.correction_engine = CorrectionEngine(on_state_change=self._on_correction_state_change)
        # Last dict returned by correction_engine.process_verdict() from
        # process_chunk_combined() (Qari Mode only). Combined Mode's WS API
        # layer needs this to relay HARAKAAT_HINT actions to the mobile
        # client — Standard mode's existing get_pending_corrections()-based
        # relay doesn't cover that action, since a harakaat hint never adds
        # anything to _pending_wrong_words. None until the first Combined +
        # Qari Mode chunk runs; not touched by process_chunk() (standard).
        self._last_qari_action: Optional[dict] = None

    def _on_correction_state_change(self, state: str):
        print(f"[CorrectionEngine] State → {state}")
        if state in ("VERIFYING", "LISTENING"):
            self._vad_paused = False  # reopen mic after correction or state reset
        elif state == "CORRECTING":
            self._vad_paused = True   # pause mic while correction TTS is playing

    def set_model_choice(self, model_choice: str):
        """Set which model to use. Forces reload if different from current."""
        # Remap legacy model names to supported ones
        if model_choice == "combined_model":
            model_choice = "whisper-base-quran-lora"

        if model_choice not in self._MODEL_REGISTRY:
            print(f"[RealtimeStreamer] Unknown model '{model_choice}', defaulting to whisper-base-quran-lora")
            model_choice = "whisper-base-quran-lora"

        if model_choice != self._model_choice:
            self._model_choice = model_choice
            self._model = None
            self._processor = None
            print(f"[RealtimeStreamer] Model choice changed to: {model_choice}")

    def set_beam_search(self, enabled: bool, beam_width: int = 5):
        """Enable/disable beam search for real-time decoding."""
        self._use_beam_realtime = enabled
        self._beam_width = max(1, beam_width)

    # Maps model_choice keys to HuggingFace model IDs or local folder names.
    # Local folders (relative to _BASE_DIR) take priority if they exist on disk.
    _MODEL_REGISTRY = {
        "whisper-base-quran-lora": {
            "type": "whisper",
            "local": "whisper-base-quran-lora-ct2",
            "hf": "KheemP/whisper-base-quran-lora",
        },
        "whisper-medium-quran-full": {
            "type": "whisper",
            "local": "tarbiyah-ai-whisper-medium-ct2",
            "hf": "Habib-HF/tarbiyah-ai-whisper-medium-merged",
        },
        "whisper-medium-quran-lora": {
            "type": "whisper_lora",
            "merged_local": "whisper-medium-quran-lora-merged",
            "base_local": "whisper-medium",
            "base_hf": "openai/whisper-medium",
            "adapter_local": "quran-lora-whisper-medium-epoch-1",
            "adapter_hf": "omartariq612/quran-lora-whisper-medium-epoch-1",
        },
        # ── Groq cloud ASR ────────────────────────────────────────────────────
        "groq-whisper-large-v3": {
            "type": "groq",
            # No local path — always uses the Groq REST API
        },
        # ── Turbo Quran LoRA (harakat/tashkeel output) ─────────────────────────
        # Full merged HF model (not an adapter) — loaded directly via
        # WhisperForConditionalGeneration, no PEFT merge needed.
        "whisper-l-v3-turbo-quran-lora": {
            "type": "transformers_whisper",
            "local": "whisper-l-v3-turbo-quran-lora-dataset-mix",
            "hf": "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix",
        },
    }

    def _resolve_model_path(self, model_choice: str) -> str:
        """Return a local path if it exists, otherwise the HuggingFace model ID."""
        info = self._MODEL_REGISTRY.get(model_choice, self._MODEL_REGISTRY["whisper-base-quran-lora"])
        local_key = info.get("local") or info.get("merged_local")
        
        if local_key:
            # Handle potential forward slashes in local_key for cross-platform compatibility
            path_parts = local_key.replace("\\", "/").split("/")
            local_path = os.path.join(_BASE_DIR, *path_parts)
            if os.path.isdir(local_path):
                print(f"[RealtimeStreamer] Using local model folder: {local_path}")
                return local_path
            
        hf_id = info.get("hf")
        if hf_id:
            print(f"[RealtimeStreamer] Local model not found — downloading from HuggingFace: {hf_id}")
            return hf_id
            
        raise FileNotFoundError(
            f"Model '{model_choice}' not found locally and no HuggingFace fallback available."
        )

    def _resolve_local_or_hf(self, local_folder: str, hf_id: str) -> str:
        local_path = os.path.join(_BASE_DIR, local_folder)
        if os.path.isdir(local_path):
            print(f"[RealtimeStreamer] Using local folder: {local_path}")
            return local_path
        print(f"[RealtimeStreamer] Local folder '{local_path}' not found — downloading from HuggingFace: {hf_id}")
        return hf_id

    def _get_ct2_threading_params(self) -> dict:
        """Determine optimal CPU/device threading and compute type for CTranslate2 / faster-whisper."""
        import multiprocessing
        env_threads = os.getenv("CT2_CPU_THREADS")
        env_inter = os.getenv("CT2_INTER_THREADS")
        env_compute = os.getenv("CT2_COMPUTE_TYPE")

        avail_cores = multiprocessing.cpu_count()
        if env_threads:
            cpu_threads = int(env_threads)
        else:
            # Physical core count preference (e.g. 6 on 12-thread i7) to prevent OpenMP thrashing
            cpu_threads = max(1, avail_cores // 2 if avail_cores >= 8 else avail_cores - 1)

        inter_threads = int(env_inter) if env_inter else 1

        if env_compute:
            compute_type = env_compute
        elif self.device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8_float32"

        print(f"[RealtimeStreamer] CTranslate2 config: device={self.device}, compute_type={compute_type}, "
              f"cpu_threads={cpu_threads}/{avail_cores}, inter_threads={inter_threads}")

        return {
            "device": self.device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "num_workers": inter_threads,
        }

    def _ensure_model_loaded(self):
        """Load model + processor + ayah map if not already loaded."""
        if self._model is not None:
            return

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_info = self._MODEL_REGISTRY.get(self._model_choice, self._MODEL_REGISTRY["whisper-base-quran-lora"])
        self._model_type = model_info["type"]

        # ── Groq cloud model — no local weights to load ───────────────────
        if self._model_type == "groq":
            from groq_transcriber import get_groq_transcriber
            self._model = get_groq_transcriber()  # acts as the "model" object
            self._processor = None
            self._forced_decoder_ids = None
            self._is_faster_whisper = False
            model_path = "groq-api:whisper-large-v3"
            print(f"[RealtimeStreamer] Groq whisper-large-v3 ready (cloud API).")
            # Jump straight to shared post-load logic
            if os.path.isfile(self.ayah_json_path):
                self._ayah_map = load_all_ayat_json(self.ayah_json_path)
                print(f"[RealtimeStreamer] Loaded {len(self._ayah_map)} ayahs")
            if self._surah_detector is None and os.path.isfile(self.ayah_json_path):
                try:
                    self._surah_detector = SurahDetector(self.ayah_json_path)
                except Exception as e:
                    print(f"[RealtimeStreamer] Surah detector not available: {e}")
            # HybridViterbiPipeline construction moved to _ensure_viterbi_pipeline()
            # (2026-08-02, Claude/chat, per Hamza's "we don't even use kenlm"
            # catch) — it was being built here unconditionally (n-gram index
            # over 6235 ayahs + kenlm LM load) even though its only consumer,
            # decode_full_session(), is unreachable from the current live flow
            # (see stop_live_session()'s own comment in app.py: the
            # full-session re-decode step was already removed). Now built
            # lazily on first actual call to decode_full_session() instead.
            return

        if self._model_type == "wav2vec2":
            model_path = self._resolve_model_path(self._model_choice)
            print(f"[RealtimeStreamer] Loading model '{self._model_choice}' from {model_path}...")
            self._model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
            self._processor = Wav2Vec2Processor.from_pretrained(model_path)
            self._forced_decoder_ids = None
        elif self._model_type == "whisper_lora":
            merged_local = os.path.join(_BASE_DIR, model_info["merged_local"])
            if os.path.isdir(merged_local):
                model_path = merged_local
                print(f"[RealtimeStreamer] Loading merged model from {merged_local}...")
                ct2_params = self._get_ct2_threading_params()
                self._model = WhisperModel(model_path, **ct2_params)
                self._processor = None
                self._forced_decoder_ids = None
                self._is_faster_whisper = True
            else:
                base_path = self._resolve_local_or_hf(model_info["base_local"], model_info["base_hf"])
                adapter_path = self._resolve_local_or_hf(model_info["adapter_local"], model_info["adapter_hf"])
                model_path = adapter_path

                print(f"[RealtimeStreamer] Loading LoRA base from {base_path} and adapter from {adapter_path}...")
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    base_path,
                    torch_dtype=dtype,
                )
                self._model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
                self._processor = WhisperProcessor.from_pretrained(base_path)

            forced_decoder_ids = None
            if self._processor is not None:
                forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                    language="arabic",
                    task="transcribe",
                )
            lang_to_id = getattr(self._model.generation_config, "lang_to_id", None)
            if isinstance(lang_to_id, dict) and lang_to_id:
                self._model.generation_config.language = "arabic"
                self._model.generation_config.task = "transcribe"
                self._model.generation_config.forced_decoder_ids = None
                self._forced_decoder_ids = None
            else:
                self._model.generation_config.language = None
                self._model.generation_config.task = None
                self._model.generation_config.forced_decoder_ids = forced_decoder_ids
                self._forced_decoder_ids = forced_decoder_ids
        elif self._model_type == "transformers_whisper":
            model_path = self._resolve_model_path(self._model_choice)
            print(f"[RealtimeStreamer] Loading HF model '{self._model_choice}' from {model_path}...")
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
            ).to(self.device)
            self._processor = WhisperProcessor.from_pretrained(model_path)
            
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language="arabic",
                task="transcribe",
            )
            lang_to_id = getattr(self._model.generation_config, "lang_to_id", None)
            if isinstance(lang_to_id, dict) and lang_to_id:
                self._model.generation_config.language = "arabic"
                self._model.generation_config.task = "transcribe"
                self._model.generation_config.forced_decoder_ids = None
                self._forced_decoder_ids = None
            else:
                self._model.generation_config.language = None
                self._model.generation_config.task = None
                self._model.generation_config.forced_decoder_ids = forced_decoder_ids
                self._forced_decoder_ids = forced_decoder_ids
        else:
            model_path = self._resolve_model_path(self._model_choice)
            print(f"[RealtimeStreamer] Loading model '{self._model_choice}' from {model_path}...")
            ct2_params = self._get_ct2_threading_params()
            self._model = WhisperModel(model_path, **ct2_params)
            self._processor = None
            self._forced_decoder_ids = None
            self._is_faster_whisper = True

        if hasattr(self._model, "eval"):
            self._model.eval()
        print(f"[RealtimeStreamer] Model ready ({model_path}) on {self.device}.")

        if os.path.isfile(self.ayah_json_path):
            self._ayah_map = load_all_ayat_json(self.ayah_json_path)
            print(f"[RealtimeStreamer] Loaded {len(self._ayah_map)} ayahs")

        # Build or load Quran prefix trie
        if self._ayah_map and self._is_faster_whisper:
            _tokenizer = getattr(self._model, 'hf_tokenizer', None)
            if _tokenizer is None:
                print("[RealtimeStreamer] hf_tokenizer not available — QuranTrie skipped")
            else:
                cached_trie = QuranTrie.load_cache()
                if cached_trie is not None:
                    self._quran_trie = cached_trie
                else:
                    try:
                        trie = QuranTrie()
                        trie.build(self._ayah_map, _tokenizer)
                        trie.save_cache()
                        self._quran_trie = trie
                    except Exception as e:
                        print(f"[RealtimeStreamer] Trie build failed: {e}")
                        self._quran_trie = None

        if self._surah_detector is None and os.path.isfile(self.ayah_json_path):
            try:
                self._surah_detector = SurahDetector(self.ayah_json_path)
            except Exception as e:
                print(f"[RealtimeStreamer] Surah detector not available: {e}")

        # Load viterbi pipeline
        # HybridViterbiPipeline construction moved to _ensure_viterbi_pipeline()
        # (2026-08-02, Claude/chat) — see comment in the groq branch above for
        # why this was unreachable dead weight at model-load time.

    def _decode_chunk(self, audio_np: np.ndarray) -> str:
        audio_np = audio_np.astype(np.float32)

        # ── Groq cloud decode path ────────────────────────────────────────
        if self._model_type == "groq":
            try:
                return self._model.transcribe_array(audio_np, sample_rate=SAMPLE_RATE)
            except Exception as exc:
                print(f"[RealtimeStreamer] Groq transcription error: {exc}")
                return ""

        # 200ms silence padding — Whisper pehla frame drop karta hai
        pad = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio_np = np.concatenate([pad, audio_np])
        
        if self._is_faster_whisper:
            # Build context-aware hotwords and initial_prompt
            ctx_surah = self._decode_context.get("surah")
            ctx_ayah = self._decode_context.get("ayah")

            hotwords = None
            initial_prompt = "بسم الله الرحمن الرحيم"

            if self._quran_trie is not None:
                hotwords = self._quran_trie.get_hotwords(ctx_surah, ctx_ayah) or None
                initial_prompt = self._quran_trie.get_initial_prompt(ctx_surah, ctx_ayah, self._ayah_map or {})

            transcribe_kwargs = dict(
                language="ar",
                beam_size=1,
                temperature=0.0,
                repetition_penalty=1.3,
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,
                initial_prompt=initial_prompt,
            )
            if hotwords:
                # faster-whisper expects hotwords as a comma-separated STRING, not a list
                transcribe_kwargs["hotwords"] = ", ".join(hotwords[:100])

            segments, _ = self._model.transcribe(audio_np, **transcribe_kwargs)
            return normalize_arabic(" ".join([s.text for s in segments]).strip())
        
        # Original whisper path (fallback)
        audio_np = audio_np[:_WHISPER_MAX_SAMPLES]
        if len(audio_np) < 400:
            return ""
        attention_mask = None
        input_features = self._processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features_tensor = input_features.input_features.to(self.device)
        if hasattr(input_features, "attention_mask"):
            attention_mask = input_features.attention_mask.to(self.device)
        generate_kwargs = {
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
        }
        if self._forced_decoder_ids is not None:
            generate_kwargs["forced_decoder_ids"] = self._forced_decoder_ids
        if self._beam_width > 1:
            generate_kwargs["num_beams"] = self._beam_width
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            generated_ids = self._model.generate(input_features_tensor, **generate_kwargs)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def _decode_raw(self, audio_np: np.ndarray) -> str:
        """Run Whisper decode on a 16kHz float32 audio array.

        Automatically splits audio longer than 30s into 25s chunks
        to stay within Whisper's hard mel-spectrogram limit.
        """
        audio_np = audio_np.astype(np.float32)

        if len(audio_np) <= _WHISPER_MAX_SAMPLES:
            return self._decode_chunk(audio_np)

        # Split into 25s chunks with 2s overlap and join
        chunk_size = 25 * SAMPLE_RATE
        overlap = 2 * SAMPLE_RATE
        parts = []
        start = 0
        while start < len(audio_np):
            end = min(start + chunk_size, len(audio_np))
            part_text = self._decode_chunk(audio_np[start:end])
            if part_text:
                parts.append(part_text)
            if end == len(audio_np):
                break
            start += chunk_size - overlap

        return " ".join(parts)

    def ensure_ayah_map_loaded(self):
        """Load just the ayah reference map, independent of any ASR model.

        Combined Mode doesn't use rt_streamer's own model at all (it uses
        Groq + the separate local turbo pipeline in
        hybrid_diacritic_pipeline.py), so nothing in that call path
        otherwise triggers _ensure_model_loaded(). guard_inference() still
        needs self._ayah_map for reference-text matching though. Standard
        mode gets this as a side effect of _ensure_model_loaded(); this
        covers Combined Mode being used before any Standard-mode call (or
        /session/start, which always calls _ensure_model_loaded() via
        create_session()) has ever warmed up this instance, which otherwise
        left _ayah_map as None and made every Combined-mode /transcribe
        call silently skip ayah matching (matched_ayah/confidence/verdict
        all stuck at their unmatched defaults).
        """
        if self._ayah_map is not None:
            return
        if os.path.isfile(self.ayah_json_path):
            self._ayah_map = load_all_ayat_json(self.ayah_json_path)
            print(f"[RealtimeStreamer] Loaded {len(self._ayah_map)} ayahs (ayah-map-only load)")

    def create_session(
        self,
        chunk_duration_s: float = 5.0,
        overlap_duration_s: float = 1.5,
        surah: Optional[int] = None,
        start_ayah: int = 1,
        use_vad: bool = True,
    ) -> RecitationSession:
        """Create a new recording session."""
        self._ensure_model_loaded()
        session = RecitationSession(
            session_dir=self.recordings_dir,
            chunk_duration_s=chunk_duration_s,
            overlap_duration_s=overlap_duration_s,
            surah=surah,
            start_ayah=start_ayah,
            use_vad=use_vad,
        )
        session.surah_lock_manager = SurahLockManager()
        session.surah_lock_state = {
            "lock_state": "INACTIVE",
            "locked_surah": None,
            "top_surah": None,
            "top_score": 0.0,
            "margin": 0.0,
            "candidates": [],
        }
        return session

    def process_chunk(
        self,
        session: RecitationSession,
        chunk_audio: np.ndarray,
        correction_mode: str = "balanced",
        chunk_start_sample: int = None,
        chunk_end_sample: int = None,
        qari_mode: bool = False,
    ) -> ChunkResult:
        """Decode and evaluate a single audio chunk. Returns the ChunkResult."""
        t0 = time.time()

        # --- Anti-hallucination gate at chunk level ---
        chunk_rms = float(np.sqrt(np.mean(chunk_audio.astype(np.float32) ** 2)))
        chunk_duration = len(chunk_audio) / SAMPLE_RATE
        if chunk_duration < 1.5 or chunk_rms < _MIN_SPEECH_RMS:
            print(f"[Chunk] Skipped — duration={chunk_duration:.2f}s, RMS={chunk_rms:.4f}")
            # Return a minimal empty result so the session doesn't stall
            return session.register_chunk_result(
                chunk_audio,
                {"corrected_text": "", "verdict": "skipped", "confidence": 0.0,
                 "raw_asr": "", "_skip_reason": f"rms={chunk_rms:.4f} dur={chunk_duration:.2f}s"},
                None, None,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
            )

        # 1. Raw ASR decode
        raw_text = self._decode_raw(chunk_audio)
        decode_time = time.time() - t0
        detection_text = _strip_leading_invocations(raw_text, strip_basmala=True)

        # 2. Accumulate text + detect surah on a rolling window
        # Initialize as a list of chunk strings rather than a flat token list
        if not hasattr(session, "_detection_buffer"):
            session._detection_buffer = []   # list of stripped chunk strings

        if detection_text and len(detection_text.split()) >= 2:
            session._detection_buffer.append(detection_text.strip())
            # Keep only the last 8 chunks (approx 40–60 tokens of context)
            session._detection_buffer = session._detection_buffer[-8:]

        # Join chunks with a double-space separator so n-grams don't bleed
        # across chunk boundaries during scoring
        accumulated_text = "  ".join(session._detection_buffer)

        lock_state = None
        lock_surah = None
        if (session.surah is None
                and self._surah_detector is not None
                and getattr(session, "surah_lock_manager", None)
                and len(session._detection_buffer) >= 2):    # was >= _MIN_DETECT_TOKENS (8 tokens) — now counting chunks
            candidates = self._surah_detector.detect(accumulated_text, top_k=5)
            lock_state = session.surah_lock_manager.update(candidates)
            session.surah_lock_state = lock_state
            if lock_state:
                lock_surah = lock_state.get("locked_surah")

        effective_surah = session.surah if session.surah is not None else lock_surah
        if session.surah is None and lock_surah is not None:
            session.surah = lock_surah
        # Update decode context so _decode_chunk can use surah/ayah for constrained decoding on next chunk
        self._decode_context = {"surah": effective_surah, "ayah": session.current_ayah}
        strip_basmala_for_analysis = not (
            effective_surah == 1 and (session.current_ayah is None or session.current_ayah <= 1)
        )
        analysis_text = _strip_leading_invocations(
            raw_text,
            strip_basmala=strip_basmala_for_analysis,
        )
        if raw_text and not analysis_text:
            return session.register_chunk_result(
                chunk_audio,
                {
                    "corrected_text": "",
                    "verdict": "skipped",
                    "confidence": 0.0,
                    "raw_asr": raw_text,
                    "_skip_reason": "leading_invocation_only",
                },
                None,
                lock_state,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
            )

        # 3. Guard inference (fast correction + ayah matching)
        effective_mode = "aggressive" if correction_mode == "balanced" else correction_mode
        correction_state = self.correction_engine.state if qari_mode else "LISTENING"
        strict_correction = qari_mode and correction_state in ("CORRECTING", "VERIFYING")
        lookahead = 0 if strict_correction else 10     # was 5 — wider forward search
        window_back = 0 if strict_correction else 5    # was 2 — wider backward search
        use_sequence = False if strict_correction else True
        sequence_max = 1 if strict_correction else 8

        guard_result = guard_inference(
            raw_text=analysis_text,
            ayah_map=self._ayah_map,
            surah=effective_surah,
            expected_ayah=session.current_ayah,
            lookahead=lookahead,
            window_back=window_back,
            correction_mode=effective_mode,
            allow_auto_correct=True,
            allow_reference_replacement=True,
            preserve_reciter=True,
            use_sequence_match=use_sequence,
            sequence_max_ayahs=sequence_max,
            lock_surah=lock_surah if session.surah is None else None,
        )

        if strict_correction and session.current_ayah is not None:
            if guard_result.get("matched_ayah") != session.current_ayah:
                fallback_text = None
                if session.surah is not None and self._ayah_map is not None:
                    fallback_text = self._ayah_map.get((session.surah, session.current_ayah))
                guard_result["matched_ayah"] = session.current_ayah
                if session.surah is not None:
                    guard_result["matched_key"] = (session.surah, session.current_ayah)
                if fallback_text:
                    guard_result["matched_ayah_text"] = fallback_text
                guard_result["verdict"] = "error"
        elif qari_mode:
            _force_expected_ayah_if_behind(guard_result, session, self._ayah_map)
            _flag_skipped_ayah_if_forward_jump(guard_result, session, self._ayah_map, self.correction_engine)

        if not getattr(session, "_guard_debug_printed", False):
            guard_debug = {
                "matched_key": guard_result.get("matched_key"),
                "matched_ayah": guard_result.get("matched_ayah"),
                "matched_start_ayah": guard_result.get("matched_start_ayah"),
                "matched_end_ayah": guard_result.get("matched_end_ayah"),
            }
            print(f"[DEBUG] guard keys: {list(guard_result.keys())}")
            print(f"[DEBUG] guard surah-related: {guard_debug}")
            session._guard_debug_printed = True

        # 4. Feed guard match back into lock manager (high-confidence vote)
        guard_conf = float(guard_result.get("confidence") or 0.0)
        guard_key = guard_result.get("matched_key")
        guard_surah = None
        guard_ayah = None
        if isinstance(guard_key, tuple) and len(guard_key) >= 2:
            guard_surah = guard_key[0]
            guard_ayah = guard_key[1]

        # Skip synthetic candidate injection when:
        #  - matched ayah is 1 (Basmala — shared by 113 surahs, inherently ambiguous)
        #  - confidence is below 0.70 (too noisy to trust)
        #  - detection buffer has < 8 tokens (not enough signal yet)
        _guard_has_enough_signal = (
            guard_ayah is not None
            and guard_ayah != 1
            and guard_conf >= 0.70
            and len(getattr(session, '_detection_buffer', [])) >= 2
        )
        if session.surah is None and guard_surah and _guard_has_enough_signal and getattr(session, "surah_lock_manager", None):
            boosted_score = min(0.95, guard_conf * 1.2)
            synthetic_candidates = [
                SurahCandidate(
                    surah=int(guard_surah),
                    score=boosted_score,
                    details={"guard_conf": guard_conf},
                )
            ]
            lock_state = session.surah_lock_manager.update(synthetic_candidates)
            session.surah_lock_state = lock_state
            if lock_state:
                lock_surah = lock_state.get("locked_surah")
                if lock_surah and session.surah is None:
                    session.surah = lock_surah
                    print(f"[Streamer] Guard locked surah -> {lock_surah} (conf={guard_conf:.2f})")

        effective_surah = session.surah if session.surah is not None else lock_surah

        if qari_mode and correction_state == "VERIFYING" and self.correction_engine.get_pending_corrections():
            # During verification we only care whether the specific mistake
            # that triggered the correction has now been said correctly —
            # not whether the whole ayah was re-recited from scratch. Long
            # ayahs get split across multiple VAD chunks, so requiring a
            # full-ayah match here would reject a perfectly correct retry
            # of just the flagged phrase.
            corrected_now = self.correction_engine.consume_pending_match(analysis_text)
            guard_result["corrected_text"] = analysis_text
            guard_result["display_text"] = analysis_text
            guard_result["verdict"] = "ok" if corrected_now else "error"
            guard_result["wrong_words"] = []
            guard_result["correction_spans"] = []
        elif qari_mode:
            ref_text = guard_result.get("matched_ayah_text") or ""
            wrong_words = []
            correction_spans = []
            if ref_text:
                hyp_text = correct_text_rules(analysis_text, mode="balanced")
                detection_text = _strip_trailing_partial_token(hyp_text, ref_text)
                annotations = get_word_error_annotations(detection_text, ref_text, confidence=None)
                annotations = _drop_trailing_future_missing(annotations)
                statuses = {a.get("status") for a in annotations}
                minor_fixed_text = _apply_minor_reference_fixes(annotations)
                if minor_fixed_text:
                    guard_result["corrected_text"] = minor_fixed_text
                    guard_result["display_text"] = minor_fixed_text
                elif hyp_text:
                    guard_result["corrected_text"] = hyp_text
                    guard_result["display_text"] = hyp_text

                first_teachable_idx = next(
                    (
                        i
                        for i, a in enumerate(annotations)
                        if a.get("status") in ("major", "missing")
                        and a.get("reference")
                    ),
                    None,
                )
                if first_teachable_idx is not None:
                    phrase_parts = []
                    phrase_start_idx = first_teachable_idx
                    for i, ann in enumerate(annotations[first_teachable_idx:], first_teachable_idx):
                        if i > first_teachable_idx and ann.get("status") == "correct":
                            break
                        status = ann.get("status")
                        ref_word = ann.get("reference")
                        if status in ("major", "missing") and ref_word:
                            phrase_parts.append(ref_word)
                            continue
                        if status not in ("extra", "uncertain") and phrase_parts:
                            wrong_words.append(" ".join(phrase_parts))
                            correction_spans.append(
                                _correction_span_from_annotations(annotations, phrase_start_idx, i - 1)
                            )
                            phrase_parts = []
                    if phrase_parts:
                        wrong_words.append(" ".join(phrase_parts))
                        correction_spans.append(
                            _correction_span_from_annotations(
                                annotations,
                                phrase_start_idx,
                                min(len(annotations) - 1, phrase_start_idx + len(phrase_parts) - 1),
                            )
                        )

                if statuses - {"correct", "minor", None}:
                    guard_result["verdict"] = "error"
                elif "minor" in statuses:
                    guard_result["verdict"] = "minor"
                elif statuses:
                    guard_result["verdict"] = "ok"

            guard_result["wrong_words"] = wrong_words
            guard_result["correction_spans"] = correction_spans

        if strict_correction and qari_mode:
            recited_clean = _strip_leading_invocations(
                guard_result.get("corrected_text") or analysis_text,
                strip_basmala=strip_basmala_for_analysis,
            )
            guard_result["corrected_text"] = recited_clean if recited_clean else ""
            guard_result["display_text"] = guard_result["corrected_text"]

        correction_verdict = guard_result.get("verdict", "unknown")

        # ── Turbo Quran LoRA: this model's whole purpose is preserving
        # harakat/tashkeel, so skip the normalize_arabic-based correction
        # pipeline for the DISPLAYED text only. Ayah matching/verdict above
        # already ran (on its own internally-normalized copies), so surah
        # progress, VAD, and everything else are unaffected — only what
        # lands in the Chunk Details table changes.
        if self._model_choice == "whisper-l-v3-turbo-quran-lora":
            guard_result["raw_asr"] = raw_text
            guard_result["corrected_text"] = raw_text
            guard_result["display_text"] = raw_text

        # Word-level error annotations, same computation as Combined Mode's
        # (see process_chunk_combined) and api/main.py's /transcribe endpoint
        # — kept here too so Standard Mode chunks get the same word_errors
        # field, not just Combined Mode ones.
        ref_text_for_words = guard_result.get("matched_ayah_text") or ""
        if ref_text_for_words:
            hyp_for_annotation = correct_text_rules(raw_text, mode="balanced")
            guard_result["word_errors"] = get_word_error_annotations(
                hyp_for_annotation,
                ref_text_for_words,
                confidence=guard_result.get("confidence"),
            )
        else:
            guard_result["word_errors"] = []

        guard_result["surah_lock_state"] = lock_state
        guard_result["_decode_time_s"] = round(decode_time, 3)
        guard_result["_chunk_duration_s"] = round(len(chunk_audio) / SAMPLE_RATE, 2)

        # 4. Register in session
        result = session.register_chunk_result(
            chunk_audio, guard_result, None, lock_state,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
        )

        print(
            f"[Chunk {result.chunk_index}] "
            f"decode={decode_time:.2f}s | "
            f"raw='{raw_text[:50]}' | "
            f"verdict={result.verdict} | "
            f"conf={result.confidence} | "
            f"surah={effective_surah} | "
            f"lock={(lock_state.get('lock_state') if lock_state else 'N/A')}"
        )
        
        if qari_mode:
            correction_result = self.correction_engine.process_verdict(
                verdict=correction_verdict,
                raw_asr=raw_text,
                correct_ayah_text=guard_result.get("matched_ayah_text", ""),
                ayah_num=guard_result.get("matched_ayah"),
                surah_num=effective_surah,
                wrong_words=guard_result.get("wrong_words", []),
                correction_spans=guard_result.get("correction_spans", []),
                confidence=float(guard_result.get("confidence") or 1.0),
            )
            if correction_result["action"] == "pause":
                self._vad_paused = True
            elif correction_result["action"] in ("continue", "skip"):
                self._vad_paused = False
        else:
            self._vad_paused = False
            
        return result

    def process_chunk_combined(
        self,
        session: RecitationSession,
        chunk_audio: np.ndarray,
        correction_mode: str = "balanced",
        chunk_start_sample: int = None,
        chunk_end_sample: int = None,
        qari_mode: bool = False,
    ) -> ChunkResult:
        """Combined Model mode sibling of process_chunk() — masterplan.md §4.2 (Phase 3).

        Decodes via hybrid_diacritic_pipeline (Groq consonant backbone + local
        turbo diacritics) instead of the single offline model, runs the same
        guard_inference ayah-matching/correction pipeline process_chunk() uses,
        and additionally runs harakaat_error_detector against the matched
        ayah's diacritized reference text. process_chunk() itself is untouched;
        this is a new, separate function.

        qari_mode (masterplan.md changelog, 2026-08-02): when True, also runs
        Qari Mode word-level scoring (_apply_qari_word_scoring) and hands the
        verdict + harakaat_errors to self.correction_engine.process_verdict(),
        the same way process_chunk()'s qari_mode branch already does for
        Standard Mode. Without this, Combined Mode's harakaat_errors were
        computed but never reached the correction engine, and
        CorrectionEngine._handle_harakaat_hint() was unreachable.
        """
        # Lazy imports — Combined Mode's local turbo model and Groq client are
        # only touched when this function actually runs, per masterplan §3.1.
        from hybrid_diacritic_pipeline import run_combined_transcription, run_hybrid_combination_logic
        from harakaat_error_detector import detect_harakaat_errors

        t0 = time.time()

        # --- Anti-hallucination gate (same thresholds as process_chunk) ---
        chunk_rms = float(np.sqrt(np.mean(chunk_audio.astype(np.float32) ** 2)))
        chunk_duration = len(chunk_audio) / SAMPLE_RATE
        if chunk_duration < 1.5 or chunk_rms < _MIN_SPEECH_RMS:
            print(f"[Chunk-Combined] Skipped — duration={chunk_duration:.2f}s, RMS={chunk_rms:.4f}")
            return session.register_chunk_result(
                chunk_audio,
                {"corrected_text": "", "verdict": "skipped", "confidence": 0.0,
                 "raw_asr": "", "_skip_reason": f"rms={chunk_rms:.4f} dur={chunk_duration:.2f}s"},
                None, None,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
            )

        # 1. Combined decode — first pass, no ayah context yet (masterplan
        #    §4.2 step 2). Groq/local only ever decode once per chunk;
        #    everything below reuses decode_result['groq_text']/['local_text'].
        decode_result = run_combined_transcription(chunk_audio.astype(np.float32), SAMPLE_RATE)
        raw_text = decode_result["combined_text"]
        decode_time = time.time() - t0
        detection_text = _strip_leading_invocations(raw_text, strip_basmala=True)

        # Per-engine terminal output — confirms both models are actually
        # running and lets you compare their raw outputs chunk by chunk.
        print(f"[Chunk-Combined] Groq:     {decode_result['groq_text']}")
        print(f"[Chunk-Combined] Local:    {decode_result['local_text']}")
        print(f"[Chunk-Combined] Combined: {decode_result['combined_text']}")

        # 2. Rolling-window surah detection — same mechanism as process_chunk
        if not hasattr(session, "_detection_buffer"):
            session._detection_buffer = []
        if detection_text and len(detection_text.split()) >= 2:
            session._detection_buffer.append(detection_text.strip())
            session._detection_buffer = session._detection_buffer[-8:]
        accumulated_text = "  ".join(session._detection_buffer)

        lock_state = None
        lock_surah = None
        if (session.surah is None
                and self._surah_detector is not None
                and getattr(session, "surah_lock_manager", None)
                and len(session._detection_buffer) >= 2):
            candidates = self._surah_detector.detect(accumulated_text, top_k=5)
            lock_state = session.surah_lock_manager.update(candidates)
            session.surah_lock_state = lock_state
            if lock_state:
                lock_surah = lock_state.get("locked_surah")

        effective_surah = session.surah if session.surah is not None else lock_surah

        strip_basmala_for_analysis = not (
            effective_surah == 1 and (session.current_ayah is None or session.current_ayah <= 1)
        )
        analysis_text = _strip_leading_invocations(raw_text, strip_basmala=strip_basmala_for_analysis)
        if raw_text and not analysis_text:
            return session.register_chunk_result(
                chunk_audio,
                {"corrected_text": "", "verdict": "skipped", "confidence": 0.0,
                 "raw_asr": raw_text, "_skip_reason": "leading_invocation_only"},
                None, lock_state,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
            )

        # 3. Existing guard pipeline for ayah matching/correction — unchanged
        #    (masterplan §4.2 step 3; same call shape as process_chunk's non-strict path)
        guard_result = guard_inference(
            raw_text=analysis_text,
            ayah_map=self._ayah_map,
            surah=effective_surah,
            expected_ayah=session.current_ayah,
            lookahead=10,
            window_back=5,
            correction_mode="aggressive" if correction_mode == "balanced" else correction_mode,
            allow_auto_correct=True,
            allow_reference_replacement=True,
            preserve_reciter=True,
            use_sequence_match=True,
            sequence_max_ayahs=8,
            lock_surah=lock_surah if session.surah is None else None,
        )

        # 3a. Reference-verified second pass (masterplan §4.2 step 3, per
        #     Hamza — "best of both" merge using the now-known matched ayah).
        #     Re-runs only the merge step (run_hybrid_combination_logic), not
        #     the ASR decode, against the SAME groq_text/local_text from step
        #     1. For each word the first-pass merge still disagrees with the
        #     reference on, adopts whichever engine actually said the
        #     reference word at that slot; if neither did, that word is left
        #     exactly as the first pass decided — a genuine mistake is never
        #     replaced by the reference text. See run_hybrid_combination_logic's
        #     stage 2b docstring for the full rule.
        ref_text_for_merge = guard_result.get("matched_ayah_text") or ""
        if ref_text_for_merge:
            verified_stages = run_hybrid_combination_logic(
                decode_result["groq_text"], decode_result["local_text"], ref_text=ref_text_for_merge,
            )
            verified_text = verified_stages["combined"]
            if verified_text != raw_text:
                print(f"[Chunk-Combined] Verified: {verified_text}")
                raw_text = verified_text
                decode_result["combined_text"] = verified_text
                analysis_text = _strip_leading_invocations(raw_text, strip_basmala=strip_basmala_for_analysis)
                guard_result["raw_asr"] = raw_text

        # 3b. Qari Mode word-level scoring (masterplan.md changelog, 2026-08-02)
        #     Mirrors process_chunk()'s qari_mode branch via _apply_qari_word_scoring
        #     so wrong_words/correction_spans/verdict feed the correction engine
        #     the same way they do in Standard Mode.
        correction_state = self.correction_engine.state if qari_mode else "LISTENING"
        if qari_mode and correction_state == "VERIFYING" and self.correction_engine.get_pending_corrections():
            corrected_now = self.correction_engine.consume_pending_match(analysis_text)
            guard_result["corrected_text"] = analysis_text
            guard_result["display_text"] = analysis_text
            guard_result["verdict"] = "ok" if corrected_now else "error"
            guard_result["wrong_words"] = []
            guard_result["correction_spans"] = []
        elif qari_mode:
            _apply_qari_word_scoring(guard_result, analysis_text)

        # 4. Harakaat error detection against the matched ayah's diacritized
        #    reference — new in Combined Mode (masterplan §4.2 step 4 / §3.2)
        matched_key = guard_result.get("matched_key")
        if isinstance(matched_key, (tuple, list)) and len(matched_key) >= 2:
            harakaat_result = detect_harakaat_errors(raw_text, int(matched_key[0]), int(matched_key[1]))
            guard_result["harakaat_errors"] = [
                {"index": w.index, "ref_index": w.ref_index, "predicted": w.predicted_word,
                 "reference": w.reference_word, "status": w.status}
                for w in harakaat_result.words if w.status == "harakaat_error"
            ]
            guard_result["harakaat_error_count"] = harakaat_result.harakaat_error_count

        # 5. Word-level error annotations (correct/minor/major/missing/extra)
        # against the matched ayah. Best-of-both-engines (2026-08-04,
        # Claude/chat, per Hamza): the transcript backbone (raw_text /
        # combined_text) always keeps Groq's word choice per
        # hybrid_diacritic_pipeline.py's documented design — it never
        # consults the reference, so an honest recitation mistake always
        # stays visible in the transcript. But that means a case where Groq
        # simply misheard a word into something completely different while
        # Local heard it correctly (e.g. "وإذ" vs حقة "غير") used to be
        # scored purely on Groq's wrong word, with no way for Local's better
        # answer to ever count. get_word_error_annotations_best_of() aligns
        # BOTH engines' raw output against the reference independently and
        # takes whichever one actually matched at each word position — this
        # only changes SCORING (word_errors), never the displayed transcript.
        ref_text_for_words = guard_result.get("matched_ayah_text") or ""
        if ref_text_for_words:
            groq_hyp = correct_text_rules(decode_result.get("groq_text") or "", mode="balanced")
            local_hyp = correct_text_rules(decode_result.get("local_text") or "", mode="balanced")
            if groq_hyp and local_hyp:
                guard_result["word_errors"] = get_word_error_annotations_best_of(
                    groq_hyp,
                    local_hyp,
                    ref_text_for_words,
                    primary_confidence=guard_result.get("confidence"),
                    alt_confidence=guard_result.get("confidence"),
                )
            else:
                # One engine returned nothing this chunk (failed/empty) —
                # fall back to scoring whichever single hypothesis exists,
                # same as before this change.
                hyp_for_annotation = correct_text_rules(raw_text, mode="balanced")
                guard_result["word_errors"] = get_word_error_annotations(
                    hyp_for_annotation,
                    ref_text_for_words,
                    confidence=guard_result.get("confidence"),
                )
        else:
            guard_result["word_errors"] = []

        guard_result["surah_lock_state"] = lock_state
        guard_result["_decode_time_s"] = round(decode_time, 3)
        guard_result["_chunk_duration_s"] = round(chunk_duration, 2)

        result = session.register_chunk_result(
            chunk_audio, guard_result, None, lock_state,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
        )

        print(
            f"[Chunk-Combined {result.chunk_index}] "
            f"decode={decode_time:.2f}s | "
            f"raw='{raw_text[:50]}' | "
            f"verdict={result.verdict} | "
            f"harakaat_errors={guard_result.get('harakaat_error_count', 0)} | "
            f"surah={effective_surah}"
        )

        # Qari Mode dispatch (masterplan.md changelog, 2026-08-02) — same
        # call shape as process_chunk()'s qari_mode branch, now including
        # harakaat_errors so CorrectionEngine._handle_harakaat_hint() can
        # actually fire for Combined Mode.
        if qari_mode:
            correction_result = self.correction_engine.process_verdict(
                verdict=guard_result.get("verdict", "unknown"),
                raw_asr=raw_text,
                correct_ayah_text=guard_result.get("matched_ayah_text", ""),
                ayah_num=guard_result.get("matched_ayah"),
                surah_num=effective_surah,
                wrong_words=guard_result.get("wrong_words", []),
                correction_spans=guard_result.get("correction_spans", []),
                confidence=float(guard_result.get("confidence") or 1.0),
                harakaat_errors=guard_result.get("harakaat_errors"),
            )
            self._last_qari_action = correction_result
            if correction_result["action"] == "pause":
                self._vad_paused = True
            elif correction_result["action"] in ("continue", "skip"):
                self._vad_paused = False
        else:
            self._vad_paused = False

        return result

    def process_chunk_api(
        self,
        session: RecitationSession,
        chunk_audio: np.ndarray,
        correction_mode: str = "balanced",
        chunk_start_sample: int = None,
        chunk_end_sample: int = None,
        qari_mode: bool = False,
        api_base_url: str = "http://127.0.0.1:8000",
        model_choice: str = "groq-whisper-large-v3",
    ) -> ChunkResult:
        """Combined Mode via a remote api/main.py server's POST /transcribe,
        instead of running Groq + the local turbo model in this process
        (masterplan.md §9 — "Use Remote API for Combined Mode").

        Only ever used for Combined Mode, opt-in via app.py's toggle;
        Standard Mode (process_chunk(), the local tarteel-ai-based model) is
        completely unaffected regardless of this toggle.

        api/main.py's /transcribe already runs the FULL pipeline server-side
        (decode + guard_inference + harakaat detection + word-level
        annotation) and returns it as one JSON blob, so on success this just
        maps that JSON onto the exact same guard_result shape
        process_chunk_combined() builds, then reuses
        session.register_chunk_result() and
        self.correction_engine.process_verdict() unchanged — this is why
        Qari Mode, TTS, and the live display's harakaat highlight all just
        work here with zero changes: they only ever read that shared shape,
        never care which pipeline produced it.

        On failure (server unreachable/timeout), degrades to a harmless
        low-confidence "error"-verdict chunk — same tolerance the rest of
        the pipeline already has for a garbled/low-confidence chunk (see
        CorrectionEngine._min_trigger_confidence) — rather than crashing the
        session or silently falling back to a local Groq/turbo-model path
        this machine may not even have installed (the whole point of using
        the remote API is often that it ISN'T installed here).

        Known limitation: unlike process_chunk_combined(), this path does
        not run local rolling-window surah detection (session._detection_buffer
        / surah_lock_manager) — the stateless /transcribe endpoint has no
        equivalent, so each ChunkResult.surah_lock_state here simply carries
        forward whatever the session's last known lock state was. Auto-surah-
        detection therefore only works well with this toggle on once a surah
        is already selected/locked; pure "Auto (Detect)" sessions are better
        served by local Combined Mode until this gets its own remote
        surah-detection call.
        """
        t0 = time.time()

        chunk_rms = float(np.sqrt(np.mean(chunk_audio.astype(np.float32) ** 2)))
        chunk_duration = len(chunk_audio) / SAMPLE_RATE
        if chunk_duration < 1.5 or chunk_rms < _MIN_SPEECH_RMS:
            print(f"[Chunk-API] Skipped — duration={chunk_duration:.2f}s, RMS={chunk_rms:.4f}")
            return session.register_chunk_result(
                chunk_audio,
                {"corrected_text": "", "verdict": "skipped", "confidence": 0.0,
                 "raw_asr": "", "_skip_reason": f"rms={chunk_rms:.4f} dur={chunk_duration:.2f}s"},
                None, None,
                chunk_start_sample=chunk_start_sample,
                chunk_end_sample=chunk_end_sample,
            )

        effective_surah = session.surah
        expected_ayah = session.current_ayah

        api_result = transcribe_via_api(
            chunk_audio.astype(np.float32),
            SAMPLE_RATE,
            surah=effective_surah,
            expected_ayah=expected_ayah,
            correction_mode=correction_mode,
            asr_engine="combined",
            model_choice=model_choice,
            api_base_url=api_base_url,
        )
        decode_time = time.time() - t0

        if api_result is None:
            guard_result = {
                "raw_asr": "", "corrected_text": "", "display_text": "",
                "matched_ayah": None, "matched_key": None, "matched_ayah_text": None,
                "cer": None, "wer": None, "coverage": None,
                "confidence": 0.0, "confidence_level": "low", "verdict": "error",
                "harakaat_errors": [], "harakaat_error_count": 0, "word_errors": [],
                "wrong_words": [], "correction_spans": [],
            }
        else:
            matched_ayah = api_result.get("matched_ayah")
            matched_key = (effective_surah, matched_ayah) if (effective_surah and matched_ayah) else None
            word_errors = api_result.get("word_errors") or []
            wrong_words, correction_spans = _wrong_words_and_spans_from_word_errors(word_errors)
            guard_result = {
                "raw_asr": api_result.get("raw_asr", ""),
                "corrected_text": api_result.get("corrected_text", ""),
                "display_text": api_result.get("corrected_text", ""),
                "matched_ayah": matched_ayah,
                "matched_key": matched_key,
                "matched_ayah_text": api_result.get("matched_ayah_text"),
                "cer": api_result.get("cer"),
                "wer": api_result.get("wer"),
                "coverage": None,
                "confidence": api_result.get("confidence"),
                "confidence_level": api_result.get("confidence_level"),
                "verdict": api_result.get("verdict", "unknown"),
                "harakaat_errors": api_result.get("harakaat_errors") or [],
                "harakaat_error_count": api_result.get("harakaat_error_count", 0),
                "word_errors": word_errors,
                "wrong_words": wrong_words,
                "correction_spans": correction_spans,
            }

        guard_result["surah_lock_state"] = getattr(session, "surah_lock_state", None)
        guard_result["_decode_time_s"] = round(decode_time, 3)
        guard_result["_chunk_duration_s"] = round(chunk_duration, 2)

        lock_state = getattr(session, "surah_lock_state", None)
        result = session.register_chunk_result(
            chunk_audio, guard_result, None, lock_state,
            chunk_start_sample=chunk_start_sample,
            chunk_end_sample=chunk_end_sample,
        )

        print(
            f"[Chunk-API {result.chunk_index}] "
            f"decode={decode_time:.2f}s | "
            f"raw='{guard_result.get('raw_asr','')[:50]}' | "
            f"verdict={result.verdict} | "
            f"harakaat_errors={guard_result.get('harakaat_error_count', 0)} | "
            f"surah={effective_surah} | api_ok={api_result is not None}"
        )

        if qari_mode:
            correction_result = self.correction_engine.process_verdict(
                verdict=guard_result.get("verdict", "unknown"),
                raw_asr=guard_result.get("raw_asr", ""),
                correct_ayah_text=guard_result.get("matched_ayah_text", ""),
                ayah_num=guard_result.get("matched_ayah"),
                surah_num=effective_surah,
                wrong_words=guard_result.get("wrong_words", []),
                correction_spans=guard_result.get("correction_spans", []),
                confidence=float(guard_result.get("confidence") or 1.0),
                harakaat_errors=guard_result.get("harakaat_errors"),
            )
            self._last_qari_action = correction_result
            if correction_result["action"] == "pause":
                self._vad_paused = True
            elif correction_result["action"] in ("continue", "skip"):
                self._vad_paused = False
        else:
            self._vad_paused = False

        return result

    def _ensure_viterbi_pipeline(self) -> None:
        """Lazily build self._viterbi_pipeline on first actual use.

        Only decode_full_session() calls this. Moved out of
        _ensure_model_loaded() (2026-08-02, Claude/chat, per Hamza) because
        it was building a HybridViterbiPipeline — n-gram index over 6235
        ayahs plus a KenLM LM load — unconditionally at model-load time,
        even though decode_full_session() is currently unreachable from both
        the desktop live flow (app.py's stop_live_session() deliberately
        removed the full-session re-decode step) and the API (api/main.py
        never calls it either). Kept, not deleted, in case a
        full-session-compare feature gets rewired to call it later.
        """
        if self._viterbi_pipeline is not None:
            return
        try:
            from hybrid_pipeline import HybridViterbiPipeline
            lm_path = os.path.join(_BASE_DIR, "quran_5gram.arpa")
            if os.path.isfile(lm_path):
                self._viterbi_pipeline = HybridViterbiPipeline(self.ayah_json_path, lm_path)
        except Exception as e:
            print(f"[RealtimeStreamer] Viterbi pipeline not available: {e}")

    def decode_full_session(
        self,
        session: RecitationSession,
        correction_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Re-decode the full session audio as one piece for comparison."""
        self._ensure_viterbi_pipeline()
        full_audio = session.get_full_session_audio()
        if len(full_audio) == 0:
            return {"raw_asr": "", "guard_result": {}, "viterbi_result": {}, "eval_result": {}}

        lock_surah = None
        if session.surah is None and hasattr(session, "get_final_locked_surah"):
            lock_surah = session.get_final_locked_surah()
        effective_surah = session.surah if session.surah is not None else lock_surah

        raw_text = self._decode_raw(full_audio)

        guard_result = guard_inference(
            raw_text=raw_text,
            ayah_map=self._ayah_map,
            surah=effective_surah,
            expected_ayah=session.start_ayah,
            lookahead=10,
            window_back=2,
            correction_mode="aggressive",
            allow_auto_correct=True,
            allow_reference_replacement=True,
            preserve_reciter=True,
            use_sequence_match=True,
            sequence_max_ayahs=30,
            lock_surah=lock_surah if session.surah is None else None,
        )

        viterbi_result = {"aligned_ayahs": []}
        if self._viterbi_pipeline:
            try:
                viterbi_result = self._viterbi_pipeline.pipeline_from_text(
                    raw_text,
                    start_surah=effective_surah,
                    lock_surah=lock_surah if session.surah is None else None,
                )
            except Exception as e:
                viterbi_result = {"error": str(e), "aligned_ayahs": []}

        return {
            "raw_asr": raw_text,
            "guard_result": guard_result,
            "viterbi_result": viterbi_result,
            "eval_result": viterbi_result,
        }


def resample_to_16k(audio_np: np.ndarray, source_sr: int) -> np.ndarray:
    """Resample audio from any sample rate to 16kHz using torchaudio."""
    if source_sr == SAMPLE_RATE:
        return audio_np.astype(np.float32)

    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.ndim == 2 and audio_tensor.shape[1] <= 2:
        audio_tensor = audio_tensor.T

    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

    resampled = torchaudio.functional.resample(audio_tensor, source_sr, SAMPLE_RATE)
    return resampled.squeeze(0).numpy().astype(np.float32)
