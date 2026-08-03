import sys
import os
import queue
import threading
from typing import Optional

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
try:
    from dotenv import load_dotenv
    if os.path.isfile(_env_path):
        load_dotenv(_env_path, override=True)
    else:
        load_dotenv()
except ImportError:
    if os.path.isfile(_env_path):
        try:
            with open(_env_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        _k, _v = _k.strip(), _v.strip().strip("'\"")
                        if _k and _k not in os.environ:
                            os.environ[_k] = _v
        except Exception:
            pass

# Ensure we can import from fyp_model
sys.path.append(os.path.join(os.path.dirname(__file__), "fyp_model"))

import gradio as gr
import torch
import numpy as np
import torchaudio
import time
import base64
from urllib.parse import quote
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from faster_whisper import WhisperModel as FasterWhisperModel
from hybrid_pipeline import HybridViterbiPipeline
from realtime_streamer import RealtimeStreamer, resample_to_16k
from session_manager import RecitationSession
from fyp_model.quran_guard import guard_inference, load_all_ayat_json, normalize_arabic
from live_display_formatter import LiveDisplayFormatter
from surah_detector import SurahDetector

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

import librosa
from math import gcd
from scipy.signal import butter, filtfilt, resample_poly

def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32)


def _clean_transcript_text(text: str) -> str:
    return normalize_arabic(text)


def _decode_raw_text(model, processor, audio_np: np.ndarray, device: str, model_type: str = "whisper") -> str:
    """Greedy decode for upload tab."""
    if model_type == "groq":
        # model is a GroqTranscriber instance
        try:
            return model.transcribe_array(audio_np, sample_rate=16000)
        except Exception as exc:
            print(f"[Hafizify] Groq transcription error: {exc}")
            return ""

    if model_type == "wav2vec2":
        inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return _clean_transcript_text(processor.batch_decode(pred_ids)[0])

    if model_type == "faster_whisper":
        # faster-whisper (CTranslate2) path
        audio_np = audio_np.astype(np.float32)
        pad = np.zeros(int(0.2 * 16000), dtype=np.float32)
        audio_np = np.concatenate([pad, audio_np])
        segments, _ = model.transcribe(
            audio_np,
            language="ar",
            beam_size=5,
            temperature=0.0,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            vad_filter=False,
        )
        return _clean_transcript_text(" ".join([s.text for s in segments]).strip())

    # Whisper: handle 30s hard limit
    max_samples = 30 * 16000
    audio_np = audio_np[:max_samples].astype(np.float32)
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = model.generation_config.forced_decoder_ids
    if forced_decoder_ids is None:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="arabic",
            task="transcribe",
        )
    generate_kwargs = {
        "forced_decoder_ids": forced_decoder_ids,
    }
    attn = inputs.get("attention_mask")
    if attn is not None:
        generate_kwargs["attention_mask"] = attn.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features, **generate_kwargs)
    return _clean_transcript_text(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model selection — supported models
MODEL_REGISTRY = {
    "whisper-base-quran-lora": {
        "type": "whisper",
        "local": "whisper-base-quran-lora",
        "hf": "KheemP/whisper-base-quran-lora",
    },
    "faster-whisper-base-ar-quran": {
        "type": "faster_whisper",
        "local": "faster-whisper-base-ar-quran",
        "hf": "OdyAsh/faster-whisper-base-ar-quran",
    },
    "whisper-l-v3-turbo-quran-lora": {
        "type": "whisper",
        "local": "whisper-l-v3-turbo-quran-lora-dataset-mix",
        "hf": "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix",
    },
    # ── Groq cloud ASR ────────────────────────────────────────────────────
    "groq-whisper-large-v3": {
        "type": "groq",
        # No local weights — always calls the Groq REST API
    },
}

MODEL_CHOICES = [
    "whisper-base-quran-lora",
    "whisper-l-v3-turbo-quran-lora",
    "faster-whisper-base-ar-quran",
    "groq-whisper-large-v3",
]
_current_model_choice = None
_model_type = "whisper"  # always whisper now
_forced_decoder_ids = None

processor = None
model = None
evaluator_obj = None
viterbi_pipeline = None
beam_decoder_obj = None
beam_decoder_available = False
ayah_map = None
SURAH_DETECTOR = None

# Real-time streaming globals
rt_streamer = RealtimeStreamer()
_active_session: RecitationSession = None
_chunk_queue = queue.Queue()
_worker_thread = None
_stop_event = threading.Event()
_display_formatter: LiveDisplayFormatter = None
_session_lock = threading.Lock()

def _process_queue_worker():
    while not _stop_event.is_set():
        try:
            task = _chunk_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Task tuple has grown over time (masterplan.md §9 added the
            # remote-API fields) — unpack by length, oldest shape first, so
            # nothing already queued in an older shape ever breaks.
            use_api_combined = False
            api_base_url = "http://127.0.0.1:8000"
            if len(task) == 7:
                session, chunk_tuple, correction_mode, qari_mode, asr_engine, use_api_combined, api_base_url = task
            elif len(task) == 5:
                session, chunk_tuple, correction_mode, qari_mode, asr_engine = task
            elif len(task) == 4:
                session, chunk_tuple, correction_mode, qari_mode = task
                asr_engine = "standard"
            else:
                session, chunk_tuple, correction_mode = task
                qari_mode = False
                asr_engine = "standard"

            chunk_audio, chunk_start, chunk_end = chunk_tuple

            # Only audio captured WHILE the correction TTS is playing
            # (state == CORRECTING) is stale — that's the reciter's audio
            # from before they heard the correction. Once state reaches
            # VERIFYING, the mic has reopened specifically to listen for the
            # reciter's correction attempt, so that chunk must be processed
            # or the Qari state machine can never advance past VERIFYING.
            if qari_mode and rt_streamer.correction_engine.state == "CORRECTING":
                session.discard_pending_chunk(chunk_start, chunk_end)
                print(f"[Worker] Dropped stale chunk (Qari state={rt_streamer.correction_engine.state})")
                continue

            if asr_engine == "combined" and use_api_combined:
                rt_streamer.process_chunk_api(
                    session, chunk_audio, correction_mode=correction_mode,
                    qari_mode=qari_mode,
                    chunk_start_sample=chunk_start, chunk_end_sample=chunk_end,
                    api_base_url=api_base_url or "http://127.0.0.1:8000",
                )
            elif asr_engine == "combined":
                rt_streamer.process_chunk_combined(
                    session, chunk_audio, correction_mode=correction_mode,
                    qari_mode=qari_mode,
                    chunk_start_sample=chunk_start, chunk_end_sample=chunk_end,
                )
            else:
                rt_streamer.process_chunk(
                    session, chunk_audio, correction_mode=correction_mode,
                    qari_mode=qari_mode,
                    chunk_start_sample=chunk_start, chunk_end_sample=chunk_end,
                )
        except Exception as e:
            print(f"[Worker] Error processing chunk: {e}")
        finally:
            _chunk_queue.task_done()


def load_models_once(model_choice="whisper-base-quran-lora"):
    global processor, model, evaluator_obj, viterbi_pipeline, _current_model_choice, _model_type, _forced_decoder_ids, ayah_map, SURAH_DETECTOR

    if model_choice not in MODEL_REGISTRY:
        print(f"[Hafizify] Unknown model '{model_choice}', defaulting to whisper-base-quran-lora")
        model_choice = "whisper-base-quran-lora"

    # If same model is already loaded, skip
    if model is not None and _current_model_choice == model_choice:
        return

    # Reset state if switching models
    model = None
    processor = None
    _forced_decoder_ids = None

    info = MODEL_REGISTRY[model_choice]
    _model_type = info["type"]

    # ── Groq cloud model — no local weights to resolve ──────────────────
    if _model_type == "groq":
        from groq_transcriber import get_groq_transcriber
        model = get_groq_transcriber()
        processor = None
        _forced_decoder_ids = None
        _current_model_choice = model_choice
        print(f"[Hafizify] Groq whisper-large-v3 ready (cloud API).")
        # Still load ayah map + surah detector (HybridViterbiPipeline/kenlm
        # moved to _ensure_viterbi_pipeline(), built lazily on first actual
        # transcribe() call instead — see that function's note)
        ayah_path = os.path.join(BASE_DIR, "fyp_model", "all_ayat.json")
        if ayah_map is None and os.path.isfile(ayah_path):
            ayah_map = load_all_ayat_json(ayah_path)
        if SURAH_DETECTOR is None and os.path.isfile(ayah_path):
            SURAH_DETECTOR = SurahDetector(ayah_path)
        return

    local_key = info.get("local")
    if local_key:
        local_path = os.path.join(BASE_DIR, *local_key.replace("\\", "/").split("/"))
        model_path = local_path if os.path.isdir(local_path) else info.get("hf")
    else:
        model_path = info.get("hf")

    if not model_path:
        raise FileNotFoundError(f"Model '{model_choice}' not found locally and no HF fallback available.")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[Hafizify] Loading model '{model_choice}' from {model_path}...")
    if _model_type == "faster_whisper":
        model = FasterWhisperModel(
            model_path,
            device="cpu",
            compute_type="int8",
        )
        processor = None
        _forced_decoder_ids = None
    elif _model_type == "groq":
        # No local weights — load the Groq API client as the "model"
        from groq_transcriber import get_groq_transcriber
        model = get_groq_transcriber()
        processor = None
        _forced_decoder_ids = None
        print(f"[Hafizify] Groq whisper-large-v3-turbo ready (cloud API).")
    else:  # whisper (HuggingFace)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)
        processor = WhisperProcessor.from_pretrained(model_path)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="arabic",
            task="transcribe",
        )
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.language = None
        model.generation_config.task = None
        _forced_decoder_ids = forced_decoder_ids

    if hasattr(model, "eval"):
        model.eval()
    _current_model_choice = model_choice
    print(f"[Hafizify] Model ready: {model_path} on {device}")

    ayah_path = os.path.join(BASE_DIR, "fyp_model", "all_ayat.json")
    # HybridViterbiPipeline/kenlm moved to _ensure_viterbi_pipeline(), built
    # lazily on first actual transcribe() call instead — see that function's
    # note (2026-08-02, Claude/chat, per Hamza's "we don't even use kenlm"
    # catch: this was loading a kenlm LM + 97k-bigram/trigram n-gram index
    # at every model load even though nothing in the live-session flow ever
    # calls it, and the file-upload tab that does isn't wired to the UI).

    if ayah_map is None and os.path.isfile(ayah_path):
        ayah_map = load_all_ayat_json(ayah_path)

    if SURAH_DETECTOR is None and os.path.isfile(ayah_path):
        SURAH_DETECTOR = SurahDetector(ayah_path)


def load_beam_decoder_once():
    """Lazily initialise the beam search decoder (only when first needed)."""
    global beam_decoder_obj, beam_decoder_available
    if beam_decoder_obj is not None:
        return

    try:
        from beam_decoder import load_beam_decoder, QURAN_HOTWORDS

        # Try to find the KenLM model
        kenlm_candidates = [
            os.path.join(BASE_DIR, "quran_5gram.bin"),
            os.path.join(BASE_DIR, "quran_5gram.arpa"),
        ]
        kenlm_path = next((p for p in kenlm_candidates if os.path.isfile(p)), None)

        beam_decoder_obj = load_beam_decoder(
            processor,
            kenlm_model_path=kenlm_path,
            hotwords=QURAN_HOTWORDS,
        )
        beam_decoder_available = True
        lm_status = kenlm_path or "no LM (beam search only)"
        print(f"Beam decoder loaded OK — LM: {lm_status}")
    except Exception as e:
        print(f"Beam decoder not available: {e}")
        beam_decoder_available = False


# ---------------------------------------------------------------------------
# Surah name list for the dropdown
# ---------------------------------------------------------------------------

SURAH_NAMES = [
    "Auto (Detect)",
    "1 - Al-Fatiha", "2 - Al-Baqarah", "3 - Aal-Imran", "4 - An-Nisa",
    "5 - Al-Ma'idah", "6 - Al-An'am", "7 - Al-A'raf", "8 - Al-Anfal",
    "9 - At-Tawbah", "10 - Yunus", "11 - Hud", "12 - Yusuf",
    "13 - Ar-Ra'd", "14 - Ibrahim", "15 - Al-Hijr", "16 - An-Nahl",
    "17 - Al-Isra", "18 - Al-Kahf", "19 - Maryam", "20 - Taha",
    "21 - Al-Anbiya", "22 - Al-Hajj", "23 - Al-Mu'minun", "24 - An-Nur",
    "25 - Al-Furqan", "26 - Ash-Shu'ara", "27 - An-Naml", "28 - Al-Qasas",
    "29 - Al-Ankabut", "30 - Ar-Rum", "31 - Luqman", "32 - As-Sajdah",
    "33 - Al-Ahzab", "34 - Saba", "35 - Fatir", "36 - Ya-Sin",
    "37 - As-Saffat", "38 - Sad", "39 - Az-Zumar", "40 - Ghafir",
    "41 - Fussilat", "42 - Ash-Shura", "43 - Az-Zukhruf", "44 - Ad-Dukhan",
    "45 - Al-Jathiyah", "46 - Al-Ahqaf", "47 - Muhammad", "48 - Al-Fath",
    "49 - Al-Hujurat", "50 - Qaf", "51 - Adh-Dhariyat", "52 - At-Tur",
    "53 - An-Najm", "54 - Al-Qamar", "55 - Ar-Rahman", "56 - Al-Waqi'ah",
    "57 - Al-Hadid", "58 - Al-Mujadilah", "59 - Al-Hashr", "60 - Al-Mumtahanah",
    "61 - As-Saff", "62 - Al-Jumu'ah", "63 - Al-Munafiqun", "64 - At-Taghabun",
    "65 - At-Talaq", "66 - At-Tahrim", "67 - Al-Mulk", "68 - Al-Qalam",
    "69 - Al-Haqqah", "70 - Al-Ma'arij", "71 - Nuh", "72 - Al-Jinn",
    "73 - Al-Muzzammil", "74 - Al-Muddaththir", "75 - Al-Qiyamah", "76 - Al-Insan",
    "77 - Al-Mursalat", "78 - An-Naba", "79 - An-Nazi'at", "80 - Abasa",
    "81 - At-Takwir", "82 - Al-Infitar", "83 - Al-Mutaffifin", "84 - Al-Inshiqaq",
    "85 - Al-Buruj", "86 - At-Tariq", "87 - Al-A'la", "88 - Al-Ghashiyah",
    "89 - Al-Fajr", "90 - Al-Balad", "91 - Ash-Shams", "92 - Al-Layl",
    "93 - Ad-Duha", "94 - Ash-Sharh", "95 - At-Tin", "96 - Al-Alaq",
    "97 - Al-Qadr", "98 - Al-Bayyinah", "99 - Az-Zalzalah", "100 - Al-Adiyat",
    "101 - Al-Qari'ah", "102 - At-Takathur", "103 - Al-Asr", "104 - Al-Humazah",
    "105 - Al-Fil", "106 - Quraysh", "107 - Al-Ma'un", "108 - Al-Kawthar",
    "109 - Al-Kafirun", "110 - An-Nasr", "111 - Al-Masad", "112 - Al-Ikhlas",
    "113 - Al-Falaq", "114 - An-Nas",
]


def _parse_surah_number(surah_str: str) -> Optional[int]:
    """Extract surah number from dropdown like '1 - Al-Fatiha'."""
    if not surah_str or surah_str.startswith("Auto"):
        return None
    try:
        return int(surah_str.split(" - ")[0].strip())
    except (ValueError, IndexError):
        return None


def _parse_asr_engine(engine_str: str) -> str:
    """Map the 'ASR Engine' radio label to 'combined' or 'standard'.
    Default is 'standard' for any unrecognized/empty value, so an existing
    user who never touches this control gets byte-identical behavior
    (masterplan.md §4.3)."""
    if engine_str and engine_str.startswith("Combined"):
        return "combined"
    return "standard"


# ---------------------------------------------------------------------------
# File Upload Transcription (original functionality)
# ---------------------------------------------------------------------------

def _ensure_viterbi_pipeline():
    """Lazily build the module-level viterbi_pipeline on first actual use.

    Only transcribe() (the file-upload tab) calls this. Moved out of
    load_models_once() (2026-08-02, Claude/chat, per Hamza) because it was
    building a HybridViterbiPipeline — n-gram index over 6235 ayahs plus a
    KenLM LM load — unconditionally at every model load, even though the
    file-upload tab that calls it isn't currently wired to any UI button
    (only the Live Recitation tab is defined in this file's gr.Blocks()),
    and the live-session flow itself never touches viterbi_pipeline at all.
    Kept, not deleted, in case that tab gets rewired to the UI later.
    """
    global viterbi_pipeline
    if viterbi_pipeline is not None:
        return
    ayah_path = os.path.join(BASE_DIR, "fyp_model", "all_ayat.json")
    lm_path = os.path.join(BASE_DIR, "quran_5gram.arpa")
    viterbi_pipeline = HybridViterbiPipeline(ayah_path, lm_path)


def transcribe(
    audio_path,
    model_choice,
    correction_mode,
    sequence_guard,
    allow_reference_replacement,
    use_beam_search,
    beam_width,
):
    if not audio_path:
        return "No audio provided.", "", "", "", "", {}

    try:
        load_models_once(model_choice)
    except Exception as e:
        return f"Failed to load model: {e}", "", "", "", "", {}

    if not os.path.isfile(audio_path):
        return f"Invalid audio file: {audio_path}", "", "", "", "", {}

    try:
        audio_np = load_audio_16k(audio_path)
    except Exception as e:
        return f"Failed to load audio: {e}", "", "", "", "", {}

    try:
        decode_method = "greedy"

        if use_beam_search and _model_type == "whisper":
            # Whisper beam search via generate() with forced_decoder_ids
            max_samples = 30 * 16000
            audio_clamped = audio_np[:max_samples].astype(np.float32)
            inputs = processor(audio_clamped, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            forced_decoder_ids = _forced_decoder_ids or model.generation_config.forced_decoder_ids
            if forced_decoder_ids is None:
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="arabic",
                    task="transcribe",
                )
            gen_kwargs = {
                "num_beams": max(1, int(beam_width)),
                "forced_decoder_ids": forced_decoder_ids,
            }
            attn = inputs.get("attention_mask")
            if attn is not None:
                gen_kwargs["attention_mask"] = attn.to(device)
            with torch.no_grad():
                generated_ids = model.generate(inputs.input_features.to(device), **gen_kwargs)
            raw_text = _clean_transcript_text(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
            decode_method = f"whisper_beam_search (num_beams={gen_kwargs['num_beams']})"
        elif use_beam_search and _model_type == "wav2vec2":
            load_beam_decoder_once()
            if beam_decoder_available:
                from beam_decoder import decode_beam

                raw_text = _clean_transcript_text(
                    decode_beam(
                        model,
                        processor,
                        beam_decoder_obj,
                        audio_np.astype(np.float32),
                        device=device,
                        beam_width=max(1, int(beam_width)),
                    )
                )
                decode_method = f"wav2vec2_beam_search (num_beams={int(beam_width)})"
            else:
                raw_text = _decode_raw_text(model, processor, audio_np, device, model_type=_model_type)
                decode_method = "wav2vec2_greedy (beam unavailable)"
        elif use_beam_search:
            raw_text = _decode_raw_text(model, processor, audio_np, device, model_type=_model_type)
            decode_method = "greedy (beam search unavailable for this model)"
        else:
            raw_text = _decode_raw_text(model, processor, audio_np, device, model_type=_model_type)

        guard_result = None
        if ayah_map is not None:
            guard_result = guard_inference(
                raw_text=raw_text,
                ayah_map=ayah_map,
                surah=None,
                expected_ayah=None,
                lookahead=5,
                window_back=2,
                correction_mode=correction_mode,
                allow_auto_correct=True,
                allow_reference_replacement=allow_reference_replacement,
                preserve_reciter=True,
                use_sequence_match=sequence_guard,
                sequence_max_ayahs=10,
            )

        guard_corrected = (
            guard_result.get("corrected_text") if guard_result else raw_text
        )

        # Run sequence alignment pipeline (no global matching)
        _ensure_viterbi_pipeline()
        viterbi_result = viterbi_pipeline.pipeline_from_text(guard_corrected)
        
        # Format the continuous reconstructed sequence
        corrected_text = guard_corrected
        
        # Display guard verdict
        verdict = guard_result.get("verdict", "") if guard_result else ""

        evaluation_report = {
            "guard_result": guard_result,
            "viterbi_result": viterbi_result,
        }

        # Surah detection for upload tab
        detected_surah_str = "N/A"
        if SURAH_DETECTOR and raw_text:
            surah_candidates = SURAH_DETECTOR.detect(raw_text, top_k=3)
            if surah_candidates:
                top = surah_candidates[0]
                surah_names_map = {i: n for i, n in enumerate(SURAH_NAMES) if i > 0}
                top_name = surah_names_map.get(top.surah, f"Surah {top.surah}")
                detected_surah_str = f"{top_name} ({top.score:.2f})"

        return (
            raw_text,
            corrected_text,
            str(guard_result.get("confidence", "N/A") if guard_result else "N/A"),
            verdict,
            decode_method,
            detected_surah_str,
            evaluation_report,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"Transcription error: {e}", "", "N/A", "error", "N/A", "N/A", {})


# ---------------------------------------------------------------------------
# Live Recitation — Streaming Handlers
# ---------------------------------------------------------------------------
def start_live_session(
    surah_choice,
    start_ayah,
    chunk_duration,
    overlap_duration,
    model_choice,
    use_vad,
    auto_surah_detect,
    qari_mode=False,
    asr_engine="Standard (offline, fast)",
    use_api_combined=False,
):
    """Initialize a new recording session when user clicks Start."""
    global _active_session, _worker_thread, _display_formatter

    # Reset Correction Engine state for new session
    if qari_mode:
        rt_streamer.correction_engine.reset()
        rt_streamer._vad_paused = False

    # Set the model choice on the streamer (triggers reload if different)
    rt_streamer.set_model_choice(model_choice)

    # Combined Mode: force the local turbo model to load NOW, synchronously,
    # instead of on the first streamed audio chunk. Without this the first
    # chunk after Start eats the multi-second HF model load on top of
    # inference, which reads as a large delay right as the reciter starts
    # speaking. Groq needs no local load (cloud API), so only the local
    # turbo pipeline is preloaded here.
    if _parse_asr_engine(asr_engine) == "combined" and not use_api_combined:
        from hybrid_diacritic_pipeline import preload_local_pipeline
        try:
            preload_local_pipeline()
        except Exception as e:
            print(f"[Hafizify] Combined Mode local model preload failed, will lazy-load on first chunk: {e}")

    if _worker_thread is None or not _worker_thread.is_alive():
        _stop_event.clear()
        _worker_thread = threading.Thread(target=_process_queue_worker, daemon=True)
        _worker_thread.start()

    surah_num = _parse_surah_number(surah_choice)
    start_ayah_val = max(1, int(start_ayah or 1))

    new_session = rt_streamer.create_session(
        chunk_duration_s=float(chunk_duration),
        overlap_duration_s=float(overlap_duration),
        surah=surah_num,
        start_ayah=start_ayah_val,
        use_vad=bool(use_vad),
    )
    new_session.surah_detection_enabled = bool(auto_surah_detect)
    if not new_session.surah_detection_enabled:
        new_session.surah_lock_manager = None

    new_formatter = LiveDisplayFormatter(os.path.join(BASE_DIR, "fyp_model", "all_ayat.json"))
    with _session_lock:
        _active_session = new_session
        _display_formatter = new_formatter

    vad_status = "VAD" if _active_session.use_vad else "Fixed-time"
    surah_label = _surah_name(surah_num) if surah_num is not None else "Auto (Detect)"
    status = (
        f"🟢 Session started — {surah_label}, Ayah {start_ayah_val}\n"
        f"Chunking: {vad_status} | Chunk: {chunk_duration}s | Overlap: {overlap_duration}s\n"
        f"Session ID: {_active_session.session_id}"
    )
    placeholder_html = LiveDisplayFormatter._placeholder_html("Start reciting...")
    empty_panel = LiveDisplayFormatter._error_panel_html(0, 0, 0, 0, 0, 0.0)
    table_header = "<div class='chunk-table'>No chunks yet.</div>"
    badge_html = _format_surah_badge(None)
    progress_html = LiveDisplayFormatter._placeholder_html("Surah progress will appear here.")
    if new_formatter and surah_num is not None:
        progress_html = new_formatter.format_surah_progress_html(
            surah_num,
            start_ayah_val,
            start_ayah_val,
            "",
        )
    
    correction_html = _format_correction_status("LISTENING") if qari_mode else ""
    return status, placeholder_html, progress_html, empty_panel, "", "", table_header, "", badge_html, correction_html


def _surah_name(surah_num):
    """Return the human-readable surah name for a given surah number."""
    if surah_num is None:
        return "Unknown"
    # SURAH_NAMES[0] = "Auto (Detect)", so surah 1 is at index 1
    idx = surah_num
    if 0 < idx < len(SURAH_NAMES):
        return SURAH_NAMES[idx]
    return f"Surah {surah_num}"


def _format_surah_guess(lock_state: Optional[dict], fallback_surah: Optional[int]) -> str:
    if lock_state:
        if lock_state.get("lock_state") == "ACTIVE" and lock_state.get("locked_surah"):
            return f"Locked: {_surah_name(lock_state['locked_surah'])}"
        if lock_state.get("top_surah"):
            score = lock_state.get("top_score", 0.0)
            return f"Detecting: {_surah_name(lock_state['top_surah'])} ({score:.2f})"
    if fallback_surah:
        return _surah_name(fallback_surah)
    return "Listening..."


def _format_surah_badge(lock_state: Optional[dict]) -> str:
    label = "Detecting..."
    color = "#64748b"

    if lock_state:
        if lock_state.get("lock_state") == "ACTIVE" and lock_state.get("locked_surah"):
            label = f"Locked: {_surah_name(lock_state['locked_surah'])}"
            color = "#10b981"
        elif lock_state.get("top_surah"):
            label = f"Detecting: {_surah_name(lock_state['top_surah'])}"
            color = "#f59e0b"

    return (
        f"<div style='display:inline-block; padding:6px 12px; border-radius:999px;"
        f" background:{color}; color:#0f172a; font-weight:700; font-size:13px;'>"
        f"{label}</div>"
    )


def _format_surah_candidates(candidates) -> str:
    if not candidates:
        return "Not enough signal"
    parts = []
    for cand in candidates[:3]:
        parts.append(f"{_surah_name(cand.surah)} ({cand.score:.2f})")
    return "; ".join(parts)


def _format_correction_status(state: str) -> str:
    if state == "LISTENING":
        color = "#10b981"  # green
    elif state == "CORRECTING":
        color = "#ef4444"  # red
    elif state == "VERIFYING":
        color = "#f59e0b"  # yellow
    else:
        color = "#64748b"  # gray

    return (
        f"<div style='padding:8px 16px; border-radius:8px; text-align:center; font-weight:bold; "
        f"background-color:rgba({LiveDisplayFormatter._hex_to_rgb(color)},0.2); color:{color}; "
        f"border: 1px solid {color}; margin-top:10px;'>"
        f"QARI MODE: {state}</div>"
    )


def _build_expected_ayah_html(session, formatter, qari_mode: bool) -> str:
    if formatter is None or session is None:
        return gr.update()

    recited_text = ""
    harakaat_errors = None
    if session.chunk_results:
        last = session.chunk_results[-1]
        if last.matched_ayah == session.current_ayah:
            recited_text = last.corrected_text or ""
            harakaat_errors = last.harakaat_errors

    return formatter.format_expected_ayah_html(
        session.surah,
        session.current_ayah,
        recited_text,
        harakaat_errors=harakaat_errors,
    )


def _stable_html_update(session, key: str, value: object) -> object:
    if session is None or not isinstance(value, str):
        return value

    last = getattr(session, key, None)
    if last == value:
        return gr.update()

    setattr(session, key, value)
    return value


def _build_surah_progress_html(session, formatter, qari_mode: bool) -> str:
    if formatter is None or session is None:
        return gr.update()

    recited_text = ""
    harakaat_errors = None
    if session.chunk_results:
        last = session.chunk_results[-1]
        if last.matched_ayah == session.current_ayah:
            recited_text = last.corrected_text or ""
            harakaat_errors = last.harakaat_errors

    return formatter.format_surah_progress_html(
        session.surah,
        session.start_ayah,
        session.current_ayah,
        recited_text,
        harakaat_errors=harakaat_errors,
    )


def process_streaming_audio(audio_data, correction_mode, qari_mode, asr_engine, use_api_combined=False, api_base_url="http://127.0.0.1:8000"):
    """Called by Gradio's streaming mic with each audio fragment.

    audio_data is a tuple (sample_rate, numpy_array) from Gradio.
    Returns updated live display, error panel, raw/corrected text, and chunk table.
    """
    global _active_session, _display_formatter

    no_update = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    with _session_lock:
        session_ref = _active_session
        formatter_ref = _display_formatter
    if session_ref is None or not session_ref.is_active:
        return no_update
    if audio_data is None:
        return no_update
        
    if qari_mode and getattr(rt_streamer, "_vad_paused", False):
        # Update _last_fed_samples so we don't process audio accumulated during TTS
        sr_temp, audio_temp = audio_data
        audio_temp = audio_temp.astype(np.float32)
        if audio_temp.max() > 1.0:
            audio_temp = audio_temp / 32768.0
        # Update cumulative offset so resume doesn't include TTS-era audio
        if sr_temp != 16000:
            g = gcd(16000, sr_temp)
            estimated_16k_len = int(len(audio_temp) * 16000 / sr_temp)
        else:
            estimated_16k_len = len(audio_temp)
        session_ref._cumulative_offset = estimated_16k_len
        session_ref._last_fed_samples = estimated_16k_len
        session_ref.skip_paused_audio()
        correction_state = rt_streamer.correction_engine.state
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), _format_correction_status(correction_state)
        )

    sr, audio_np = audio_data

    # ── Step 1: Ensure float32 in [-1, 1] — handle ALL Gradio formats ────
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    elif audio_np.dtype == np.uint8:
        audio_np = (audio_np.astype(np.float32) - 128.0) / 128.0
    else:
        audio_np = audio_np.astype(np.float32)
        # Normalize if values exceed [-1, 1]
        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            audio_np = audio_np / max_val

    # ── Step 2: Mono conversion ───────────────────────────────────────────
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    elif audio_np.ndim > 2:
        audio_np = audio_np[:, 0]

    # ── Step 3: Resample to 16kHz using scipy (more accurate for streaming)
    if sr != 16000:
        g = gcd(16000, sr)
        audio_np = resample_poly(audio_np, 16000 // g, sr // g).astype(np.float32)

    # ── Step 4: Skip pure silence ──────────────────────────────────────────
    # NOTE: Amplitude normalization and high-pass filtering are intentionally
    # NOT done here.  They are applied per-chunk in _emit_segment() after VAD
    # assembles a full speech segment.  Applying them per-fragment causes gain
    # pumping and filtfilt edge transients → choppy audio.
    if np.max(np.abs(audio_np)) < 0.001:
        return no_update  # pure silence, skip

    audio_16k = audio_np

    # Gradio streaming mode differs by version:
    #   - Gradio 3.x: cumulative (each call receives the full buffer from start)
    #   - Gradio 4.x+/6.x: incremental (each call receives only the new fragment)
    # Detect once on the 2nd callback: if the buffer grew significantly (>1.5x
    # the first fragment), it's cumulative; otherwise it's incremental.
    if not hasattr(session_ref, '_audio_call_count'):
        session_ref._audio_call_count = 0
        session_ref._first_fragment_len = 0
        session_ref._is_cumulative = False
        session_ref._cumulative_offset = 0

    session_ref._audio_call_count += 1
    n = len(audio_16k)

    if session_ref._audio_call_count == 1:
        # First callback — always feed the whole fragment
        session_ref._first_fragment_len = n
        session_ref._cumulative_offset = n
        audio_delta = audio_16k
    elif session_ref._audio_call_count == 2:
        # Second callback — detect mode
        if n > session_ref._first_fragment_len * 1.5:
            session_ref._is_cumulative = True
            audio_delta = audio_16k[session_ref._cumulative_offset:]
            session_ref._cumulative_offset = n
            print(f"[Audio] Detected CUMULATIVE streaming (Gradio 3.x style)")
        else:
            session_ref._is_cumulative = False
            audio_delta = audio_16k
            print(f"[Audio] Detected INCREMENTAL streaming (Gradio 4.x+/6.x)")
    else:
        # Subsequent callbacks — use detected mode
        if session_ref._is_cumulative:
            if n > session_ref._cumulative_offset:
                audio_delta = audio_16k[session_ref._cumulative_offset:]
                session_ref._cumulative_offset = n
            else:
                audio_delta = np.array([], dtype=np.float32)
        else:
            audio_delta = audio_16k

    if len(audio_delta) == 0:
        return no_update

    # Feed into session buffer and extract any ready chunks
    ready_chunks = session_ref.feed_audio(audio_delta)

    # Queue each ready chunk for background processing
    for chunk_tuple in ready_chunks:
        chunk_audio, chunk_start, chunk_end = chunk_tuple
        session_ref.register_pending_chunk(chunk_start, chunk_end)
        _chunk_queue.put((
            session_ref, chunk_tuple, correction_mode, qari_mode,
            _parse_asr_engine(asr_engine), bool(use_api_combined), api_base_url,
        ))

    # Build color-coded merged HTML display
    chunk_dicts = session_ref.get_chunk_results_as_dicts()
    if formatter_ref and chunk_dicts:
        expected_html = _build_expected_ayah_html(session_ref, formatter_ref, qari_mode)
        progress_html = _build_surah_progress_html(session_ref, formatter_ref, qari_mode)
        error_html = formatter_ref.generate_error_panel(chunk_dicts)
    else:
        expected_html = _build_expected_ayah_html(session_ref, formatter_ref, qari_mode)
        progress_html = _build_surah_progress_html(session_ref, formatter_ref, qari_mode)
        error_html = gr.update()

    expected_html = _stable_html_update(session_ref, "_last_expected_html", expected_html)
    progress_html = _stable_html_update(session_ref, "_last_progress_html", progress_html)
    error_html = _stable_html_update(session_ref, "_last_error_html", error_html)

    # Raw vs corrected text
    all_raw = " ".join(r.raw_asr for r in session_ref.chunk_results if r.raw_asr)
    all_corrected = session_ref.get_merged_transcript() or ""

    chunks_table = _build_chunks_table(session_ref)

    # Detect guessed surah from matched chunk results
    detected_surah = session_ref.get_detected_surah()
    lock_state = getattr(session_ref, "surah_lock_state", None)
    if getattr(session_ref, "surah_detection_enabled", True) is False:
        guessed_surah_text = "Auto detection off"
        badge_html = _format_surah_badge(None)
    else:
        guessed_surah_text = _format_surah_guess(lock_state, detected_surah)
        badge_html = _format_surah_badge(lock_state)

    correction_state = rt_streamer.correction_engine.state if qari_mode else "INACTIVE"
    correction_html = _format_correction_status(correction_state) if qari_mode else ""

    return expected_html, progress_html, error_html, all_raw, all_corrected, chunks_table, guessed_surah_text, badge_html, correction_html


def _build_chunks_table(session):
    """Build the HTML table of chunk results for the given session."""
    def _audio_src(path: str) -> str:
        if not path or not os.path.isfile(path):
            return ""

        cache = getattr(session, "_chunk_audio_cache", None)
        if cache is None:
            cache = {}
            session._chunk_audio_cache = cache

        cached = cache.get(path)
        if cached:
            return cached

        try:
            with open(path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            src = f"data:audio/wav;base64,{b64}"
        except Exception:
            normalized = path.replace("\\", "/")
            src = f"/file={quote(normalized)}"

        cache[path] = src
        return src

    rows = [
        "<table class='chunk-table'>",
        "<thead><tr><th>Audio</th><th>#</th><th>Time Range</th><th>Raw ASR</th>"
        "<th>Corrected</th><th>Confidence</th><th>Verdict</th></tr></thead>",
        "<tbody>",
    ]
    for r in session.chunk_results:
        raw_short = (r.raw_asr[:40] + "...") if len(r.raw_asr) > 40 else r.raw_asr
        corr_short = (r.corrected_text[:40] + "...") if len(r.corrected_text) > 40 else r.corrected_text
        conf_str = f"{r.confidence:.2f}" if r.confidence is not None else "-"
        time_range = f"{r.start_time_s:.1f}-{r.end_time_s:.1f}s"
        src = _audio_src(r.chunk_wav_path)
        if src:
            audio_html = f"<audio controls preload='metadata' src='{src}'></audio>"
        else:
            audio_html = "-"
        rows.append(
            "<tr>"
            f"<td>{audio_html}</td>"
            f"<td>{r.chunk_index}</td>"
            f"<td>{time_range}</td>"
            f"<td>{raw_short}</td>"
            f"<td>{corr_short}</td>"
            f"<td>{conf_str}</td>"
            f"<td>{r.verdict}</td>"
            "</tr>"
        )
    for p in getattr(session, "pending_chunks", []):
        time_range = f"{p['start_time_s']:.1f}-{p['end_time_s']:.1f}s"
        rows.append(
            "<tr>"
            "<td>-</td>"
            f"<td>{p['chunk_index']}</td>"
            f"<td>{time_range}</td>"
            "<td>Processing...</td>"
            "<td>Processing...</td>"
            "<td>-</td>"
            "<td>pending</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _build_live_ui_snapshot(session):
    """Build a full snapshot of all live-updating UI components for the session."""
    chunk_dicts = session.get_chunk_results_as_dicts()
    if _display_formatter and chunk_dicts:
        expected_html = _build_expected_ayah_html(session, _display_formatter, qari_mode=False)
        progress_html = _build_surah_progress_html(session, _display_formatter, qari_mode=False)
        error_html = _display_formatter.generate_error_panel(chunk_dicts)
    else:
        expected_html = _build_expected_ayah_html(session, _display_formatter, qari_mode=False)
        progress_html = _build_surah_progress_html(session, _display_formatter, qari_mode=False)
        error_html = LiveDisplayFormatter._error_panel_html(0, 0, 0, 0, 0, 0.0)

    all_raw = " ".join(r.raw_asr for r in session.chunk_results if r.raw_asr)
    all_corrected = session.get_merged_transcript() or ""
    chunks_table = _build_chunks_table(session)
    detected = session.get_detected_surah()
    lock_state = getattr(session, "surah_lock_state", None)
    if getattr(session, "surah_detection_enabled", True) is False:
        guessed = "Auto detection off"
        badge_html = _format_surah_badge(None)
    else:
        guessed = _format_surah_guess(lock_state, detected)
        badge_html = _format_surah_badge(lock_state)

    return expected_html, progress_html, error_html, all_raw, all_corrected, chunks_table, guessed, badge_html


def stop_live_session(correction_mode, qari_mode, asr_engine, use_api_combined=False, api_base_url="http://127.0.0.1:8000"):
    """Drain the queue (yielding updates), then return all 13 UI outputs with full results."""
    global _active_session

    # 13 outputs: live_status, live_merged_display, surah_progress_display, error_panel, raw_asr_box,
    #             corrected_box, chunks_table_md, comparison_md,
    #             session_eval_json, guessed_surah_box, surah_badge_html, session_wav, correction_status
    if _active_session is None:
        yield ("No active session.", "", "", "", "", "", "", "", {}, "", "", None, "")
        return

    with _session_lock:
        session = _active_session
        _active_session = None  # prevent streaming callback from feeding more audio

    try:
        # ---- 1. Flush remaining audio into the queue ----
        print("[Session] Flushing remaining audio...")
        remaining = session.flush_remaining_audio()
        for chunk_tuple in remaining:
            chunk_audio, chunk_start, chunk_end = chunk_tuple
            session.register_pending_chunk(chunk_start, chunk_end)
            _chunk_queue.put((
                session, chunk_tuple, correction_mode, qari_mode,
                _parse_asr_engine(asr_engine), bool(use_api_combined), api_base_url,
            ))
        print(f"[Session] {len(remaining)} final chunk(s) queued; waiting for ALL chunks…")

        # ---- 2. Yield updates while worker processes remaining chunks ----
        while _chunk_queue.unfinished_tasks > 0:
            merged_html, progress_html, error_html, all_raw, all_corrected, chunks_table, guessed, badge_html = \
                _build_live_ui_snapshot(session)
            yield (
                f"🔴 Stopping... finishing {int(_chunk_queue.unfinished_tasks)} remaining chunk(s)...",
                merged_html, progress_html, error_html, all_raw, all_corrected,
                chunks_table, "", {}, guessed, badge_html, None, ""
            )
            time.sleep(1.0)

        # ---- 3. Build the COMPLETE chunk table & display (every chunk is now done) ----
        merged_html, progress_html, error_html, all_raw, all_corrected, chunks_table, guessed, badge_html = \
            _build_live_ui_snapshot(session)
        yield (
            "🔴 Finalizing session...",
            merged_html, progress_html, error_html, all_raw, all_corrected,
            chunks_table, "", {}, guessed, badge_html, None, ""
        )

        # ---- 4. Finalize (saves WAVs + JSON) ----
        results_path = session.finalize()

        # ---- 5. Build session summary from chunk-by-chunk results only ----
        # (Removed the full-session re-decode/"compare" step — re-decoding the
        # entire session's audio from scratch on CPU as one blob was taking
        # ~1 minute and crashing after Stop was pressed. The chunk-by-chunk
        # merged transcript already covers everything needed; this just
        # drops the redundant second decode pass.)
        chunk_merged = session.get_merged_transcript()

        detected_surah = session.get_detected_surah()
        lock_state = getattr(session, "surah_lock_state", None)
        if getattr(session, "surah_detection_enabled", True) is False:
            guessed_surah_name = "Auto detection off"
            badge_html = _format_surah_badge(None)
        else:
            guessed_surah_name = _format_surah_guess(lock_state, detected_surah)
            badge_html = _format_surah_badge(lock_state)

        comparison = f"""## 📊 Session Summary

### Chunk-by-Chunk Merged Transcript
> {chunk_merged or '(no chunks processed)'}

---

### Session Stats
- **Total Duration:** {session.total_duration_s:.1f}s
- **Chunks Processed:** {len(session.chunk_results)}
- **Guessed Surah:** {guessed_surah_name}
- **Session ID:** {session.session_id}
"""

        eval_report = {}

        status = (
            f"🔴 Session ended — {len(session.chunk_results)} chunks, "
            f"{session.total_duration_s:.1f}s total\n"
            f"Guessed Surah: {guessed_surah_name}\n"
            f"Files saved to: {session.session_dir}"
        )
        
        if qari_mode:
            summary = rt_streamer.correction_engine.get_session_summary()
            print(f"Session accuracy: {summary['accuracy']}%")
            print(f"Errors: {summary['total_errors']}, Skipped: {summary['skipped_ayahs']}")

        session_wav = session.session_wav_path if os.path.isfile(session.session_wav_path) else None
        yield (
            status, merged_html, progress_html, error_html, all_raw, all_corrected,
            chunks_table, comparison, eval_report, guessed_surah_name, badge_html, session_wav, ""
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Even if something crashes, still return the chunk table we have
        merged_html, progress_html, error_html, all_raw, all_corrected, chunks_table, guessed, badge_html = \
            _build_live_ui_snapshot(session)
        yield (
            f"🔴 Session ended with error: {e}",
            merged_html, progress_html, error_html, all_raw, all_corrected,
            chunks_table, "", {}, guessed, badge_html, None, "",
        )


def _preload_live_model():
    """Warm up the Live Recitation model (rt_streamer) at app startup.

    Without this, the model load happens lazily inside start_live_session(),
    which now fires directly off mic_input.start_recording() since the
    separate Start Session button was removed. That made model loading a
    multi-second BLOCKING call that runs at the exact moment the browser
    starts uploading its first streaming audio chunk — on Windows/Chrome
    this races and can leave a truncated temp .wav on disk, which ffmpeg
    then fails to decode (CouldntDecodeError / 'Invalid data found when
    processing input'). Preloading here means create_session()'s internal
    _ensure_model_loaded() call is a no-op by the time Record is tapped.
    """
    try:
        rt_streamer.set_model_choice("whisper-base-quran-lora")
        rt_streamer._ensure_model_loaded()
        print("[Hafizify] Live Recitation model preloaded at startup.")
    except Exception as e:
        print(f"[Hafizify] Live model preload failed, will lazy-load on first Record tap: {e}")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_APP_CSS = """
/* Hide Tab Navigation Bar */
.tabs > .tab-nav, .tab-nav, div[role="tablist"], button[role="tab"], .tabs-nav, .tab-buttons {
    display: none !important;
}

/* ===================== Dark navy / teal reskin =====================
   These override Gradio's own CSS custom properties, so they apply
   regardless of exact Gradio version without touching any Python
   theme API surface. Component wiring below this block is untouched. */
:root, .dark {
    --body-background-fill: #0a0f17 !important;
    --background-fill-primary: #131b28 !important;
    --background-fill-secondary: #0f1622 !important;
    --block-background-fill: #131b28 !important;
    --panel-background-fill: #131b28 !important;
    --border-color-primary: rgba(255,255,255,0.08) !important;
    --block-border-color: rgba(255,255,255,0.08) !important;
    --block-label-background-fill: transparent !important;
    --block-label-text-color: #7f93aa !important;
    --block-title-text-color: #e7edf5 !important;
    --body-text-color: #e7edf5 !important;
    --body-text-color-subdued: #8fa3b8 !important;
    --input-background-fill: #0f1622 !important;
    --input-border-color: rgba(79,179,196,0.28) !important;
    --input-border-color-focus: #4fb3c4 !important;
    --button-primary-background-fill: linear-gradient(135deg, #5ec4d6 0%, #3f97a8 100%) !important;
    --button-primary-background-fill-hover: linear-gradient(135deg, #6ed0e1 0%, #4aa6b8 100%) !important;
    --button-primary-text-color: #06141c !important;
    --button-primary-border-color: transparent !important;
    --button-secondary-background-fill: #16202f !important;
    --button-secondary-text-color: #cfe0ee !important;
    --button-secondary-border-color: rgba(255,255,255,0.08) !important;
    --button-cancel-background-fill: linear-gradient(135deg, #e07a86 0%, #b8505f 100%) !important;
    --button-cancel-text-color: #06141c !important;
    --button-cancel-border-color: transparent !important;
    --slider-color: #4fb3c4 !important;
}

body, .gradio-container {
    background:
        radial-gradient(circle at 20% 0%, rgba(79, 179, 196, 0.10), transparent 40%),
        linear-gradient(180deg, #0c121c 0%, #0a0f17 100%) !important;
}

.gradio-container .block, .gradio-container .form {
    border-radius: 16px !important;
}

/* Sidebar / main-panel columns on the Live Recitation tab */
.sidebar-col > .form, .sidebar-col {
    background: transparent !important;
}
.sidebar-col .block {
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28) !important;
}
.main-col .block {
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22) !important;
}

/* Buttons */
button.primary, .btn-start button {
    box-shadow: 0 12px 26px rgba(79, 179, 196, 0.22) !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}
button.stop, .btn-stop button {
    box-shadow: 0 12px 26px rgba(216, 90, 105, 0.22) !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
}

.live-status { 
    font-size: 1.05em; 
    padding: 14px; 
    border-radius: 12px; 
    background: rgba(79, 179, 196, 0.08) !important; 
    color: #e7edf5 !important; 
    border: 1px solid rgba(79, 179, 196, 0.28) !important;
}
.live-status textarea {
    background: transparent !important;
    color: #e7edf5 !important;
}

.transcript-box textarea { 
    font-family: 'Amiri', 'Traditional Arabic', serif !important; 
    font-size: 1.4em !important; 
    line-height: 2.1 !important; 
    direction: rtl !important; 
    text-align: right !important;
}

.chunk-table { 
    font-size: 0.88em; 
    color: #cfe0ee;
}
.chunk-table th {
    color: #7f93aa;
}

.comparison-panel {
    border: 1px solid rgba(79, 179, 196, 0.22);
    border-radius: 16px;
    padding: 16px;
    background: #131b28;
}

/* Settings gear button */
.sidebar-top-row {
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 0 !important;
}
.settings-gear-btn, .settings-gear-btn button {
    width: 42px !important;
    min-width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    border-radius: 12px !important;
    font-size: 1.15rem !important;
    flex: none !important;
}

/* Settings popup (Qari Mode + Tajweed Detection) — visibility is fully
   controlled by the .modal-hidden class we toggle ourselves, not by
   Gradio's own visible= prop, so there's no conflict with however a given
   Gradio version implements component hiding internally. */
.settings-modal-overlay {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(5, 8, 13, 0.72) !important;
    backdrop-filter: blur(3px);
    display: flex;
    align-items: center !important;
    justify-content: center !important;
    padding: 20px !important;
    z-index: 9999 !important;
}
.settings-modal-overlay.modal-hidden {
    display: none !important;
    pointer-events: none !important;
}
.settings-modal-card {
    max-width: 420px !important;
    width: 100% !important;
    background: #131b28 !important;
    border: 1px solid rgba(79, 179, 196, 0.28) !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 30px 70px rgba(0, 0, 0, 0.55) !important;
}
.settings-close-btn, .settings-close-btn button {
    width: 100% !important;
    margin-top: 6px !important;
    background: #16202f !important;
    color: #cfe0ee !important;
    border: 1px solid rgba(79, 179, 196, 0.35) !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}
.settings-close-btn button:hover {
    background: #1c2a3d !important;
    border-color: #4fb3c4 !important;
}

/* Hide the mic component's native toolbar overflow ("⋯") menu —
   best-effort: Gradio's exact internal class name for this varies by
   version, so several common patterns are covered; any that don't match
   simply do nothing. */
.mic-visualizer .overflow-menu,
.mic-visualizer .dropdown-arrow,
.mic-visualizer [aria-label="More options"],
.mic-visualizer [aria-label="Show menu"],
.mic-visualizer button[title="More options"],
.mic-visualizer .icon-button-wrapper:has(> button[aria-haspopup]) {
    display: none !important;
}
"""

with gr.Blocks(title="Hafizify — Quran ASR") as app:
    gr.Markdown(
        "# 📖 Hafizify — Quran ASR with Real-Time Recitation"
    )

    with gr.Tabs():
        # =====================================================================
        # TAB 1: Live Recitation (NEW)
        # =====================================================================
        with gr.Tab("🎙️ Live Recitation", id="live_tab"):
            gr.Markdown(
                "### Real-Time Quran Recitation\n"
                "Select your surah, configure settings, then click **Start** "
                "and begin reciting. VAD-based chunking splits audio on silence "
                "for accurate word boundaries."
            )

            with gr.Row():
                # ---- Left: Controls ----
                with gr.Column(scale=1, elem_classes=["sidebar-col"]):
                    with gr.Row(elem_classes=["sidebar-top-row"]):
                        settings_btn = gr.Button("⚙️", elem_classes=["settings-gear-btn"], size="sm")
                    live_status = gr.Textbox(
                        label="Session Status",
                        value="⚪ Ready — Configure settings and click Start",
                        interactive=False,
                        elem_classes=["live-status"],
                        visible=False,
                    )

                    with gr.Accordion("📋 Recitation Settings", open=True):
                        live_model_selector = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value="whisper-base-quran-lora",
                            label="🤖 ASR Model",
                            visible=False,
                        )
                        live_asr_engine_selector = gr.Radio(
                            choices=[
                                "Standard (offline, fast)",
                                "Combined (Groq + Local, harakaat-aware, needs internet)",
                            ],
                            value="Standard (offline, fast)",
                            label="⚙️ ASR Engine",
                            info="Combined Mode: requires internet + GPU, more accurate diacritics.",
                            visible=False,
                        )
                        surah_dropdown = gr.Dropdown(
                            choices=SURAH_NAMES,
                            value="1 - Al-Fatiha",
                            label="Surah",
                        )
                        start_ayah_input = gr.Number(
                            value=1,
                            label="Start Ayah",
                            minimum=1,
                            maximum=286,
                            precision=0,
                            visible=False,
                        )
                        live_correction_mode = gr.Dropdown(
                            ["safe", "balanced", "aggressive"],
                            value="balanced",
                            label="Correction Mode",
                            visible=False,
                        )
                        auto_surah_detect_checkbox = gr.Checkbox(
                            value=False,
                            label="Auto Surah Detection",
                            info="Disable automatic surah locking and guessing",
                            visible=False,
                        )

                    with gr.Column(elem_classes=["settings-modal-overlay", "modal-hidden"]) as settings_modal:
                        with gr.Group(elem_classes=["settings-modal-card"]):
                            gr.Markdown("### ⚙️ Advanced Settings")
                            qari_mode_checkbox = gr.Checkbox(
                                value=False,
                                label="🎙️ Interactive Qari Mode",
                                info="Enables interactive TTS feedback to correct recitation mistakes.",
                            )
                            tajweed_toggle = gr.Checkbox(
                                value=False,
                                label="🎯 Active Tajweed Detection",
                                info="On: Combined Groq + Local, harakaat-aware (needs internet). Off: local offline model.",
                            )
                            use_api_combined_checkbox = gr.Checkbox(
                                value=False,
                                label="🌐 Use Remote API for Combined Mode",
                                info="Send Combined Mode chunks to a running api/main.py server (POST /transcribe) "
                                     "instead of loading Groq + the local turbo model in this process. Ignored in "
                                     "Standard Mode.",
                            )
                            api_base_url_input = gr.Textbox(
                                value="http://127.0.0.1:8000",
                                label="API Base URL",
                                info="Only used when 'Use Remote API for Combined Mode' is on.",
                            )
                            close_settings_btn = gr.Button("Close", elem_classes=["settings-close-btn"], size="sm")

                    with gr.Accordion("⚙️ Chunk Settings", open=False, visible=False):
                        use_vad_checkbox = gr.Checkbox(
                            value=True,
                            label="🧠 Use VAD Chunking (recommended)",
                            info="Split audio on silence instead of fixed time windows",
                        )
                        chunk_duration_slider = gr.Slider(
                            minimum=3.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.5,
                            label="Chunk Duration (fallback if VAD off)",
                        )
                        overlap_slider = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=1.5,
                            step=0.5,
                            label="Overlap Duration (fallback if VAD off)",
                        )

                    start_session_btn = gr.Button(
                        "▶️ Start Session",
                        elem_classes=["btn-start"],
                        variant="primary",
                    )

                    mic_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        label="🎤 Tap Record to Start/Stop Reciting",
                        type="numpy",
                        elem_classes=["mic-visualizer"],
                    )

                # ---- Right: Live Output ----
                with gr.Column(scale=2, elem_classes=["main-col"]):
                    guessed_surah_box = gr.Textbox(
                        label="🔍 Guessed Surah (auto-detected)",
                        value="Listening...",
                        interactive=False,
                        visible=False,
                    )

                    surah_badge_html = gr.HTML(
                        label="Surah Lock",
                        value=_format_surah_badge(None),
                        visible=False,
                    )
                    
                    correction_status_box = gr.HTML(
                        label="Qari Mode Status",
                        value="",
                    )

                    live_merged_display = gr.HTML(
                        label="Expected Ayah (Word-by-Word)",
                        value=LiveDisplayFormatter._placeholder_html("Start reciting..."),
                        visible=False,
                    )

                    surah_progress_display = gr.HTML(
                        label="Full Surah Progress",
                        value=LiveDisplayFormatter._placeholder_html("Surah progress will appear here."),
                    )

                    error_panel = gr.HTML(
                        label="Error Analysis",
                        value=LiveDisplayFormatter._error_panel_html(0, 0, 0, 0, 0, 0.0),
                        visible=False,
                    )

                    with gr.Accordion("📝 Raw vs Corrected", open=False, visible=False):
                        with gr.Row():
                            raw_asr_box = gr.Textbox(
                                label="Raw ASR Output",
                                lines=3,
                                interactive=False,
                                rtl=True,
                                elem_classes=["transcript-box"],
                            )
                            corrected_box = gr.Textbox(
                                label="Corrected (Quran-aware)",
                                lines=3,
                                interactive=False,
                                rtl=True,
                                elem_classes=["transcript-box"],
                            )

                    with gr.Accordion("📊 Chunk Details", open=False):
                        chunks_table_md = gr.HTML(
                            value="<div class='chunk-table'>No chunks yet.</div>",
                        )

            # Comparison section (shown after stopping)
            with gr.Accordion("📊 Session Comparison (after stop)", open=False):
                comparison_md = gr.Markdown(
                    value="*Stop the session to see the comparison.*"
                )
                with gr.Row():
                    session_eval_json = gr.JSON(label="Full Session Error Analysis")
                    session_audio_output = gr.Audio(
                        label="Full Session Recording",
                        type="filepath",
                        interactive=False,
                    )

            # ---- Event wiring ----
            settings_btn.click(
                fn=lambda: gr.update(elem_classes=["settings-modal-overlay"]),
                outputs=[settings_modal],
            )
            close_settings_btn.click(
                fn=lambda: gr.update(elem_classes=["settings-modal-overlay", "modal-hidden"]),
                outputs=[settings_modal],
            )

            tajweed_toggle.change(
                fn=lambda active: (
                    "Combined (Groq + Local, harakaat-aware, needs internet)"
                    if active else "Standard (offline, fast)"
                ),
                inputs=[tajweed_toggle],
                outputs=[live_asr_engine_selector],
            )

            # Start Session button loads the model + creates the session BEFORE
            # the mic ever opens, so create_session()'s _ensure_model_loaded()
            # blocking call can never race with the browser's first streaming
            # audio upload (that race caused ffmpeg CouldntDecodeError on Record tap).
            start_session_btn.click(
                fn=start_live_session,
                inputs=[surah_dropdown, start_ayah_input, chunk_duration_slider, overlap_slider, live_model_selector, use_vad_checkbox, auto_surah_detect_checkbox, qari_mode_checkbox, live_asr_engine_selector, use_api_combined_checkbox],
                outputs=[live_status, live_merged_display, surah_progress_display, error_panel, raw_asr_box, corrected_box, chunks_table_md, comparison_md, surah_badge_html, correction_status_box],
            )

            mic_input.stream(
                fn=process_streaming_audio,
                inputs=[mic_input, live_correction_mode, qari_mode_checkbox, live_asr_engine_selector, use_api_combined_checkbox, api_base_url_input],
                outputs=[live_merged_display, surah_progress_display, error_panel, raw_asr_box, corrected_box, chunks_table_md, guessed_surah_box, surah_badge_html, correction_status_box],
            )

            mic_input.stop_recording(
                fn=stop_live_session,
                inputs=[live_correction_mode, qari_mode_checkbox, live_asr_engine_selector, use_api_combined_checkbox, api_base_url_input],
                outputs=[live_status, live_merged_display, surah_progress_display, error_panel, raw_asr_box,
                         corrected_box, chunks_table_md, comparison_md,
                         session_eval_json, guessed_surah_box, surah_badge_html, session_audio_output, correction_status_box],
            )

    app.load(fn=_preload_live_model)

if __name__ == "__main__":
    print("Starting Hafizify Gradio app ...")
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "inbrowser": True,
        "theme": gr.themes.Soft(),
        "css": _APP_CSS,
        "allowed_paths": [os.path.join(BASE_DIR, "recordings")],
    }
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    try:
        app.launch(server_port=server_port, **launch_kwargs)
    except OSError as exc:
        if "Cannot find empty port" in str(exc) and server_port == 7860:
            fallback_port = 7861
            print(f"Port 7860 busy, retrying on {fallback_port}...")
            app.launch(server_port=fallback_port, **launch_kwargs)
        else:
            raise