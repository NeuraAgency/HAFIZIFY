"""
hybrid_diacritic_pipeline.py
-----------------------------
Combined Model mode — Phase 1 of masterplan.md.

Runs two ASR models on the same audio chunk and merges their output into a
single diacritized (harakaat-carrying) transcription:

  - Groq cloud Whisper (whisper-large-v3)   -> reliable consonant backbone
  - local turbo LoRA model (this repo)      -> the only model that outputs
                                                Arabic diacritics (harakaat)

The merge takes Groq's word/consonant sequence as the backbone and injects
the local model's diacritics onto it by character position. This module
does NOT decide whether the recitation was correct — it only reconstructs
"what was actually said, with vowels". Error detection against the
reference ayah is a separate concern (see harakaat_error_detector.py) and
deliberately stays out of this file so the two responsibilities don't get
tangled.

Design note (read before touching the merge logic):
    The original Colab draft of this pipeline had a hardcoded override that
    force-injected a fixed diacritization whenever the word "مالك" appeared,
    regardless of what the reciter actually said. That's removed here on
    purpose: forcing a "known correct" diacritization onto the combined
    output would make it impossible for harakaat_error_detector.py to ever
    catch a real vowel mistake on that word — it would always look correct.
    The general character-position injection algorithm below already
    handles "مالك" like any other word; no special case needed.

Standard mode (the existing single offline CT2 model) is completely
unaffected by this file. Nothing here is imported or executed unless
Combined Mode is explicitly selected in the UI (wired in a later phase).
"""

import io
import os
import re
import wave
import difflib
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reuse the existing, already-safe Groq client (env-based key, no duplicate
# API key handling in this file).
# ---------------------------------------------------------------------------
from groq_transcriber import get_groq_transcriber

DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")
OTHMANI_GLYPHS_RE = re.compile(r"[\u0671\u0653\u0654\u0655]")

LOCAL_MODEL_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"


# ---------------------------------------------------------------------------
# Lazy local-model loader
# ---------------------------------------------------------------------------
# Loaded ONLY on first call to transcribe_local() / run_combined_transcription().
# Standard-mode sessions never trigger this, so app startup time/memory for
# anyone not using Combined Mode is unaffected.
#
# Auto-detects CUDA (RTX 2060 Super in dev, GPU server in deployment) and
# falls back to CPU if unavailable, rather than hard-crashing.
_local_pipeline = None


def _load_local_pipeline():
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

    device_available = torch.cuda.is_available()
    dtype = torch.float16 if device_available else torch.float32

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        LOCAL_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device_available else None,
    )
    if not device_available:
        model = model.to("cpu")

    _local_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=2,
        batch_size=1,
        device=0 if device_available else -1,
    )
    return _local_pipeline


# ---------------------------------------------------------------------------
# Per-engine transcription
# ---------------------------------------------------------------------------

def transcribe_groq(chunk_numpy: np.ndarray, sample_rate: int = 16000) -> str:
    """Undiacritized consonant-backbone transcription via the existing Groq client."""
    try:
        return get_groq_transcriber().transcribe_array(chunk_numpy, sample_rate=sample_rate)
    except Exception:
        return ""


def transcribe_local(chunk_numpy: np.ndarray, sample_rate: int = 16000) -> str:
    """Diacritized transcription via the local turbo LoRA model."""
    try:
        engine = _load_local_pipeline()
        result = engine(
            {"raw": chunk_numpy, "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe", "language": "arabic"},
        )
        return result["text"].strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Merge logic (cleaned up from the Colab draft)
# ---------------------------------------------------------------------------

def smart_normalize_word(word: str) -> str:
    if not word:
        return ""
    cleaned = DIACRITICS_PATTERN.sub("", OTHMANI_GLYPHS_RE.sub("", word))
    cleaned = cleaned.replace("إ", "ا").replace("أ", "ا").replace("ٱ", "ا").replace("ٰ", "")
    return cleaned


def inject_vowels_by_character_position(groq_word: str, local_voweled_word: str) -> str:
    """Rebuild groq_word's consonant skeleton with local_voweled_word's diacritics,
    matched by character position. Falls back to whichever side already has
    diacritics if the two consonant skeletons don't line up in length."""
    g_clean = DIACRITICS_PATTERN.sub("", OTHMANI_GLYPHS_RE.sub("", groq_word))
    l_clean = DIACRITICS_PATTERN.sub("", OTHMANI_GLYPHS_RE.sub("", local_voweled_word))

    g_clean_norm = g_clean.replace("إ", "ا").replace("أ", "ا").replace("ٱ", "ا")
    l_clean_norm = l_clean.replace("إ", "ا").replace("أ", "ا").replace("ٱ", "ا")

    if len(g_clean_norm) != len(l_clean_norm):
        return local_voweled_word if DIACRITICS_PATTERN.search(local_voweled_word) else groq_word

    vowel_map = {}
    vowel_buffer = []
    char_idx = 0

    for char in local_voweled_word:
        if DIACRITICS_PATTERN.match(char):
            vowel_buffer.append(char)
        else:
            if vowel_buffer:
                vowel_map[char_idx - 1] = "".join(vowel_buffer)
                vowel_buffer = []
            char_idx += 1
    if vowel_buffer:
        vowel_map[char_idx - 1] = "".join(vowel_buffer)

    rebuilt = []
    for idx, char in enumerate(g_clean):
        rebuilt.append(char)
        if idx in vowel_map:
            rebuilt.append(vowel_map[idx])

    return "".join(rebuilt)


def run_hybrid_combination_logic(groq_raw_text: str, local_raw_text: str) -> str:
    """Word-align Groq vs local output, keep Groq's consonants, inject local's
    diacritics. No reference/ground-truth is consulted here on purpose —
    see the module docstring."""
    groq_words = groq_raw_text.strip().split()
    local_words = local_raw_text.strip().split()

    groq_stripped = [smart_normalize_word(w) for w in groq_words]
    local_stripped = [smart_normalize_word(w) for w in local_words]

    matcher = difflib.SequenceMatcher(None, groq_stripped, local_stripped)
    local_word_mapping = {}
    for tag, g_start, g_end, l_start, l_end in matcher.get_opcodes():
        if tag == "equal":
            for g_idx, l_idx in zip(range(g_start, g_end), range(l_start, l_end)):
                local_word_mapping[g_idx] = local_words[l_idx]

    final_words = []
    for idx, groq_word in enumerate(groq_words):
        if idx in local_word_mapping:
            patched = inject_vowels_by_character_position(groq_word, local_word_mapping[idx])
        elif idx < len(local_words):
            patched = inject_vowels_by_character_position(groq_word, local_words[idx])
        else:
            patched = groq_word

        # Fixed version of the original "kasra eradication" rule — the old
        # code had two branches that produced an identical result; only the
        # 'starts with اهد' condition ever mattered.
        if patched.startswith("اِ"):
            patched = ("اهْ" + patched[2:]) if groq_word.startswith("اهد") else ("ا" + patched[2:])

        final_words.append(patched)

    return " ".join(final_words)


# ---------------------------------------------------------------------------
# Public entry point — this is what the app calls per VAD chunk
# ---------------------------------------------------------------------------

def run_combined_transcription(chunk_numpy: np.ndarray, sample_rate: int = 16000) -> dict:
    """Runs Groq + local model concurrently on one audio chunk and returns
    the merged diacritized transcription plus both raw outputs (useful for
    debugging / eval, and for the standalone test harness below)."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        groq_future = executor.submit(transcribe_groq, chunk_numpy, sample_rate)
        local_future = executor.submit(transcribe_local, chunk_numpy, sample_rate)
        groq_text = groq_future.result()
        local_text = local_future.result()

    combined_text = run_hybrid_combination_logic(groq_text, local_text) if groq_text else local_text

    return {
        "groq_text": groq_text,
        "local_text": local_text,
        "combined_text": combined_text,
    }


# ---------------------------------------------------------------------------
# Standalone test harness (Phase 1 verification only)
# ---------------------------------------------------------------------------
# NOT used by the live app — the live app reuses session_manager.py's VAD
# chunking (per masterplan.md §4.1/§5). This is purely for testing this
# module in isolation against a full audio file before wiring it in.

def _compute_levenshtein(a, b):
    m, n = len(a), len(b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def _wer_cer(predicted: str, truth: str):
    if not predicted:
        return 100.0, 100.0
    p_words, t_words = predicted.strip().split(), truth.strip().split()
    w = _compute_levenshtein(p_words, t_words) / max(1, len(t_words)) * 100
    c = _compute_levenshtein(list(predicted), list(truth)) / max(1, len(truth)) * 100
    return w, c


def _standalone_test(audio_path: str, ground_truth: Optional[str] = None):
    """Quick CLI check: `python hybrid_diacritic_pipeline.py path/to.wav`"""
    import torch
    import soundfile as sf
    import librosa

    print("Loading Silero VAD (test harness only)...")
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False)
    get_speech_timestamps, _, read_audio, _, _ = utils

    SAMPLE_RATE = 16000
    audio_tensor = read_audio(audio_path, sampling_rate=SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        audio_tensor, vad_model, sampling_rate=SAMPLE_RATE,
        threshold=0.25, min_speech_duration_ms=150, min_silence_duration_ms=500,
    )

    groq_segments, local_segments, combined_segments = [], [], []
    for ts in timestamps:
        chunk = audio_tensor[ts["start"]:ts["end"]].numpy()
        if len(chunk) < int(0.5 * SAMPLE_RATE):
            continue
        result = run_combined_transcription(chunk, SAMPLE_RATE)
        if result["groq_text"]:
            groq_segments.append(result["groq_text"])
        if result["local_text"]:
            local_segments.append(result["local_text"])
        if result["combined_text"]:
            combined_segments.append(result["combined_text"])

    final_groq = " ".join(groq_segments)
    final_local = " ".join(local_segments)
    final_combined = " ".join(combined_segments)

    print(f"\nGroq:     {final_groq}")
    print(f"Local:    {final_local}")
    print(f"Combined: {final_combined}")

    if ground_truth:
        for label, text in (("Groq", final_groq), ("Local", final_local), ("Combined", final_combined)):
            w, c = _wer_cer(text, ground_truth)
            print(f"{label}: WER {w:.2f}% | CER {c:.2f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_diacritic_pipeline.py <audio_path> [ground_truth_text]")
    else:
        _standalone_test(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
