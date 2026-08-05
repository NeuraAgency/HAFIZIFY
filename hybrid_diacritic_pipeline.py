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
from typing import Dict, List, Optional, Tuple

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

# Docker's docker-compose.yml mounts a local copy of this exact model at
# /app/whisper-l-v3-turbo-quran-lora-dataset-mix specifically so the
# container never has to hit the network for it (already contains
# config.json/model.safetensors/tokenizer files — a complete, ready-to-load
# checkpoint). _load_local_pipeline() below now loads from there first when
# present. LOCAL_MODEL_ID (the HF Hub repo id) is the fallback for any
# environment without that mount (e.g. running this file directly outside
# Docker) — kept as-is for that case, cache-then-download exactly like
# before.
_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-l-v3-turbo-quran-lora-dataset-mix")


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

    if os.path.isdir(_LOCAL_MODEL_DIR) and os.path.isfile(os.path.join(_LOCAL_MODEL_DIR, "config.json")):
        # The mounted local checkpoint (see the _LOCAL_MODEL_DIR comment
        # above) — loaded straight off disk, no network involved at all.
        # This is the path Docker deployments should always hit.
        processor = AutoProcessor.from_pretrained(_LOCAL_MODEL_DIR)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _LOCAL_MODEL_DIR,
            torch_dtype=dtype,
            device_map="auto" if device_available else None,
        )
        print(f"[Combined Mode] Loaded local turbo model from mounted directory {_LOCAL_MODEL_DIR} (no network call).")
    else:
        # Fallback for environments without the local mount (e.g. running
        # this file directly, outside Docker). Try the local HF cache first
        # (local_files_only=True) so a machine with no/unreliable internet
        # never even attempts the network "check for updates" HEAD request.
        # Observed failure mode without this: on a machine with DNS
        # resolution failing (getaddrinfo failed), that request retries 5
        # times, fails, and leaves huggingface_hub's underlying HTTP client
        # in a broken state ("Cannot send a request, as the client has been
        # closed") — which silently killed every subsequent local-model call
        # for the rest of the process's life, not just the first one. Only
        # falls through to a real network download if nothing is cached yet
        # (first run on a fresh machine).
        try:
            processor = AutoProcessor.from_pretrained(LOCAL_MODEL_ID, local_files_only=True)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                LOCAL_MODEL_ID,
                torch_dtype=dtype,
                device_map="auto" if device_available else None,
                local_files_only=True,
            )
            print("[Combined Mode] Loaded local turbo model from HF cache (offline, no network call).")
        except Exception as e:
            print(f"[Combined Mode] Not found in local HF cache ({e!r}); downloading from HuggingFace Hub...")
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
    except Exception as e:
        print(f"[Combined Mode] Groq transcription failed, returning empty text: {e!r}")
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
    except Exception as e:
        print(f"[Combined Mode] Local model transcription failed, returning empty text "
              f"(chunk length {len(chunk_numpy) / sample_rate:.2f}s): {e!r}")
        return ""


def preload_local_pipeline() -> None:
    """Public wrapper around _load_local_pipeline() so callers outside this
    module (app.py's start_live_session()) can force the local turbo
    model to load synchronously at session-start time, instead of it
    lazy-loading on the first streamed audio chunk. A no-op if already
    loaded (e.g. a later session in the same app run)."""
    _load_local_pipeline()


# ---------------------------------------------------------------------------
# Quran vocabulary (for word-choice arbitration only — see run_hybrid_
# combination_logic below). NOT a per-ayah reference lookup: this is the
# full set of every word that appears anywhere in the Quran, with no
# information about which ayah/position is currently expected. Checking
# "is this a real Quranic word at all" is a fair signal to arbitrate between
# two ASR engines' disagreeing word choices — it never tells the merge what
# the CORRECT word at this position was, only whether a candidate is a
# plausible word in the language domain at all. This is a different, weaker
# signal than consulting the matched ayah's reference text, which is what
# the module docstring's "never consult ground truth" warning is about.
# ---------------------------------------------------------------------------
_quran_vocab: Optional[set] = None


def _load_quran_vocab() -> set:
    global _quran_vocab
    if _quran_vocab is not None:
        return _quran_vocab

    _quran_vocab = set()
    try:
        import json
        json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fyp_model", "all_ayat.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break
        for value in data.values():
            text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
            for word in text.split():
                skeleton = smart_normalize_word(word)
                if skeleton:
                    _quran_vocab.add(skeleton)
        print(f"[Combined Mode] Loaded {len(_quran_vocab)} unique Quran word skeletons for merge arbitration.")
    except Exception as e:
        print(f"[Combined Mode] Could not load Quran vocabulary for merge arbitration ({e!r}); "
              f"word-choice arbitration will default to Groq on every disagreement.")
        _quran_vocab = set()

    return _quran_vocab


# ---------------------------------------------------------------------------
# Merge logic (cleaned up from the Colab draft)
# ---------------------------------------------------------------------------

def smart_normalize_word(word: str) -> str:
    if not word:
        return ""
    cleaned = DIACRITICS_PATTERN.sub("", OTHMANI_GLYPHS_RE.sub("", word))
    cleaned = cleaned.replace("إ", "ا").replace("أ", "ا").replace("ٱ", "ا").replace("ٰ", "")
    return cleaned


# Max normalized-skeleton edit distance allowed before two same-length,
# non-identical words in a "replace" block are still treated as a probable
# same-word pairing for vowel-borrowing purposes (see run_hybrid_combination_logic).
# Kept small and length-scaled on purpose: short words (<3 letters) are too
# ambiguous to fuzzy-match at all (e.g. من vs لن is a real, different word,
# not ASR noise), so those require an exact match.
_FUZZY_MATCH_MAX_LEN_FOR_TIGHT_THRESHOLD = 6


def _is_close_enough(groq_skeleton: str, local_skeleton: str) -> bool:
    """Same-length-only, small-edit-distance check used to decide whether a
    Groq/local word pair that difflib did NOT consider an exact match is
    still close enough to be "the same word, mis-heard by one letter" — as
    opposed to two genuinely different words that happen to share a slot.
    Deliberately conservative: exact length match required (character-
    position vowel injection needs that anyway), and very short words
    aren't fuzzy-matched at all since a 1-letter edit on a 2-3 letter word
    is usually a different word, not noise."""
    if len(groq_skeleton) != len(local_skeleton):
        return False
    if len(groq_skeleton) < 3:
        return groq_skeleton == local_skeleton
    distance = _compute_levenshtein(groq_skeleton, local_skeleton)
    threshold = 1 if len(groq_skeleton) <= _FUZZY_MATCH_MAX_LEN_FOR_TIGHT_THRESHOLD else 2
    return distance <= threshold


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


def _strip_diacritics_only(word: str) -> str:
    """Remove harakaat/othmani glyphs but keep the real letters as spoken
    (no alef-variant folding) — used when a word is chosen as the combined1
    backbone so its actual spelling survives, only vowels are dropped."""
    return DIACRITICS_PATTERN.sub("", OTHMANI_GLYPHS_RE.sub("", word))


def _choose_word_backbone(g_word: str, l_word: str, g_skel: str, l_skel: str):
    """Stage 2 (combined1) arbitration for one aligned word slot.

    This compares Groq's output to the LOCAL model's output — never to the
    reference/expected ayah — so it is a fair ensemble decision between two
    ASR engines, not the "answer key" problem the module docstring warns
    about (nothing here is told which word was supposed to be recited).

    Signal used: Quran-vocabulary membership (see _load_quran_vocab). If
    the local model's word skeleton is an actual word that appears
    somewhere in the Quran and Groq's isn't, that's real evidence Groq
    hallucinated a non-Quranic word here — take local's word instead.
    Every other case defaults to Groq, preserving the existing "Groq is the
    backbone" behavior when there's no such signal.

    Returns (chosen_bare_word, source) where source is 'groq' or 'local'.
    """
    vocab = _load_quran_vocab()
    g_in_vocab = bool(g_skel) and g_skel in vocab
    l_in_vocab = bool(l_skel) and l_skel in vocab
    if l_in_vocab and not g_in_vocab:
        return _strip_diacritics_only(l_word), "local"
    return g_word, "groq"


def _align_words_to_reference(words: List[str], words_stripped: List[str], ref_stripped: List[str]) -> Dict[int, dict]:
    """Align one engine's word list to the reference word order — same
    difflib-opcode approach get_word_error_annotations() in quran_guard.py
    uses (reference first, hypothesis second), so combined1's word choice
    and word_errors' scoring are judging "did this word match the reference"
    the same way instead of two independently-drifting implementations.

    Returns {ref_idx: {"word": <raw engine word>, "matched": bool}} for every
    reference position this engine's output reached. matched=True means this
    engine's word equals the reference at this position (difflib 'equal');
    matched=False means this engine said something at this position but it
    didn't match ('replace'). A reference position with no entry at all means
    this engine said nothing there ('delete' — genuinely skipped/unreached).
    """
    aligned: Dict[int, dict] = {}
    matcher = difflib.SequenceMatcher(None, ref_stripped, words_stripped, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                aligned[i1 + offset] = {"word": words[j1 + offset], "matched": True}
        elif tag == "replace":
            shared = min(i2 - i1, j2 - j1)
            for offset in range(shared):
                aligned[i1 + offset] = {"word": words[j1 + offset], "matched": False}
    return aligned


def _build_reference_aligned_combined1(
    groq_words: List[str], local_words: List[str],
    groq_stripped: List[str], local_stripped: List[str],
    ref_text: str,
) -> Tuple[List[str], Dict[int, str]]:
    """Stage 2 (combined1), reference-aligned version — used once the ayah is
    known (ref_text set). Per Hamza's spec: build combined1 word-by-word in
    REFERENCE word order, taking whichever engine actually matched the
    reference at that position (same principle get_word_error_annotations_
    best_of() already uses for word_errors scoring, applied here to the
    transcript itself instead of just the score).

    Priority per reference position:
      1. Groq matched the reference here -> use Groq's word.
      2. Else local matched the reference here -> use local's word.
      3. Else, whichever engine said SOMETHING at this position (even if
         wrong) is kept as-is -> a genuine mistake stays visible, never
         replaced by the reference word. Groq is preferred when both engines
         got it wrong, same tie-break the old vocab-based backbone used.
      4. Neither engine reached this position at all -> skip it (genuinely
         unspoken, matches how word_errors would mark it 'missing').

    Returns (combined1_words, vowel_donor) where vowel_donor[i] is the raw
    (possibly diacritized) local-engine word to pull harakaat from for
    combined1_words[i], when one is available.
    """
    ref_words = ref_text.strip().split()
    ref_stripped = [smart_normalize_word(w) for w in ref_words]

    groq_aligned = _align_words_to_reference(groq_words, groq_stripped, ref_stripped)
    local_aligned = _align_words_to_reference(local_words, local_stripped, ref_stripped)

    combined1_words: List[str] = []
    vowel_donor: Dict[int, str] = {}

    for ref_idx in range(len(ref_words)):
        g = groq_aligned.get(ref_idx)
        l = local_aligned.get(ref_idx)

        if g and g["matched"]:
            chosen_word, donor = g["word"], (l["word"] if l else None)
        elif l and l["matched"]:
            chosen_word, donor = _strip_diacritics_only(l["word"]), l["word"]
        elif g:
            chosen_word, donor = g["word"], (l["word"] if l else None)
        elif l:
            chosen_word, donor = _strip_diacritics_only(l["word"]), l["word"]
        else:
            continue  # neither engine said anything here — genuinely unspoken

        pos = len(combined1_words)
        combined1_words.append(chosen_word)
        if donor:
            vowel_donor[pos] = donor

    return combined1_words, vowel_donor


def run_hybrid_combination_logic(groq_raw_text: str, local_raw_text: str, ref_text: str = None) -> dict:
    """Explicit stages, per Hamza's spec:

      1. groq / local_normalized — each engine's own text; local_normalized
         is local's output with diacritics stripped and alef variants
         folded (its word skeletons), so word-CHOICE comparison in stage 2
         isn't biased by which side happens to carry vowels.

      2. combined1 — word-choice merge, best of both engines:
         - No reference known yet (ref_text=None, the cheap first pass
           whose only job is finding which ayah this is): pure two-engine
           arbitration via _choose_word_backbone — keep Groq's word unless
           local's is real Quran vocabulary and Groq's isn't. Never
           consults ref_text, per this module's "never consult ground
           truth without a match" rule.
         - Reference known (ref_text set, the second pass after ayah
           match): built word-by-word in REFERENCE order by
           _build_reference_aligned_combined1 — whichever engine actually
           matches the reference at each position wins; if neither does,
           the mistake is kept exactly as whichever engine said it (never
           papered over by the reference word). This is the "best of both
           against the actual Quranic text" merge.

      3. combined (final) — harakaat injection on top of whatever combined1
         decided: local's diacritics are layered onto those words wherever
         a confident alignment/donor exists.

    Called twice in the live pipeline: once with ref_text=None (a cheap
    first pass, just to find which ayah this is), and once more with
    ref_text set to that matched ayah's reference text (the real, final
    transcript) — see realtime_streamer.py's process_chunk_combined().
    """
    groq_words = groq_raw_text.strip().split()
    local_words = local_raw_text.strip().split()

    groq_stripped = [smart_normalize_word(w) for w in groq_words]
    local_stripped = [smart_normalize_word(w) for w in local_words]
    local_normalized = " ".join(local_stripped)

    if ref_text:
        combined1_words, vowel_donor = _build_reference_aligned_combined1(
            groq_words, local_words, groq_stripped, local_stripped, ref_text,
        )
    else:
        matcher = difflib.SequenceMatcher(None, groq_stripped, local_stripped)

        # backbone[i] = (bare_word, source) — the combined1 decision for groq
        # word slot i. Defaults to Groq's own bare word for any slot an opcode
        # block below doesn't touch (e.g. groq-only "delete" blocks).
        backbone = {idx: (word, "groq") for idx, word in enumerate(groq_words)}
        # local_word_mapping[i] = the actual local word (WITH diacritics) to
        # pull vowels from for slot i, when the chosen backbone word came from
        # Groq and aligned closely enough to local's word at that slot, OR is
        # itself the chosen local word (re-affirmed for harakaat injection).
        local_word_mapping = {}

        for tag, g_start, g_end, l_start, l_end in matcher.get_opcodes():
            if tag == "equal":
                for g_idx, l_idx in zip(range(g_start, g_end), range(l_start, l_end)):
                    local_word_mapping[g_idx] = local_words[l_idx]

            elif tag == "replace":
                shared = min(g_end - g_start, l_end - l_start)
                for offset in range(shared):
                    g_idx = g_start + offset
                    l_idx = l_start + offset
                    g_word, l_word = groq_words[g_idx], local_words[l_idx]
                    g_skel, l_skel = groq_stripped[g_idx], local_stripped[l_idx]

                    chosen_word, source = _choose_word_backbone(g_word, l_word, g_skel, l_skel)
                    backbone[g_idx] = (chosen_word, source)

                    if source == "local":
                        local_word_mapping[g_idx] = l_word
                    elif _is_close_enough(g_skel, l_skel):
                        local_word_mapping[g_idx] = l_word

        combined1_words = [backbone[idx][0] for idx in range(len(groq_words))]
        vowel_donor = local_word_mapping

    combined1 = " ".join(combined1_words)

    # Stage 3 — harakaat injection onto combined1.
    final_words = []
    for idx, word in enumerate(combined1_words):
        if idx in vowel_donor:
            patched = inject_vowels_by_character_position(word, vowel_donor[idx])
        else:
            patched = word

        if patched.startswith("اِ"):
            patched = ("اهْ" + patched[2:]) if word.startswith("اهد") else ("ا" + patched[2:])

        final_words.append(patched)

    return {
        "local_normalized": local_normalized,
        "combined1": combined1,
        "combined": " ".join(final_words),
    }


# ---------------------------------------------------------------------------
# Anti-hallucination guard
# ---------------------------------------------------------------------------

def _collapse_exact_repeats(text: str) -> str:
    """Collapse a transcript that is an exact word-for-word repeat of a
    shorter phrase down to a single occurrence.

    Whisper (both OpenAI's and Groq's hosted whisper-large-v3) has a known
    failure mode of repeating the last phrase it heard into trailing/leading
    silence within its fixed-size decode window, especially on short clips
    at temperature=0.0. Groq's REST API doesn't expose the HF-only
    repetition_penalty / no_repeat_ngram_size generate() kwargs that guard
    against this elsewhere in the codebase, so this catches it after the
    fact instead, on whichever engine's raw text triggered it.

    Deliberately conservative: only collapses an EXACT, whole-sequence
    repeat (the word list tiles perfectly into 2+ identical blocks). A
    genuinely different second half — e.g. two different ayahs recited
    back to back — will not tile exactly and is left untouched.
    """
    words = text.strip().split()
    n = len(words)
    if n < 2:
        return text
    for repeat_len in range(1, n // 2 + 1):
        if n % repeat_len != 0:
            continue
        block = words[:repeat_len]
        if all(words[i:i + repeat_len] == block for i in range(0, n, repeat_len)):
            return " ".join(block)
    return text


# ---------------------------------------------------------------------------
# Public entry point — this is what the app calls per VAD chunk
# ---------------------------------------------------------------------------

def run_combined_transcription(chunk_numpy: np.ndarray, sample_rate: int = 16000, ref_text: Optional[str] = None) -> dict:
    """Runs Groq + local model concurrently on one audio chunk and returns
    the merged diacritized transcription plus both raw outputs (useful for
    debugging / eval, and for the standalone test harness below).

    ref_text: optional matched-ayah reference text (see
    run_hybrid_combination_logic's stage 2b docstring). Pass None for a
    cheap first pass with no ayah context yet; pass the matched ayah's text
    for a second pass once one is known, to get the reference-verified
    merge. The Groq/local decode itself never re-runs between passes —
    callers should reuse groq_text/local_text from the first pass and call
    run_hybrid_combination_logic() directly for a second pass instead of
    calling this function twice. This function always does a fresh decode,
    so it stays the single-pass entry point (used by the standalone test
    harness and any ref_text=None caller).
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        groq_future = executor.submit(transcribe_groq, chunk_numpy, sample_rate)
        local_future = executor.submit(transcribe_local, chunk_numpy, sample_rate)
        groq_text = groq_future.result()
        local_text = local_future.result()

    # Guard both sides against the trailing-silence repeat hallucination
    # before they ever reach the merge — see _collapse_exact_repeats().
    groq_text = _collapse_exact_repeats(groq_text)
    local_text = _collapse_exact_repeats(local_text)

    if groq_text:
        stages = run_hybrid_combination_logic(groq_text, local_text, ref_text=ref_text)
    else:
        # No Groq output this chunk — nothing to build a backbone from, so
        # combined1/combined both just fall back to local's own text (same
        # fallback behavior as before this change).
        stages = {"local_normalized": local_text, "combined1": local_text, "combined": local_text}

    combined_text = stages["combined"]

    if groq_text and local_text and combined_text == groq_text and DIACRITICS_PATTERN.search(local_text) and not DIACRITICS_PATTERN.search(combined_text):
        print(
            "[Combined Mode] Merge produced zero diacritic injections despite local "
            f"having them — groq_text={groq_text!r} local_text={local_text!r}"
        )

    return {
        "groq_text": groq_text,
        "local_text": local_text,
        # New (per Hamza's 3-stage spec): local's text with diacritics
        # stripped/alef-folded, and the word-choice-only merge BEFORE
        # harakaat injection — exposed for debugging/eval, not consumed by
        # the live pipeline yet.
        "local_normalized": stages["local_normalized"],
        "combined1": stages["combined1"],
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
