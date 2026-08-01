"""
harakaat_error_detector.py
----------------------------
Combined Model mode — Phase 2 of masterplan.md.

Compares a diacritized ASR transcription (from hybrid_diacritic_pipeline.py)
against the expected ayah's diacritized reference text and classifies each
word as:

    "ok"              - consonants and diacritics both match
    "harakaat_error"  - consonant skeleton matches, diacritics differ
                         (vowel/tajweed-mark mistake — NEW, previously
                         undetectable anywhere in this codebase, since
                         quran_guard.normalize_arabic() strips all
                         diacritics before comparing)
    "makhraj_error"   - consonant skeleton itself differs (wrong word/
                         wrong letter — already detectable elsewhere via
                         the existing guard/Viterbi pipeline; flagged here
                         too so Combined Mode's output is self-consistent)

Reference source: fyp_model/all_ayat.json. Phase 0 confirmed 6,215/6,235
ayahs (99.7%) carry full Uthmani tashkeel in `text` — see masterplan.md §2.
Two edge cases that file has:
    - "1_1" (the Basmala) has no key at all -> skip_reason="no_reference"
    - ~20 ayahs have no diacritics in `text` -> skip_reason="reference_not_diacritized"

IMPORTANT — do not reuse RealtimeStreamer's self._ayah_map here:
    fyp_model/quran_guard.py::load_all_ayat_json() runs every ayah through
    normalize_arabic() before storing it, which strips ALL diacritics. That
    map (keyed by (surah, ayah) tuples -> plain undiacritized strings) is
    exactly right for the existing matching/correction pipeline and exactly
    wrong for this module. This file loads fyp_model/all_ayat.json a second
    time, independently, keeping the raw diacritized text. Same source
    file, two different in-memory representations for two different jobs —
    this module does not touch quran_guard.py or its normalization at all.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")
_ALEF_VARIANTS = re.compile(r"[إأٱآ]")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AYAT_JSON = os.path.join(_BASE_DIR, "fyp_model", "all_ayat.json")

_diacritized_ayah_map: Optional[Dict[Tuple[int, int], str]] = None


def load_diacritized_ayat_map(json_path: str = _DEFAULT_AYAT_JSON) -> Dict[Tuple[int, int], str]:
    """Load fyp_model/all_ayat.json keeping full tashkeel, keyed by
    (surah, ayah) tuples to match the rest of the app's convention. Cached
    at module level after first load — call this once, not per chunk."""
    global _diacritized_ayah_map
    if _diacritized_ayah_map is not None:
        return _diacritized_ayah_map

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("tafsir", "all_ayat", "verses", "data"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
                break

    ayah_map: Dict[Tuple[int, int], str] = {}
    for key, value in data.items():
        if not (isinstance(key, str) and "_" in key):
            continue
        try:
            surah_str, ayah_str = key.split("_", 1)
            surah, ayah = int(surah_str), int(ayah_str)
        except Exception:
            continue
        text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
        if text:
            ayah_map[(surah, ayah)] = text  # deliberately NOT normalize_arabic()'d

    _diacritized_ayah_map = ayah_map
    return ayah_map


def _strip_diacritics(word: str) -> str:
    return DIACRITICS_PATTERN.sub("", word)


def _normalize_skeleton(word: str) -> str:
    """Consonant-only normalization, matching the conflation rules already
    used elsewhere in this codebase (quran_guard / error_analysis) so that
    'differs in consonants' means the same thing in every module."""
    word = _strip_diacritics(word)
    word = _ALEF_VARIANTS.sub("ا", word)
    word = word.replace("ى", "ي").replace("ة", "ه")
    return word


@dataclass
class WordAnnotation:
    index: int
    predicted_word: str
    reference_word: Optional[str]
    status: str  # "ok" | "harakaat_error" | "makhraj_error" | "missing" | "extra"


@dataclass
class HarakaatCheckResult:
    words: list = field(default_factory=list)
    harakaat_error_count: int = 0
    makhraj_error_count: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None


def get_reference_text(
    surah: int,
    ayah: int,
    ayah_map: Optional[Dict[Tuple[int, int], str]] = None,
) -> Optional[str]:
    """Look up the diacritized reference text for one ayah. Uses this
    module's own diacritized map (load_diacritized_ayat_map), NOT
    RealtimeStreamer's self._ayah_map — see module docstring for why."""
    if ayah_map is None:
        ayah_map = load_diacritized_ayat_map()
    return ayah_map.get((surah, ayah)) or None


def detect_harakaat_errors(
    predicted_diacritized_text: str,
    surah: int,
    ayah: int,
    ayah_map: Optional[Dict[Tuple[int, int], str]] = None,
) -> HarakaatCheckResult:
    """Main entry point. Returns word-level annotations plus counts.

    ayah_map is optional — if not passed, this module lazy-loads and caches
    its own diacritized copy of fyp_model/all_ayat.json on first call
    (see load_diacritized_ayat_map). Pass it explicitly only to override
    (e.g. tests) or to avoid a redundant load if the caller already has one.

    Safe by construction: any lookup miss or missing-diacritics case
    returns skipped=True rather than raising or guessing, so a Combined
    Mode chunk with no usable reference never crashes the session worker.
    """
    if surah == 1 and ayah == 1:
        return HarakaatCheckResult(skipped=True, skip_reason="no_reference")

    reference_text = get_reference_text(surah, ayah, ayah_map)
    if not reference_text:
        return HarakaatCheckResult(skipped=True, skip_reason="no_reference")

    if not DIACRITICS_PATTERN.search(reference_text):
        return HarakaatCheckResult(skipped=True, skip_reason="reference_not_diacritized")

    pred_words = predicted_diacritized_text.strip().split()
    ref_words = reference_text.strip().split()

    annotations = []
    harakaat_errors = 0
    makhraj_errors = 0

    # Same alignment strategy as error_analysis.py's align_words: difflib
    # over the normalized (consonant-only) skeleton, so word-count drift
    # (missing/extra words) doesn't misalign the rest of the comparison.
    import difflib
    pred_skeleton = [_normalize_skeleton(w) for w in pred_words]
    ref_skeleton = [_normalize_skeleton(w) for w in ref_words]
    matcher = difflib.SequenceMatcher(None, ref_skeleton, pred_skeleton)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, (ref_idx, pred_idx) in enumerate(zip(range(i1, i2), range(j1, j2))):
                ref_word, pred_word = ref_words[ref_idx], pred_words[pred_idx]
                if ref_word == pred_word:
                    status = "ok"
                else:
                    status = "harakaat_error"
                    harakaat_errors += 1
                annotations.append(WordAnnotation(pred_idx, pred_word, ref_word, status))
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                pred_idx = j1 + k if j1 + k < j2 else None
                ref_idx = i1 + k if i1 + k < i2 else None
                pred_word = pred_words[pred_idx] if pred_idx is not None else None
                ref_word = ref_words[ref_idx] if ref_idx is not None else None
                if pred_word is not None:
                    makhraj_errors += 1
                    annotations.append(WordAnnotation(pred_idx, pred_word, ref_word, "makhraj_error"))
        elif tag == "delete":
            for ref_idx in range(i1, i2):
                annotations.append(WordAnnotation(-1, "", ref_words[ref_idx], "missing"))
        elif tag == "insert":
            for pred_idx in range(j1, j2):
                annotations.append(WordAnnotation(pred_idx, pred_words[pred_idx], None, "extra"))

    return HarakaatCheckResult(
        words=annotations,
        harakaat_error_count=harakaat_errors,
        makhraj_error_count=makhraj_errors,
    )


if __name__ == "__main__":
    # Minimal unit test (Phase 2 verification) - hand-written pairs, no
    # audio/model needed.
    fake_ayah_map = {
        "1_3": {"text": "الرَّحْمَٰنِ الرَّحِيمِ"},
    }

    print("Test 1: exact match (expect all 'ok')")
    r = detect_harakaat_errors("الرَّحْمَٰنِ الرَّحِيمِ", fake_ayah_map, 1, 3)
    print(r)

    print("\nTest 2: wrong vowel on first word (expect harakaat_error)")
    r = detect_harakaat_errors("الرَّحْمَٰنُ الرَّحِيمِ", fake_ayah_map, 1, 3)
    print(r)

    print("\nTest 3: wrong consonant (expect makhraj_error)")
    r = detect_harakaat_errors("الرَّجْمَٰنِ الرَّحِيمِ", fake_ayah_map, 1, 3)
    print(r)

    print("\nTest 4: Basmala (expect skipped, no_reference)")
    r = detect_harakaat_errors("بِسْمِ اللَّهِ", fake_ayah_map, 1, 1)
    print(r)
