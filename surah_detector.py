import json
import os
import re
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None

from fyp_model.quran_guard import normalize_arabic


# CRITICAL FIX: Only Ta'awwuz should be stripped, NOT Basmala
# Basmala is the first ayah of 113 surahs and critical for identification
_LEADING_INVOCATIONS = (
    "اعوذ بالله من الشيطان الرجيم",  # Ta'awwuz - OK to strip
    # "بسم الله الرحمن الرحيم",      # Basmala - REMOVED, it's part of ayahs
)


@dataclass
class SurahCandidate:
    surah: int
    score: float
    details: Dict[str, float]


class SurahDetector:
    """Lightweight surah detector based on n-gram overlap of normalized text."""

    def __init__(
        self,
        ayah_json_path: str,
        max_tokens_per_surah: Optional[int] = None,
        strip_invocations: bool = True,
    ) -> None:
        self._surah_index: Dict[int, Dict[str, object]] = {}
        self._strip_invocations = strip_invocations
        self._max_tokens_per_surah = max_tokens_per_surah

        if ayah_json_path and os.path.isfile(ayah_json_path):
            self._load_reference(ayah_json_path)

    def _load_reference(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break

        if not isinstance(data, dict):
            return

        surah_tokens: Dict[int, List[str]] = {}
        for key, value in data.items():
            if not (isinstance(key, str) and "_" in key):
                continue
            try:
                surah_str, _ = key.split("_", 1)
                surah = int(surah_str)
            except Exception:
                continue

            text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
            text = normalize_arabic(text)
            if not text:
                continue

            tokens = text.split()
            if not tokens:
                continue

            surah_tokens.setdefault(surah, []).extend(tokens)

        for surah, tokens in surah_tokens.items():
            if self._max_tokens_per_surah:
                tokens = tokens[: self._max_tokens_per_surah]

            self._surah_index[surah] = {
                "tokens": tokens,
                "unigrams": set(_ngrams(tokens, 1)),
                "bigrams": set(_ngrams(tokens, 2)),
                "trigrams": set(_ngrams(tokens, 3)),
                "fourgrams": set(_ngrams(tokens, 4)),  # 4-grams for better precision
            }

        print(f"[SurahDetector] Loaded {len(self._surah_index)} surahs into index")

    def detect(self, text: str, top_k: int = 5) -> List[SurahCandidate]:
        if not text:
            return []

        # Only strip Ta'awwuz, keep Basmala (it's part of surah identification)
        if self._strip_invocations:
            text = _strip_leading_invocations(text)

        tokens = normalize_arabic(text).split()
        if not tokens:
            return []

        input_unigrams = set(_ngrams(tokens, 1))
        input_bigrams = set(_ngrams(tokens, 2))
        input_trigrams = set(_ngrams(tokens, 3))
        input_fourgrams = set(_ngrams(tokens, 4))

        candidates: List[SurahCandidate] = []
        for surah, info in self._surah_index.items():
            score, details = _score_ngrams(
                input_unigrams,
                input_bigrams,
                input_trigrams,
                input_fourgrams,
                info,
                len(tokens),
            )
            if score > 0:
                candidates.append(SurahCandidate(surah=surah, score=score, details=details))

        candidates.sort(key=lambda c: c.score, reverse=True)

        # DEBUG logging (use ascii-safe output to avoid Windows cp1252 crash)
        if candidates:
            try:
                print(f"[SurahDetector] Top 3 candidates for input ({len(tokens)} tokens):")
                for c in candidates[:3]:
                    print(f"  Surah {c.surah}: {c.score:.3f} - {c.details}")
            except (UnicodeEncodeError, OSError):
                pass
        else:
            try:
                print(f"[SurahDetector] No candidates found ({len(tokens)} tokens)")
            except (UnicodeEncodeError, OSError):
                pass

        return candidates[: max(1, top_k)]


class SurahLockManager:
    """State machine that locks onto a surah after consistent detection."""

    def __init__(
        self,
        min_score: float = 0.20,
        avg_score_threshold: float = 0.28,
        margin_threshold: float = 0.08,
        history_size: int = 4,
        lock_votes: int = 2,
        unlock_score: float = 0.15,
        unlock_votes: int = 3,
    ) -> None:
        self.min_score = min_score
        self.avg_score_threshold = avg_score_threshold
        self.margin_threshold = margin_threshold
        self.history_size = history_size
        self.lock_votes = lock_votes
        self.unlock_score = unlock_score
        self.unlock_votes = unlock_votes

        self.lock_state = "INACTIVE"
        self.locked_surah: Optional[int] = None
        self._history: List[Dict[str, float]] = []
        self._mismatch_count = 0

    def update(self, candidates: List[SurahCandidate]) -> Dict[str, object]:
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None

        margin = 0.0
        if top is not None:
            margin = top.score - (second.score if second else 0.0)

        if top is not None and top.score >= self.min_score:
            self._history.append({
                "surah": int(top.surah),
                "score": float(top.score),
                "margin": float(margin),
            })
            if len(self._history) > self.history_size:
                self._history.pop(0)

        if self.lock_state == "ACTIVE":
            if top is None or top.score < self.unlock_score:
                self._mismatch_count += 1
            elif top.surah != self.locked_surah:
                self._mismatch_count += 1
            else:
                self._mismatch_count = 0

            if self._mismatch_count >= self.unlock_votes:
                self.lock_state = "INACTIVE"
                self.locked_surah = None
                self._mismatch_count = 0
                print(f"[SurahLockManager] 🔓 UNLOCKED (mismatches: {self._mismatch_count})")

        if self.lock_state == "INACTIVE" and len(self._history) >= self.lock_votes:
            stats = _aggregate_history(self._history)
            for surah, summary in stats.items():
                if summary["count"] < self.lock_votes:
                    continue

                avg_score = summary["sum_score"] / summary["count"]
                avg_margin = summary["sum_margin"] / summary["count"]
                if avg_score >= self.avg_score_threshold and avg_margin >= self.margin_threshold:
                    self.lock_state = "ACTIVE"
                    self.locked_surah = surah
                    self._mismatch_count = 0
                    print(f"[SurahLockManager] 🔒 LOCKED on Surah {surah} (avg_score={avg_score:.3f}, margin={avg_margin:.3f})")
                    break

        return {
            "lock_state": self.lock_state,
            "locked_surah": self.locked_surah,
            "top_surah": top.surah if top else None,
            "top_score": float(top.score) if top else 0.0,
            "margin": float(margin),
            "candidates": [
                {"surah": c.surah, "score": float(c.score)} for c in candidates
            ],
        }


def _aggregate_history(history: List[Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    for item in history:
        surah = int(item["surah"])
        summary.setdefault(surah, {"count": 0, "sum_score": 0.0, "sum_margin": 0.0})
        summary[surah]["count"] += 1
        summary[surah]["sum_score"] += float(item["score"])
        summary[surah]["sum_margin"] += float(item["margin"])
    return summary


def _strip_leading_invocations(
    text: str,
    max_prefix_words: int = 20,  # Only check first ~20 words
    min_ratio: int = 75,         # Require 75% similarity to strip
) -> str:
    """
    Strip ONLY Ta'awwuz ('أعوذ بالله من الشيطان الرجيم') if present.
    Does NOT strip Basmala - it's part of 113 surahs and critical for identification.
    """
    if not text:
        return text

    original_tokens = re.split(r"\s+", text.strip())
    norm_tokens = normalize_arabic(text).split()
    if not norm_tokens or not original_tokens:
        return text

    for phrase in _LEADING_INVOCATIONS:  # Now only contains Ta'awwuz
        phrase_norm = normalize_arabic(phrase)
        if not phrase_norm:
            continue

        prefix_tokens = norm_tokens[:max_prefix_words]
        if len(prefix_tokens) < 2:
            break

        phrase_len = len(phrase_norm.split())
        min_tokens = max(2, int(phrase_len * 0.7))  # At least 70% of phrase
        max_tokens = min(len(prefix_tokens), phrase_len + 2)

        best_len = 0
        best_score = 0.0
        for length in range(min_tokens, max_tokens + 1):
            candidate = " ".join(prefix_tokens[:length])
            score = _phrase_similarity(candidate, phrase_norm)
            if score > best_score:
                best_score = score
                best_len = length

        if best_score >= min_ratio and best_len > 0:
            norm_tokens = norm_tokens[best_len:]
            original_tokens = original_tokens[best_len:]
            print(f"[SurahDetector] Stripped Ta'awwuz (similarity: {best_score:.1f}%)")

    return " ".join(original_tokens).strip()


def _phrase_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return float(fuzz.ratio(a, b))  # Full ratio, not partial_ratio
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return tokens
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _score_ngrams(
    input_unigrams: set,
    input_bigrams: set,
    input_trigrams: set,
    input_fourgrams: set,
    info: Dict[str, object],
    input_length: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Score n-gram overlap with ADAPTIVE WEIGHTING based on input length.
    Short inputs (< 10 words) use more weight on bigrams/unigrams.
    Long inputs use more weight on trigrams/fourgrams for precision.
    """
    uni = _overlap(input_unigrams, info.get("unigrams", set()))
    bi = _overlap(input_bigrams, info.get("bigrams", set()))
    tri = _overlap(input_trigrams, info.get("trigrams", set()))
    four = _overlap(input_fourgrams, info.get("fourgrams", set()))

    # ADAPTIVE WEIGHTS based on input length
    if input_length < 10:
        # Short input: emphasize bigrams and unigrams
        weights = [(0.25, uni), (0.45, bi), (0.25, tri), (0.05, four)]
    elif input_length < 20:
        # Medium input: balanced
        weights = [(0.15, uni), (0.30, bi), (0.35, tri), (0.20, four)]
    else:
        # Long input: emphasize longer n-grams for precision
        weights = [(0.10, uni), (0.20, bi), (0.35, tri), (0.35, four)]

    score_sum = 0.0
    weight_sum = 0.0
    for weight, value in weights:
        if value > 0:
            score_sum += weight * value
            weight_sum += weight

    score = score_sum / weight_sum if weight_sum > 0 else 0.0
    return score, {"unigram": uni, "bigram": bi, "trigram": tri, "fourgram": four}


def _overlap(input_set: set, reference_set: set) -> float:
    """
    Measures the percentage of the input n-grams that exist in the reference.
    Unlike Jaccard (Intersection over Union), this does NOT penalize long surahs.
    """
    if not input_set or not reference_set:
        return 0.0
    inter = len(input_set.intersection(reference_set))
    return inter / len(input_set)
