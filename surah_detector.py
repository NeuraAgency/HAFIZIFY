"""
surah_detector.py — High-Precision Auto Surah Detection
========================================================
Two-stage information retrieval pipeline:

  Stage 1  Word-level BM25 retrieval at the ayah level (6,236 documents).
           Unique words receive exponentially higher IDF weights than common
           words like "الله" or "رب".  The raw BM25 score is normalised to
           [0, 1] by dividing by a theoretical perfect-match score.

  Stage 2  Character-level fuzzy substring reranker.
           fuzz.partial_ratio handles short recitation fragments, ASR typos,
           and split-word spacing.  Falls back gracefully to difflib when
           rapidfuzz is not installed.

  Aggregation
           Surah score = max(ayah scores) across all ayahs in that surah.
           Correct matches score near 1.0; wrong matches score near 0.1,
           giving the SurahLockManager a large, reliable margin to lock on.
"""

from __future__ import annotations

import json
import math
import os
import re
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:          # pragma: no cover
    _fuzz = None
    _HAS_RAPIDFUZZ = False

from fyp_model.quran_guard import normalize_arabic


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CRITICAL: Only Ta'awwuz is stripped — NOT Basmala.
# Basmala is Ayah 1 of Al-Fatiha and appears at the head of 113 surahs;
# removing it destroys critical identification signal.
_LEADING_INVOCATIONS: Tuple[str, ...] = (
    "اعوذ بالله من الشيطان الرجيم",   # Ta'awwuz — safe to strip
    # "بسم الله الرحمن الرحيم",        # Basmala  — DO NOT strip
)

# BM25 hyperparameters (Robertson & Zaragoza, 2009)
_BM25_K1: float = 1.5    # term-frequency saturation ceiling
_BM25_B:  float = 0.75   # length-normalisation factor

# Pipeline hyperparameters
_BM25_TOP_N:    int   = 30   # candidate ayahs forwarded to fuzzy reranker
_SCORE_W_BM25:  float = 0.4  # weight for normalised BM25 component
_SCORE_W_FUZZY: float = 0.6  # weight for fuzzy substring component


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AyahEntry:
    """One indexed ayah."""
    surah:  int
    ayah:   int
    text:   str                         # normalised Arabic text
    tokens: List[str]
    tf:     Dict[str, int] = field(default_factory=dict)   # term → raw count


@dataclass
class SurahCandidate:
    """Detected surah candidate returned by SurahDetector.detect()."""
    surah:   int
    score:   float
    details: Dict[str, float]


# ---------------------------------------------------------------------------
# SurahDetector
# ---------------------------------------------------------------------------

class SurahDetector:
    """
    High-precision surah detector using BM25 + fuzzy substring reranking.

    Parameters
    ----------
    ayah_json_path : str
        Path to the ayah JSON file.  Expected key format: ``"<surah>_<ayah>"``.
        Each value may be a dict with a ``"text"`` field, or a plain string.
    strip_invocations : bool
        When *True* (default), Ta'awwuz is stripped from the start of input
        before scoring.
    """

    def __init__(
        self,
        ayah_json_path: str,
        strip_invocations: bool = True,
    ) -> None:
        self._strip_invocations: bool = strip_invocations

        # Populated by _load_reference()
        self._ayahs: List[AyahEntry] = []

        # Populated by _build_bm25_stats()
        self._idf:    Dict[str, float] = {}
        self._avg_dl: float = 0.0
        self._N:      int   = 0

        if ayah_json_path and os.path.isfile(ayah_json_path):
            self._load_reference(ayah_json_path)
            self._build_bm25_stats()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _load_reference(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unwrap common envelope keys
        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break

        if not isinstance(data, dict):
            print("[SurahDetector] WARNING: unexpected JSON structure — index is empty.")
            return

        skipped = 0
        for key, value in data.items():
            if not (isinstance(key, str) and "_" in key):
                skipped += 1
                continue
            try:
                surah_str, ayah_str = key.split("_", 1)
                surah = int(surah_str)
                ayah  = int(ayah_str)
            except (ValueError, TypeError):
                skipped += 1
                continue

            raw_text = (
                str(value.get("text", "")) if isinstance(value, dict)
                else str(value or "")
            )
            norm_text = normalize_arabic(raw_text)
            if not norm_text:
                skipped += 1
                continue

            tokens = norm_text.split()
            if not tokens:
                skipped += 1
                continue

            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            self._ayahs.append(
                AyahEntry(surah=surah, ayah=ayah, text=norm_text, tokens=tokens, tf=tf)
            )

        print(
            f"[SurahDetector] Loaded {len(self._ayahs)} ayahs "
            f"({skipped} entries skipped)."
        )

    def _build_bm25_stats(self) -> None:
        """Compute per-term IDF and average document length."""
        self._N = len(self._ayahs)
        if self._N == 0:
            print("[SurahDetector] WARNING: ayah list is empty — BM25 stats not built.")
            return

        df: Dict[str, int] = {}
        total_len = 0
        for entry in self._ayahs:
            total_len += len(entry.tokens)
            for term in entry.tf:          # one count per ayah regardless of tf
                df[term] = df.get(term, 0) + 1

        self._avg_dl = total_len / self._N

        # Robertson-Sparck Jones IDF with +1 smoothing (always positive)
        for term, doc_freq in df.items():
            self._idf[term] = math.log(
                (self._N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )

        print(
            f"[SurahDetector] BM25 stats ready: "
            f"{self._N} ayahs | {len(self._idf)} unique terms | "
            f"avg_dl={self._avg_dl:.1f} tokens"
        )

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _bm25_score(self, query_tokens: List[str], entry: AyahEntry) -> float:
        """Raw BM25 score for *query_tokens* against one ayah *entry*."""
        dl = len(entry.tokens)
        score = 0.0
        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            tf = entry.tf.get(term, 0)
            if tf == 0:
                continue
            numerator   = tf * (_BM25_K1 + 1.0)
            denominator = tf + _BM25_K1 * (
                1.0 - _BM25_B + _BM25_B * dl / self._avg_dl
            )
            score += idf * (numerator / denominator)
        return score

    def _perfect_bm25_score(self, query_tokens: List[str]) -> float:
        """
        Theoretical upper-bound for normalisation.

        Scores the query against a hypothetical ayah where every query term
        appears exactly once and the document length equals *avg_dl*.  Caps
        the normalised final score at 1.0 for the rare case where an actual
        ayah's repeated terms push above this ceiling.

        When k1=1.5 and b=0.75, the formula simplifies to:

            perfect_score per term = IDF(term) * (k1 + 1) / (1 + k1)
                                   = IDF(term)

        so the perfect score is simply the sum of IDF values for unique query
        terms.
        """
        score = 0.0
        seen: set = set()
        for term in query_tokens:
            if term in seen:
                continue
            seen.add(term)
            idf = self._idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            # tf=1, dl=avg_dl  →  denominator = 1 + k1*(1 - b + b*1) = 1 + k1
            numerator   = 1.0 * (_BM25_K1 + 1.0)
            denominator = 1.0 + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * 1.0)
            score += idf * (numerator / denominator)
        return score

    @staticmethod
    def _fuzzy_score(query: str, candidate: str) -> float:
        """
        Character-level partial substring ratio → [0.0, 1.0].

        ``fuzz.partial_ratio`` scores the shorter string against the best
        aligned window of the longer string, so a short 3-word recitation
        inside a long 30-word ayah can still score 1.0.
        """
        if not query or not candidate:
            return 0.0
        if _HAS_RAPIDFUZZ:
            return _fuzz.partial_ratio(query, candidate) / 100.0
        # difflib fallback: SequenceMatcher.ratio() is a full-string metric,
        # not a substring metric, so recall will be lower for short inputs.
        return difflib.SequenceMatcher(None, query, candidate).ratio()

    # ------------------------------------------------------------------
    # Main detection pipeline
    # ------------------------------------------------------------------

    def detect(self, text: str, top_k: int = 5) -> List[SurahCandidate]:
        """
        Detect the most likely surah(s) for *text*.

        Parameters
        ----------
        text : str
            Raw ASR transcription of the recitation.
        top_k : int
            Maximum number of surah candidates to return.

        Returns
        -------
        List[SurahCandidate]
            Sorted descending by score.  Empty list if index is empty or
            *text* produces no tokens after normalisation.
        """
        if not text or not self._ayahs:
            return []

        # ── Pre-processing ────────────────────────────────────────────
        if self._strip_invocations:
            text = _strip_leading_invocations(text)

        # Work entirely on the normalised query from this point forward.
        norm_query    = normalize_arabic(text)
        query_tokens  = norm_query.split()
        if not query_tokens:
            return []

        # ── Stage 1: BM25 retrieval ───────────────────────────────────
        perfect = self._perfect_bm25_score(query_tokens)

        bm25_scores: List[Tuple[float, AyahEntry]] = [
            (self._bm25_score(query_tokens, entry), entry)
            for entry in self._ayahs
        ]
        bm25_scores.sort(key=lambda x: x[0], reverse=True)
        top_ayahs = bm25_scores[:_BM25_TOP_N]

        # ── Stage 2: Fuzzy substring reranking ───────────────────────
        combined: List[Tuple[float, AyahEntry, float, float]] = []
        for raw_bm25, entry in top_ayahs:
            norm_bm25 = min((raw_bm25 / perfect) if perfect > 0.0 else 0.0, 1.0)
            fuzzy     = self._fuzzy_score(norm_query, entry.text)
            final     = _SCORE_W_BM25 * norm_bm25 + _SCORE_W_FUZZY * fuzzy
            combined.append((final, entry, norm_bm25, fuzzy))

        combined.sort(key=lambda x: x[0], reverse=True)

        # ── Aggregation: surah score = max ayah score ─────────────────
        surah_best: Dict[int, Tuple[float, float, float]] = {}
        for final, entry, norm_bm25, fuzzy in combined:
            prev = surah_best.get(entry.surah)
            if prev is None or final > prev[0]:
                surah_best[entry.surah] = (final, norm_bm25, fuzzy)

        candidates: List[SurahCandidate] = [
            SurahCandidate(
                surah=surah,
                score=round(scores[0], 6),
                details={
                    "bm25":  round(scores[1], 4),
                    "fuzzy": round(scores[2], 4),
                },
            )
            for surah, scores in surah_best.items()
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        # ── Debug logging ─────────────────────────────────────────────
        try:
            print(
                f"[SurahDetector] Top 3 for input "
                f"({len(query_tokens)} tokens, rapidfuzz={'yes' if _HAS_RAPIDFUZZ else 'no'}):"
            )
            for c in candidates[:3]:
                print(f"  Surah {c.surah:3d}: {c.score:.4f}  {c.details}")
        except (UnicodeEncodeError, OSError):
            pass

        return candidates[: max(1, top_k)]


# ---------------------------------------------------------------------------
# SurahLockManager
# ---------------------------------------------------------------------------

class SurahLockManager:
    """
    State machine that locks onto a surah after consistent high-confidence
    detections and unlocks when evidence collapses.

    Parameters
    ----------
    min_score : float
        Minimum score for a candidate to be admitted to history.
    avg_score_threshold : float
        Minimum *average* score across the history window to trigger a lock.
    margin_threshold : float
        Minimum *average* score margin (top − second) to trigger a lock.
    history_size : int
        Rolling window length for vote aggregation.
    lock_votes : int
        Minimum number of frames agreeing on a surah before locking.
    unlock_score : float
        If the top candidate falls below this, count it as a mismatch.
    unlock_votes : int
        Consecutive mismatches required to release the lock.
    """

    def __init__(
        self,
        min_score:            float = 0.20,
        avg_score_threshold:  float = 0.28,
        margin_threshold:     float = 0.08,
        history_size:         int   = 4,
        lock_votes:           int   = 2,
        unlock_score:         float = 0.15,
        unlock_votes:         int   = 3,
    ) -> None:
        self.min_score            = min_score
        self.avg_score_threshold  = avg_score_threshold
        self.margin_threshold     = margin_threshold
        self.history_size         = history_size
        self.lock_votes           = lock_votes
        self.unlock_score         = unlock_score
        self.unlock_votes         = unlock_votes

        self.lock_state:    str           = "INACTIVE"
        self.locked_surah:  Optional[int] = None
        self._history:      List[Dict[str, float]] = []
        self._mismatch_count: int = 0

    def update(self, candidates: List[SurahCandidate]) -> Dict[str, object]:
        """
        Ingest the latest detection result and advance the lock state machine.

        Returns a status dict consumed by the Gradio UI.
        """
        top    = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None

        margin = 0.0
        if top is not None:
            margin = top.score - (second.score if second else 0.0)

        # Admit to history only when score is meaningful
        if top is not None and top.score >= self.min_score:
            self._history.append({
                "surah":  int(top.surah),
                "score":  float(top.score),
                "margin": float(margin),
            })
            if len(self._history) > self.history_size:
                self._history.pop(0)

        # ── Unlock logic ──────────────────────────────────────────────
        if self.lock_state == "ACTIVE":
            if top is None or top.score < self.unlock_score:
                self._mismatch_count += 1
            elif top.surah != self.locked_surah:
                self._mismatch_count += 1
            else:
                self._mismatch_count = 0

            if self._mismatch_count >= self.unlock_votes:
                # FIX: capture count BEFORE reset so the log is accurate
                mismatches_at_unlock = self._mismatch_count
                self.lock_state    = "INACTIVE"
                self.locked_surah  = None
                self._mismatch_count = 0
                self._history.clear()
                try:
                    print(
                        f"[SurahLockManager] UNLOCKED "
                        f"(mismatches: {mismatches_at_unlock})"
                    )
                except (UnicodeEncodeError, OSError):
                    pass

        # ── Lock logic ────────────────────────────────────────────────
        if self.lock_state == "INACTIVE" and len(self._history) >= self.lock_votes:
            stats = _aggregate_history(self._history)
            for surah, summary in stats.items():
                if summary["count"] < self.lock_votes:
                    continue
                avg_score  = summary["sum_score"]  / summary["count"]
                avg_margin = summary["sum_margin"] / summary["count"]

                if (
                    avg_score  >= self.avg_score_threshold
                    and avg_margin >= self.margin_threshold
                ):
                    self.lock_state   = "ACTIVE"
                    self.locked_surah = surah
                    self._mismatch_count = 0
                    try:
                        print(
                            f"[SurahLockManager] LOCKED on Surah {surah} "
                            f"(avg_score={avg_score:.3f}, avg_margin={avg_margin:.3f})"
                        )
                    except (UnicodeEncodeError, OSError):
                        pass
                    break

        return {
            "lock_state":   self.lock_state,
            "locked_surah": self.locked_surah,
            "top_surah":    top.surah if top else None,
            "top_score":    float(top.score) if top else 0.0,
            "margin":       float(margin),
            "candidates": [
                {"surah": c.surah, "score": float(c.score)} for c in candidates
            ],
        }

    def reset(self) -> None:
        """Hard-reset the lock manager to its initial state."""
        self.lock_state    = "INACTIVE"
        self.locked_surah  = None
        self._history.clear()
        self._mismatch_count = 0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aggregate_history(
    history: List[Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """Summarise rolling history into per-surah vote counts and score sums."""
    summary: Dict[int, Dict[str, float]] = {}
    for item in history:
        surah = int(item["surah"])
        summary.setdefault(surah, {"count": 0, "sum_score": 0.0, "sum_margin": 0.0})
        summary[surah]["count"]      += 1
        summary[surah]["sum_score"]  += float(item["score"])
        summary[surah]["sum_margin"] += float(item["margin"])
    return summary


def _strip_leading_invocations(
    text: str,
    max_prefix_words: int = 20,
    min_ratio:        int = 75,
) -> str:
    """
    Strip Ta'awwuz from the start of *text* if detected, then return the
    remainder as a normalised string.

    Implementation note
    -------------------
    The original code kept a parallel ``original_tokens`` list and sliced
    it by ``best_len`` from the normalised token count.  If ``normalize_arabic``
    changed the token count (e.g. by splitting or merging tokens), this
    produced a misaligned result.

    This version works *entirely* on the normalised token sequence and
    returns normalised text.  The caller (``SurahDetector.detect``) calls
    ``normalize_arabic`` again immediately afterward, so the behaviour is
    identical to the old approach for all correct inputs, without the
    alignment risk.
    """
    if not text:
        return text

    norm_tokens = normalize_arabic(text).split()
    if not norm_tokens:
        return text

    for phrase in _LEADING_INVOCATIONS:
        phrase_norm = normalize_arabic(phrase)
        if not phrase_norm:
            continue

        prefix_tokens = norm_tokens[:max_prefix_words]
        if len(prefix_tokens) < 2:
            break

        phrase_len = len(phrase_norm.split())
        min_tokens = max(2, int(phrase_len * 0.7))
        max_tokens = min(len(prefix_tokens), phrase_len + 2)

        best_len   = 0
        best_score = 0.0
        for length in range(min_tokens, max_tokens + 1):
            candidate = " ".join(prefix_tokens[:length])
            score     = _phrase_similarity(candidate, phrase_norm)
            if score > best_score:
                best_score = score
                best_len   = length

        if best_score >= min_ratio and best_len > 0:
            norm_tokens = norm_tokens[best_len:]
            try:
                print(
                    f"[SurahDetector] Stripped Ta'awwuz "
                    f"(similarity: {best_score:.1f}%)"
                )
            except (UnicodeEncodeError, OSError):
                pass

    return " ".join(norm_tokens).strip()


def _phrase_similarity(a: str, b: str) -> float:
    """Full-string similarity ratio as a percentage [0, 100]."""
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(_fuzz.ratio(a, b))
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0
