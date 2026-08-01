"""
Live Display Formatter — Real-time Color-Coded Quran Recitation Display
------------------------------------------------------------------------
Generates HTML with word-level color coding for live merged transcript.
Colors:
  ✅ Green  (#10b981) — correct word (exact match after normalization)
  ⚠️  Amber  (#f59e0b) — minor error (>70% character similarity)
  ❌ Red    (#ef4444) — major error (<70% similarity)
  ⬜ Gray   (#9ca3af) — low confidence (<0.4)
"""

import json
import os
import re
import difflib
try:
    from rapidfuzz import fuzz as _rfuzz_ldf
    _LDF_HAS_RAPIDFUZZ = True
except ImportError:
    _rfuzz_ldf = None
    _LDF_HAS_RAPIDFUZZ = False
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

COLORS = {
    "correct": "#10b981",   # green
    "minor": "#f59e0b",     # amber / orange
    "major": "#ef4444",     # red — also used as "wrong" in the simplified display
    "uncertain": "#9ca3af", # gray
    "pending": "#94a3b8",   # gray — not yet reached / capped by an earlier mistake
}

_TAWWUZ_TEXT = "أعوذ بالله من الشيطان الرجيم"
_BASMALA_TEXT = "بسم الله الرحمن الرحيم"

# Lower threshold so exact matches can still show green in low-confidence chunks.
CONFIDENCE_UNCERTAIN_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Arabic helpers — reuse quran_guard normalizer
# ---------------------------------------------------------------------------

from fyp_model.quran_guard import normalize_arabic as _normalize


_TAWWUZ_TOKENS = _normalize(_TAWWUZ_TEXT).split()
_BASMALA_TOKENS = _normalize(_BASMALA_TEXT).split()


def _safe_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio between two strings."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _LDF_HAS_RAPIDFUZZ:
        return _rfuzz_ldf.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _word_color(
    recited_word: str,
    reference_word: str,
    confidence: Optional[float] = None,
) -> str:
    """Determine the colour for a single word based on similarity to reference."""
    norm_rec = _normalize(recited_word)
    norm_ref = _normalize(reference_word)

    if norm_rec == norm_ref:
        return COLORS["correct"]

    # Low confidence → gray, but only after checking for exact match
    if confidence is not None and confidence < CONFIDENCE_UNCERTAIN_THRESHOLD:
        return COLORS["uncertain"]

    sim = _char_similarity(norm_rec, norm_ref)
    if sim >= 0.70:
        return COLORS["minor"]
    return COLORS["major"]


def _classify_word(
    recited_word: str,
    reference_word: str,
    confidence: Optional[float] = None,
) -> str:
    """Return classification string for a word."""
    norm_rec = _normalize(recited_word)
    norm_ref = _normalize(reference_word)
    if norm_rec == norm_ref:
        return "correct"
    if confidence is not None and confidence < CONFIDENCE_UNCERTAIN_THRESHOLD:
        return "uncertain"
    sim = _char_similarity(norm_rec, norm_ref)
    if sim >= 0.70:
        return "minor"
    return "major"


# ---------------------------------------------------------------------------
# Main formatter class
# ---------------------------------------------------------------------------

class LiveDisplayFormatter:
    """Formats chunk results into HTML with colour-coded words and error stats."""

    def __init__(self, ayah_json_path: Optional[str] = None):
        self._ayah_map: Dict[str, str] = {}
        self._ayah_raw_map: Dict[str, str] = {}
        self._surah_cache: Dict[int, Dict[str, object]] = {}
        if ayah_json_path and os.path.isfile(ayah_json_path):
            self._load_reference(ayah_json_path)

        # Running statistics
        self._stats = {
            "total_words": 0,
            "correct": 0,
            "minor": 0,
            "major": 0,
            "uncertain": 0,
        }
        # Persistent per-word status for the ayah currently being recited,
        # keyed by (surah, ayah). "correct" is sticky (never reverts once
        # set); "wrong" can later flip to "correct" when the reciter fixes
        # it. Shared by both the expected-ayah panel and the surah-progress
        # panel so they always agree on what's actually been gotten right.
        self._word_status: Dict[Tuple[int, int], List[str]] = {}

    # ---- reference loading ----

    def _load_reference(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Unwrap nested dicts
        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break
        if isinstance(data, dict):
            for k, v in data.items():
                text = v.get("text", "") if isinstance(v, dict) else str(v or "")
                self._ayah_raw_map[k] = _safe_text(text)
                self._ayah_map[k] = _normalize(text)
        self._surah_cache.clear()

    def _get_surah_word_cache(self, surah: int) -> Optional[Dict[str, object]]:
        if surah in self._surah_cache:
            return self._surah_cache[surah]

        ayah_entries = []
        prefix = f"{surah}_"
        for key, raw_text in self._ayah_raw_map.items():
            if not key.startswith(prefix):
                continue
            try:
                _s, ayah_num = key.split("_", 1)
                ayah_entries.append((int(ayah_num), raw_text))
            except Exception:
                continue

        if not ayah_entries:
            return None

        ayah_entries.sort(key=lambda item: item[0])
        all_raw_words: List[str] = []
        all_norm_words: List[str] = []
        ayah_ranges: Dict[int, Tuple[int, int]] = {}

        for ayah_num, raw_text in ayah_entries:
            start_idx = len(all_raw_words)
            words = _safe_text(raw_text).split()
            all_raw_words.extend(words)
            all_norm_words.extend([_normalize(w) for w in words])
            end_idx = max(start_idx, len(all_raw_words) - 1)
            ayah_ranges[ayah_num] = (start_idx, end_idx)

        cache_entry = {
            "raw_words": all_raw_words,
            "norm_words": all_norm_words,
            "ayah_ranges": ayah_ranges,
        }
        self._surah_cache[surah] = cache_entry
        return cache_entry

    # ---- shared word-status engine (green/red, sticky, qari-aligned) ----

    def _get_word_status(self, surah: int, ayah: int, ref_words: List[str]) -> List[str]:
        """Return the persistent per-word status list for (surah, ayah),
        creating a fresh all-"pending" list the first time it's needed."""
        key = (int(surah), int(ayah))
        existing = self._word_status.get(key)
        if existing is None or len(existing) != len(ref_words):
            existing = ["pending"] * len(ref_words)
            self._word_status[key] = existing
        return existing

    def _update_word_status(
        self,
        surah: int,
        ayah: int,
        ref_words: List[str],
        recited_words: List[str],
    ) -> List[str]:
        """Diff this chunk's recited words against the full ayah reference
        and merge the result into the persistent status for (surah, ayah).

        - A word that matches becomes "correct". This is also how a
          previously "wrong" word gets fixed — it's the same check either way.
        - A word that mismatches becomes "wrong", but never overwrites an
          already-"correct" word: green is sticky.
        - Words this chunk doesn't touch keep whatever status they already had.
        """
        status = self._get_word_status(surah, ayah, ref_words)
        if not recited_words:
            return status

        ref_norm = [_normalize(w) for w in ref_words]
        rec_norm = [_normalize(w) for w in recited_words]

        # Word-similarity tolerance — matches the fuzzy threshold used
        # everywhere else in this file (_word_color/_classify_word) and in
        # correction_engine._words_close. Without this, a word that the
        # chunk-level guard scored as "ok" (CER/coverage tolerant) but that
        # differs from the reference by a single character (e.g. a hamza
        # form or a harmless ASR near-miss) got marked "wrong" here via
        # strict byte-exact SequenceMatcher comparison, and "wrong" is
        # sticky — so a correctly recited word could stay red forever.
        _WORD_MATCH_THRESHOLD = 0.72

        sm = difflib.SequenceMatcher(None, rec_norm, ref_norm)
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for j in range(j1, j2):
                    status[j] = "correct"
            elif tag == "replace":
                rec_slice = rec_norm[_i1:_i2]
                ref_slice = ref_norm[j1:j2]
                shared = min(len(rec_slice), len(ref_slice))
                for offset in range(shared):
                    j = j1 + offset
                    if status[j] == "correct":
                        continue
                    sim = _char_similarity(rec_slice[offset], ref_slice[offset])
                    status[j] = "correct" if sim >= _WORD_MATCH_THRESHOLD else "wrong"
                # Any remaining ref words in this span with no recited
                # counterpart (ref longer than rec) haven't been reached —
                # leave them as-is rather than falsely marking "wrong".
            # "insert" (ref words this chunk never reached) and "delete"
            # (extra recited words with no ref counterpart) leave status as-is.
        return status

    @staticmethod
    def _render_qari_aligned(ref_words: List[str], status: List[str]) -> List[str]:
        """Render word spans with qari-style sequential gating: once a word
        is wrong, no later word may show green until that specific word is
        fixed (its status flips back to "correct"). Later words show red
        (if actually wrong) or gray (if correct but not yet credited)."""
        frontier = next((i for i, s in enumerate(status) if s == "wrong"), None)
        spans: List[str] = []
        for idx, word in enumerate(ref_words):
            s = status[idx] if idx < len(status) else "pending"
            if frontier is not None and idx >= frontier and s == "correct":
                color = COLORS["pending"]
            elif s == "correct":
                color = COLORS["correct"]
            elif s == "wrong":
                color = COLORS["major"]
            else:
                color = COLORS["pending"]
            spans.append(
                f'<span style="color:{color}; font-weight:600; margin:0 2px; padding:2px 4px; '
                f'border-radius:4px; background:rgba({LiveDisplayFormatter._hex_to_rgb(color)},0.12);">{word}</span>'
            )
        return spans

    def format_surah_progress_html(
        self,
        surah: Optional[int],
        start_ayah: int,
        current_ayah: int,
        recited_text: str,
    ) -> str:
        if surah is None:
            return self._placeholder_html("Select a surah to view progress.")

        cache_entry = self._get_surah_word_cache(int(surah))
        if not cache_entry:
            return self._placeholder_html("Surah text not available.")

        raw_words: List[str] = cache_entry["raw_words"]  # type: ignore[assignment]
        ayah_ranges: Dict[int, Tuple[int, int]] = cache_entry["ayah_ranges"]  # type: ignore[assignment]

        all_status = ["pending"] * len(raw_words)

        for ayah_num, (start_idx, end_idx) in ayah_ranges.items():
            if ayah_num < start_ayah:
                continue
            if ayah_num < current_ayah:
                for idx in range(start_idx, end_idx + 1):
                    all_status[idx] = "correct"
                continue
            if ayah_num > current_ayah:
                continue

            ref_words = raw_words[start_idx:end_idx + 1]
            rec_words = _safe_text(recited_text).split() if recited_text else []
            status = self._update_word_status(surah, ayah_num, ref_words, rec_words)
            for offset, idx in enumerate(range(start_idx, end_idx + 1)):
                all_status[idx] = status[offset]

        spans = self._render_qari_aligned(raw_words, all_status)
        body = " ".join(spans)
        return (
            f'<div dir="rtl" style="'
            f"font-family: 'Amiri', 'Traditional Arabic', 'Arial', serif; "
            f'font-size: 22px; line-height: 2.2; text-align: right; '
            f'padding: 18px; border-radius: 12px; '
            f'background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); '
            f'border: 1px solid #334155; min-height: 120px;">'
            f'{body}</div>'
        )

    def _get_reference_text(
        self,
        surah: Optional[int],
        ayah: Optional[int],
        matched_ayah_text: Optional[str] = None,
    ) -> Optional[str]:
        """Get reference text for an ayah, preferring the already-matched text."""
        if matched_ayah_text:
            return _normalize(matched_ayah_text)
        if surah is not None and ayah is not None:
            key = f"{surah}_{ayah}"
            return self._ayah_map.get(key)
        return None

    def get_raw_ayah_text(self, surah: Optional[int], ayah: Optional[int]) -> Optional[str]:
        if surah is None or ayah is None:
            return None
        key = f"{surah}_{ayah}"
        return self._ayah_raw_map.get(key)

    # ---- HTML generation ----

    def format_chunk_html(self, chunk_result: Dict[str, Any]) -> str:
        """Format a single chunk result into HTML with colour-coded words.

        Uses sequence alignment (SequenceMatcher) so that inserted or
        deleted words don't throw off the colouring of all subsequent words.

        Parameters
        ----------
        chunk_result : dict
            Must contain at minimum ``corrected_text``.
            Optionally: ``matched_ayah_text``, ``confidence``, ``matched_ayah``,
            ``surah``, ``raw_asr``.

        Returns
        -------
        str
            HTML fragment with ``<span>`` tags for each word.
        """
        corrected = chunk_result.get("corrected_text", "")
        if not corrected or not corrected.strip():
            return ""

        ref_text = self._get_reference_text(
            chunk_result.get("surah"),
            chunk_result.get("matched_ayah"),
            chunk_result.get("matched_ayah_text"),
        )
        confidence = chunk_result.get("confidence")

        recited_words = corrected.split()
        ref_words = ref_text.split() if ref_text else []

        # If no reference, mark everything uncertain
        if not ref_words:
            spans: List[str] = []
            for word in recited_words:
                color = COLORS["uncertain"]
                self._stats["total_words"] += 1
                self._stats["uncertain"] += 1
                spans.append(self._word_span(word, color))
            return " ".join(spans)

        # --- Sequence-aligned word-level comparison ---
        norm_rec = [_normalize(w) for w in recited_words]
        norm_ref = [_normalize(w) for w in ref_words]

        # Build a per-recited-word classification using SequenceMatcher opcodes
        word_classes: List[str] = ["major"] * len(recited_words)

        sm = difflib.SequenceMatcher(None, norm_rec, norm_ref)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # All words in this range match exactly
                for k in range(i1, i2):
                    word_classes[k] = "correct"
            elif tag == "replace":
                # Words differ — check character similarity for each pair
                rec_slice = list(range(i1, i2))
                ref_slice = list(range(j1, j2))
                for idx, k in enumerate(rec_slice):
                    if idx < len(ref_slice):
                        sim = _char_similarity(norm_rec[k], norm_ref[ref_slice[idx]])
                        word_classes[k] = "correct" if sim >= 0.85 else ("minor" if sim >= 0.60 else "major")
                    else:
                        word_classes[k] = "major"  # extra recited word
            elif tag == "insert":
                # Words in recited_words[i1:i2] have no counterpart in ref
                for k in range(i1, i2):
                    word_classes[k] = "major"
            # tag == "delete" means ref words missing from recitation — no recited word to color

        # If low confidence, keep exact matches green and gray out the rest.
        if confidence is not None and confidence < CONFIDENCE_UNCERTAIN_THRESHOLD:
            word_classes = ["correct" if c == "correct" else "uncertain" for c in word_classes]

        # Build HTML spans
        spans = []
        for i, word in enumerate(recited_words):
            cls = word_classes[i]
            color = COLORS[cls]
            self._stats["total_words"] += 1
            self._stats[cls] += 1
            spans.append(self._word_span(word, color))

        return " ".join(spans)

    def _word_span(self, word: str, color: str) -> str:
        """Build a single colored word span."""
        return (
            f'<span style="color:{color}; font-weight:600; '
            f'margin:0 2px; padding:2px 4px; border-radius:4px; '
            f'background:rgba({self._hex_to_rgb(color)},0.12);">'
            f'{word}</span>'
        )

    def _build_invocation_spans(self, include_basmala: bool) -> List[str]:
        """Return no invocation text — users requested it be removed from display."""
        return []

    def _strip_invocations(self, text: str, include_basmala: bool) -> str:
        tokens = _safe_text(text).split()
        norm_tokens = [_normalize(t) for t in tokens]

        if norm_tokens[: len(_TAWWUZ_TOKENS)] == _TAWWUZ_TOKENS:
            tokens = tokens[len(_TAWWUZ_TOKENS):]
            norm_tokens = norm_tokens[len(_TAWWUZ_TOKENS):]

        if include_basmala and norm_tokens[: len(_BASMALA_TOKENS)] == _BASMALA_TOKENS:
            tokens = tokens[len(_BASMALA_TOKENS):]

        return " ".join(tokens).strip()

    def format_expected_ayah_html(
        self,
        surah: Optional[int],
        ayah: Optional[int],
        recited_text: str,
    ) -> str:
        """Render the current ayah, word by word, green/red against the ASR
        text. Always compares against the ACTUAL expected ayah (surah, ayah)
        — never against whatever ayah the fuzzy matcher guessed — so a bad
        or low-confidence guess can't make the display show a false match."""
        if surah is None or ayah is None:
            return self._placeholder_html("Select a surah to begin.")

        raw_ref_text = self.get_raw_ayah_text(surah, ayah)
        ref_words = _safe_text(raw_ref_text).split() if raw_ref_text else []
        if not ref_words:
            return self._placeholder_html("Ayah text not available.")

        include_basmala = _normalize(raw_ref_text).split()[: len(_BASMALA_TOKENS)] != _BASMALA_TOKENS
        recited_clean = self._strip_invocations(recited_text, include_basmala=include_basmala)
        rec_words = _safe_text(recited_clean).split()

        status = self._update_word_status(surah, ayah, ref_words, rec_words)
        spans = self._render_qari_aligned(ref_words, status)

        body = " ".join(spans)
        return (
            f'<div dir="rtl" style="'
            f"font-family: 'Amiri', 'Traditional Arabic', 'Arial', serif; "
            f'font-size: 26px; line-height: 2.2; text-align: right; '
            f'padding: 20px; border-radius: 12px; '
            f'background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); '
            f'border: 1px solid #334155; min-height: 100px;">'
            f'{body}</div>'
        )

    def merge_chunks_html(self, all_chunk_results: List[Dict[str, Any]]) -> str:
        """Merge all chunk results into a single continuous HTML display.

        Handles chunk-boundary overlap by fuzzy-matching the last few words
        of each chunk with the first few of the next.
        """
        # Reset stats for fresh calculation
        self._stats = {
            "total_words": 0,
            "correct": 0,
            "minor": 0,
            "major": 0,
            "uncertain": 0,
        }

        all_spans: List[str] = []
        prev_words: List[str] = []

        for chunk in all_chunk_results:
            corrected = chunk.get("corrected_text", "")
            if not corrected or not corrected.strip():
                continue

            current_words = corrected.split()

            # Deduplicate overlap
            if prev_words:
                overlap = self._find_overlap(prev_words, current_words)
                if overlap > 0:
                    current_words = current_words[overlap:]

            if not current_words:
                prev_words = corrected.split()
                continue

            # Build a sub-chunk dict with only the non-overlapping words
            sub_chunk = dict(chunk)
            sub_chunk["corrected_text"] = " ".join(current_words)

            # Adjust reference words too if available
            ref_text = chunk.get("matched_ayah_text")
            if ref_text:
                ref_words = ref_text.split()
                # Estimate offset
                orig_len = len(corrected.split())
                offset = orig_len - len(current_words)
                if offset > 0 and offset < len(ref_words):
                    sub_chunk["matched_ayah_text"] = " ".join(ref_words[offset:])

            html = self.format_chunk_html(sub_chunk)
            if html:
                all_spans.append(html)

            prev_words = corrected.split()

        if not all_spans:
            return self._placeholder_html("Start reciting...")

        body = " ".join(all_spans)
        return (
            f'<div dir="rtl" style="'
            f"font-family: 'Amiri', 'Traditional Arabic', 'Arial', serif; "
            f'font-size: 26px; line-height: 2.2; text-align: right; '
            f'padding: 20px; border-radius: 12px; '
            f'background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); '
            f'border: 1px solid #334155; min-height: 100px;">'
            f'{body}</div>'
        )

    def generate_error_panel(self, all_chunk_results: List[Dict[str, Any]]) -> str:
        """Generate an error summary HTML panel from running statistics."""
        total = self._stats["total_words"]
        if total == 0:
            return self._error_panel_html(0, 0, 0, 0, 0, 0.0)

        correct = self._stats["correct"]
        minor = self._stats["minor"]
        major = self._stats["major"]
        uncertain = self._stats["uncertain"]
        accuracy = (correct / total * 100) if total > 0 else 0.0

        return self._error_panel_html(total, correct, minor, major, uncertain, accuracy)

    # ---- helpers ----

    @staticmethod
    def _find_overlap(
        prev_words: List[str],
        curr_words: List[str],
        max_check: int = 8,
    ) -> int:
        """Find the number of overlapping words at the boundary."""
        if not prev_words or not curr_words:
            return 0

        max_len = min(len(prev_words), len(curr_words), max_check)
        best_len = 0
        best_ratio = 0.0

        for length in range(1, max_len + 1):
            prev_tail = " ".join(_normalize(w) for w in prev_words[-length:])
            curr_head = " ".join(_normalize(w) for w in curr_words[:length])
            if _LDF_HAS_RAPIDFUZZ:
                ratio = _rfuzz_ldf.ratio(prev_tail, curr_head) / 100.0
            else:
                ratio = difflib.SequenceMatcher(None, prev_tail, curr_head).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_len = length

        return best_len if best_ratio >= 0.80 else 0

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """Convert '#rrggbb' to 'r,g,b'."""
        h = hex_color.lstrip("#")
        return f"{int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

    @staticmethod
    def _placeholder_html(text: str) -> str:
        return (
            f'<div dir="rtl" style="'
            f"font-family: 'Amiri', 'Traditional Arabic', 'Arial', serif; "
            f'font-size: 22px; line-height: 2; text-align: center; '
            f'padding: 24px; color: #64748b; border-radius: 12px; '
            f'background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); '
            f'border: 1px solid #334155; min-height: 100px;">'
            f'<em>{text}</em></div>'
        )

    @staticmethod
    def _error_panel_html(
        total: int,
        correct: int,
        minor: int,
        major: int,
        uncertain: int,
        accuracy: float,
    ) -> str:
        """Render a premium statistics panel."""
        # Accuracy bar colour
        if accuracy >= 85:
            bar_color = COLORS["correct"]
        elif accuracy >= 60:
            bar_color = COLORS["minor"]
        else:
            bar_color = COLORS["major"]

        return f"""
<div style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    padding: 16px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    font-family: 'Inter', 'Segoe UI', sans-serif;
">
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);">
        <div style="font-size:28px; font-weight:700; color:#10b981;">{correct}</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Correct</div>
    </div>
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);">
        <div style="font-size:28px; font-weight:700; color:#f59e0b;">{minor}</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Minor</div>
    </div>
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);">
        <div style="font-size:28px; font-weight:700; color:#ef4444;">{major}</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Major</div>
    </div>
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba(156,163,175,0.1); border:1px solid rgba(156,163,175,0.3);">
        <div style="font-size:28px; font-weight:700; color:#9ca3af;">{uncertain}</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Uncertain</div>
    </div>
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); grid-column: span 2;">
        <div style="font-size:28px; font-weight:700; color:#3b82f6;">{total}</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Total Words</div>
    </div>
    <div style="text-align:center; padding:12px; border-radius:8px; background:rgba({LiveDisplayFormatter._hex_to_rgb(bar_color)},0.1); border:1px solid rgba({LiveDisplayFormatter._hex_to_rgb(bar_color)},0.3); grid-column: span 2;">
        <div style="font-size:32px; font-weight:800; color:{bar_color};">{accuracy:.1f}%</div>
        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Accuracy</div>
        <div style="margin-top:6px; height:6px; border-radius:3px; background:#1e293b; overflow:hidden;">
            <div style="width:{min(accuracy, 100):.1f}%; height:100%; border-radius:3px; background:{bar_color}; transition: width 0.5s ease;"></div>
        </div>
    </div>
</div>
"""
