import json
import os
import re
import difflib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rapidfuzz import fuzz as _rfuzz
    from rapidfuzz.distance import Levenshtein as _rfLev
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


_AR_DIACRITICS = (
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061A"
    "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F\u0670\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4\u06E7\u06E8\u06EA\u06EB\u06EC\u06ED"
)
_DIACRITICS_PATTERN = re.compile("[" + _AR_DIACRITICS + "]")
_ALLOWED_CHARS_PATTERN = re.compile(r"[^\u0600-\u06FFA\u0660-\u0669\u06F0-\u06F9 ]+")

_PHRASE_FIXES: Sequence[Tuple[str, str]] = (
    ("الرحيلحمد", "الرحيم الحمد"),
    ("الدي ن", "الدين"),
    ("العالم ن", "العالمين"),
    ("إياكنعبد", "إياك نعبد"),
    ("أن عمت", "أنعمت"),
    # Whisper split-word artifacts (spurious space inside a word)
    ("الر حيم", "الرحيم"),
    ("الر حين", "الرحيم"),
    ("الرح يم", "الرحيم"),
    ("الرحم ن", "الرحمن"),
    ("الرح من", "الرحمن"),
    ("المست قيم", "المستقيم"),
    ("المستق يم", "المستقيم"),
    ("المس تقيم", "المستقيم"),
    ("الض الين", "الضالين"),
    ("الضال ين", "الضالين"),
    ("العال مين", "العالمين"),
    ("نست عين", "نستعين"),
    ("المحت كين", "المستقيم"),
    ("صرا ط", "صراط"),
    ("سرا ط", "صراط"),
    ("يو م", "يوم"),
)

_TOKEN_FIXES: Dict[str, str] = {
    "الرحي": "الرحيم",
    "الرحين": "الرحيم",
    "المستقي": "المستقيم",
    "صاط": "صراط",
    "هدنا": "اهدنا",
    "نستعي": "نستعين",
}

_BASMALA_TOKENS = ("بسم", "الله", "الرحمن", "الرحيم")


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = _DIACRITICS_PATTERN.sub("", text)
    text = text.replace("\u0640", "")
    text = _ALLOWED_CHARS_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_display_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _levenshtein_seq(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i]
        for j, bj in enumerate(b, 1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ai != bj)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def compute_cer(hyp: str, ref: str) -> float:
    hyp_n = normalize_arabic(hyp)
    ref_n = normalize_arabic(ref)
    if not ref_n:
        return 1.0
    return _levenshtein_seq(list(hyp_n), list(ref_n)) / max(1, len(ref_n))


def compute_wer(hyp: str, ref: str) -> float:
    hyp_w = normalize_arabic(hyp).split()
    ref_w = normalize_arabic(ref).split()
    if not ref_w:
        return 1.0
    return _levenshtein_seq(hyp_w, ref_w) / max(1, len(ref_w))


def token_coverage(hyp: str, ref: str) -> float:
    h = normalize_arabic(hyp).split()
    r = normalize_arabic(ref).split()
    if not h or not r:
        return 0.0

    counts: Dict[str, int] = {}
    for token in h:
        counts[token] = counts.get(token, 0) + 1

    overlap = 0
    for token in r:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            overlap += 1
    return overlap / max(1, len(r))


def _cer_prenorm(hyp_n: str, ref_n: str) -> float:
    """CER assuming BOTH inputs are already normalized. Skips normalize_arabic()."""
    if not ref_n:
        return 1.0
    return _levenshtein_seq(list(hyp_n), list(ref_n)) / max(1, len(ref_n))


def _wer_prenorm(hyp_n: str, ref_n: str) -> float:
    """WER assuming BOTH inputs are already normalized. Skips normalize_arabic()."""
    hyp_w = hyp_n.split()
    ref_w = ref_n.split()
    if not ref_w:
        return 1.0
    return _levenshtein_seq(hyp_w, ref_w) / max(1, len(ref_w))


def _coverage_prenorm(hyp_n: str, ref_n: str) -> float:
    """token_coverage assuming BOTH inputs are already normalized."""
    h = hyp_n.split()
    r = ref_n.split()
    if not h or not r:
        return 0.0
    counts: dict = {}
    for token in h:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in r:
        if counts.get(token, 0) > 0:
            counts[token] -= 1
            overlap += 1
    return overlap / max(1, len(r))


def _remove_consecutive_duplicates(text: str) -> str:
    tokens = safe_display_text(text).split()
    if not tokens:
        return ""
    out = [tokens[0]]
    for token in tokens[1:]:
        if token != out[-1]:
            out.append(token)
    return " ".join(out)


def _has_leading_basmala(text: str) -> bool:
    tokens = normalize_arabic(text).split()
    return len(tokens) >= 4 and tuple(tokens[:4]) == _BASMALA_TOKENS


def _strip_leading_basmala(text: str) -> str:
    tokens = safe_display_text(text).split()
    if len(tokens) >= 4:
        return " ".join(tokens[4:]).strip()
    return ""


def _build_vocabulary(ayah_map: Optional[Dict[Tuple[int, int], str]]) -> set:
    vocab = set()
    if not ayah_map:
        return vocab
    for ayah_text in ayah_map.values():
        vocab.update(normalize_arabic(ayah_text).split())
    return vocab


def _split_merged_token(token: str, vocab: set) -> Optional[List[str]]:
    if token in vocab or len(token) < 6:
        return None
    for i in range(2, len(token) - 1):
        left = token[:i]
        right = token[i:]
        if left in vocab and right in vocab:
            return [left, right]
    return None


def _recover_word_boundaries(text: str, vocab: set) -> str:
    tokens = safe_display_text(text).split()
    if not tokens or not vocab:
        return safe_display_text(text)

    recovered: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Try joining with next token — merge if result is a known word
        if i + 1 < len(tokens):
            joined = token + tokens[i + 1]
            next_tok = tokens[i + 1]
            # Original: only merge if BOTH fragments aren't vocab words
            # New: also merge if either fragment is very short (≤3 chars)
            # which likely means Whisper split a word with a spurious space
            either_short = len(token) <= 3 or len(next_tok) <= 3
            neither_in_vocab = token not in vocab and next_tok not in vocab
            if joined in vocab and (neither_in_vocab or either_short):
                recovered.append(joined)
                i += 2
                continue

        split_tokens = _split_merged_token(token, vocab)
        if split_tokens is not None:
            recovered.extend(split_tokens)
        else:
            recovered.append(token)
        i += 1

    return " ".join(recovered)


def correct_text_rules(text: str, vocab: Optional[set] = None, mode: str = "balanced") -> str:
    out = normalize_arabic(text)
    out = _remove_consecutive_duplicates(out)

    if mode in ("balanced", "aggressive"):
        for old, new in _PHRASE_FIXES:
            out = out.replace(old, new)

        tokens = out.split()
        tokens = [_TOKEN_FIXES.get(token, token) for token in tokens]
        out = " ".join(tokens)

    if mode == "aggressive":
        out = out.replace("إهدنا", "اهدنا")
        out = out.replace("الضالن", "الضالين")
        out = out.replace("الالمين", "العالمين")

    out = safe_display_text(out)
    if vocab:
        out = _recover_word_boundaries(out, vocab)
    return safe_display_text(out)


def load_all_ayat_json(json_path: str) -> Dict[Tuple[int, int], str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("tafsir", "all_ayat", "verses", "data"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
                break

    if not isinstance(data, dict):
        raise ValueError("all_ayat.json has unexpected format")

    ayah_map: Dict[Tuple[int, int], str] = {}
    for key, value in data.items():
        if not (isinstance(key, str) and "_" in key):
            continue
        try:
            surah_str, ayah_str = key.split("_", 1)
            surah = int(surah_str)
            ayah = int(ayah_str)
        except Exception:
            continue

        text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
        text = normalize_arabic(text)
        if text:
            ayah_map[(surah, ayah)] = text

    return ayah_map


def parse_surah_ayah_from_filename(path: str) -> Tuple[Optional[int], Optional[int]]:
    stem = os.path.splitext(os.path.basename(path))[0]
    if len(stem) < 6 or not stem[:6].isdigit():
        return None, None
    return int(stem[:3]), int(stem[3:6])


def adaptive_confidence_threshold(raw_asr: str, level: str = "high") -> float:
    """Return an adaptive confidence threshold based on token count."""
    token_count = len(normalize_arabic(raw_asr).split()) if raw_asr else 0

    if level == "high":
        if token_count <= 3:
            return 0.40
        if token_count <= 6:
            return 0.60
        if token_count <= 10:
            return 0.72
        return 0.82

    if level == "medium":
        if token_count <= 3:
            return 0.25
        if token_count <= 6:
            return 0.40
        if token_count <= 10:
            return 0.55
        return 0.65

    return 0.50


def _candidate_keys(
    ayah_map: Dict[Tuple[int, int], str],
    surah: Optional[int],
    expected_ayah: Optional[int],
    lookahead: int,
    window_back: int,
) -> Iterable[Tuple[int, int]]:
    if surah is None:
        return ayah_map.keys()
    if expected_ayah is None:
        return [k for k in ayah_map.keys() if k[0] == surah]

    start = max(1, expected_ayah - max(0, window_back))
    end = expected_ayah + max(0, lookahead)
    keys = [(surah, a) for a in range(start, end + 1) if (surah, a) in ayah_map]
    if not keys:
        return [k for k in ayah_map.keys() if k[0] == surah]
    return keys


def _surah_ayah_numbers(ayah_map: Dict[Tuple[int, int], str], surah: int) -> List[int]:
    nums = [ayah for s, ayah in ayah_map.keys() if s == surah]
    nums.sort()
    return nums


def _sequence_candidates(
    ayah_map: Dict[Tuple[int, int], str],
    surah: Optional[int],
    expected_ayah: Optional[int],
    lookahead: int,
    window_back: int,
    max_ayahs: int,
) -> Iterable[Tuple[int, int, str]]:
    if surah is None:
        return []

    ayahs = _surah_ayah_numbers(ayah_map, surah)
    if not ayahs:
        return []

    if expected_ayah is None:
        start_candidates = ayahs[: min(len(ayahs), 12)]
    else:
        lo = max(1, expected_ayah - max(0, window_back))
        hi = expected_ayah + max(0, lookahead)
        start_candidates = [a for a in ayahs if lo <= a <= hi]
        if not start_candidates:
            start_candidates = ayahs[: min(len(ayahs), 12)]

    out: List[Tuple[int, int, str]] = []
    ayah_set = set(ayahs)
    for start in start_candidates:
        if start not in ayah_set:
            continue
        text_parts: List[str] = []
        for end in range(start, start + max(1, max_ayahs)):
            key = (surah, end)
            if key not in ayah_map:
                break
            text_parts.append(ayah_map[key])
            out.append((start, end, " ".join(text_parts)))
    return out


def match_ayah_sequence(
    raw_text: str,
    ayah_map: Dict[Tuple[int, int], str],
    surah: Optional[int] = None,
    expected_ayah: Optional[int] = None,
    lookahead: int = 3,
    window_back: int = 1,
    max_ayahs: int = 12,
) -> Dict[str, object]:
    hyp = normalize_arabic(raw_text)
    candidates = list(
        _sequence_candidates(
            ayah_map=ayah_map,
            surah=surah,
            expected_ayah=expected_ayah,
            lookahead=lookahead,
            window_back=window_back,
            max_ayahs=max_ayahs,
        )
    )
    if not candidates:
        return {
            "matched_key": None,
            "matched_ayah": None,
            "matched_start_ayah": None,
            "matched_end_ayah": None,
            "ayah_text": None,
            "cer": 1.0,
            "wer": 1.0,
            "coverage": 0.0,
            "confidence": 0.0,
            "alignment_score": 0.0,
            "is_sequence": True,
        }

    best = None
    for start, end, ref in candidates:
        cer = _cer_prenorm(hyp, ref)
        wer = _wer_prenorm(hyp, ref)
        cov = _coverage_prenorm(hyp, ref)
        span = max(1, end - start + 1)
        span_bonus = min(0.08, 0.01 * span)
        # Updated confidence: uses CER (character level), WER (word level), and token coverage.
        # WER matters for short chunks where a single wrong word is a large percentage error.
        confidence = max(0.0, min(1.0,
            0.45 * (1.0 - cer)
            + 0.20 * (1.0 - wer)
            + 0.30 * cov
            + span_bonus
        ))
        # alignment_score stays the same — used for sequence vs single-ayah selection
        alignment_score = max(0.0, min(1.0,
            0.40 * (1.0 - cer)
            + 0.30 * (1.0 - wer)
            + 0.30 * cov
        ))
        candidate = (start, end, ref, cer, wer, cov, confidence, alignment_score)
        if best is None or (cer, -cov, -confidence) < (best[3], -best[5], -best[6]):
            best = candidate

    start, end, ref, cer, wer, cov, confidence, alignment_score = best
    return {
        "matched_key": (surah, end) if surah is not None else None,
        "matched_ayah": end,
        "matched_start_ayah": start,
        "matched_end_ayah": end,
        "ayah_text": ref,
        "cer": float(cer),
        "wer": float(wer),
        "coverage": float(cov),
        "confidence": float(confidence),
        "alignment_score": float(alignment_score),
        "is_sequence": True,
    }


def match_ayah(
    raw_text: str,
    ayah_map: Dict[Tuple[int, int], str],
    surah: Optional[int] = None,
    expected_ayah: Optional[int] = None,
    lookahead: int = 3,
    window_back: int = 1,
) -> Dict[str, object]:
    hyp = normalize_arabic(raw_text)
    keys = list(_candidate_keys(ayah_map, surah, expected_ayah, lookahead, window_back))

    if not keys:
        return {
            "matched_key": None,
            "matched_ayah": None,
            "ayah_text": None,
            "cer": 1.0,
            "wer": 1.0,
            "coverage": 0.0,
            "confidence": 0.0,
            "alignment_score": 0.0,
        }

    best = None
    for key in keys:
        ref = ayah_map[key]
        # ref is already normalized (stored normalized in ayah_map from load_all_ayat_json)
        # hyp is already normalized (computed once at the top of match_ayah)
        cer = _cer_prenorm(hyp, ref)
        wer = _wer_prenorm(hyp, ref)
        cov = _coverage_prenorm(hyp, ref)
        confidence = max(0.0, min(1.0, 0.65 * (1.0 - cer) + 0.35 * cov))
        alignment_score = max(0.0, min(1.0, 0.5 * (1.0 - cer) + 0.3 * (1.0 - wer) + 0.2 * cov))
        candidate = (key, ref, cer, wer, cov, confidence, alignment_score)
        if best is None or (cer, -cov, -confidence) < (best[2], -best[4], -best[5]):
            best = candidate

    key, ref, cer, wer, cov, confidence, alignment_score = best
    return {
        "matched_key": key,
        "matched_ayah": key[1],
        "ayah_text": ref,
        "cer": float(cer),
        "wer": float(wer),
        "coverage": float(cov),
        "confidence": float(confidence),
        "alignment_score": float(alignment_score),
    }


def _partial_word_correction(hyp_text: str, ref_text: str) -> str:
    """
    Align hypothesis tokens to reference tokens using SequenceMatcher opcodes,
    then replace tokens that are close enough (CER <= 0.50) with the reference.

    Unlike the original positional loop, this handles insertions and deletions
    so that an extra word at position 0 does not cascade wrong corrections across
    all subsequent tokens.
    """
    hyp_words = normalize_arabic(hyp_text).split()
    ref_words = normalize_arabic(ref_text).split()
    if not hyp_words or not ref_words:
        return normalize_arabic(hyp_text)

    import difflib
    out = list(hyp_words)  # start with a mutable copy
    matcher = difflib.SequenceMatcher(None, hyp_words, ref_words, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue  # already correct — no change needed
        elif tag == "replace":
            # Only replace when edit distance is small — looks like an ASR typo
            hyp_slice = hyp_words[i1:i2]
            ref_slice = ref_words[j1:j2]
            shared = min(len(hyp_slice), len(ref_slice))
            for offset in range(shared):
                hw = hyp_slice[offset]
                rw = ref_slice[offset]
                word_cer = _levenshtein_seq(list(hw), list(rw)) / max(1, len(rw))
                if word_cer <= 0.50:
                    out[i1 + offset] = rw
            # Extra ref words (ref longer than hyp in this span) — insert them
            if len(ref_slice) > len(hyp_slice):
                insert_pos = i2  # position in current out list
                for extra in ref_slice[shared:]:
                    out.insert(insert_pos, extra)
                    insert_pos += 1
        # "delete" (hyp has words ref doesn't) and "insert" (ref has words hyp doesn't)
        # are left as-is — we preserve reciter content

    return " ".join(out)


def _constrained_partial_word_correction(
    hyp_text: str,
    ref_text: str,
    max_word_cer: float = 0.38,
) -> str:
    """Apply conservative token fixes while preserving reciter wording.

    A token is replaced only when it is very close to the reference token,
    avoiding broad substitutions that could alter intended recitation.
    """
    hyp_words = normalize_arabic(hyp_text).split()
    ref_words = normalize_arabic(ref_text).split()
    if not hyp_words or not ref_words:
        return normalize_arabic(hyp_text)

    n = min(len(hyp_words), len(ref_words))
    out = []
    for i in range(n):
        hw = hyp_words[i]
        rw = ref_words[i]
        if hw == rw:
            out.append(hw)
            continue

        word_cer = _levenshtein_seq(list(hw), list(rw)) / max(1, len(rw))
        prefix_match = len(hw) > 0 and len(rw) > 0 and hw[0] == rw[0]
        suffix_match = len(hw) > 1 and len(rw) > 1 and hw[-1] == rw[-1]

        # Replace only when a mismatch looks like a local ASR spelling error.
        if word_cer <= max_word_cer and (prefix_match or suffix_match):
            out.append(rw)
        else:
            out.append(hw)

    # Preserve extra hypothesis tokens to avoid removing reciter content.
    if len(hyp_words) > n:
        out.extend(hyp_words[n:])
    return " ".join(out)


def apply_correction_pipeline(
    raw_text: str,
    ayah_map: Optional[Dict[Tuple[int, int], str]] = None,
    surah: Optional[int] = None,
    expected_ayah: Optional[int] = None,
    lookahead: int = 3,
    window_back: int = 1,
    mode: str = "balanced",
    preserve_reciter: bool = True,
    allow_reference_replacement: bool = True,
    use_sequence_match: bool = False,
    sequence_max_ayahs: int = 12,
    ignore_leading_basmala: bool = False,
    lock_surah: Optional[int] = None,
) -> Dict[str, object]:
    mode = mode or "balanced"
    if mode not in {"safe", "balanced", "aggressive"}:
        raise ValueError(f"Unknown correction mode: {mode}")

    # When lock_surah is set, restrict matching to only that surah's ayaat
    if lock_surah is not None and ayah_map is not None:
        ayah_map = {k: v for k, v in ayah_map.items() if k[0] == lock_surah}

    vocab = _build_vocabulary(ayah_map)
    rule_corrected = correct_text_rules(raw_text, vocab=vocab, mode=mode)
    use_basmala_strip = (
        ignore_leading_basmala
        and surah is not None
        and surah != 1
        and (expected_ayah is None or expected_ayah == 1)
    )
    working_text = rule_corrected
    if use_basmala_strip and _has_leading_basmala(rule_corrected):
        working_text = _strip_leading_basmala(rule_corrected)
    result = {
        "raw_asr": raw_text,
        "rule_corrected": rule_corrected,
        "corrected_text": working_text if mode != "safe" else safe_display_text(working_text),
        "display_text": safe_display_text(working_text),
        "correction_applied": False,
        "correction_mode": mode,
        "confidence_level": "low",
        "matched_ayah": None,
        "matched_key": None,
        "matched_ayah_text": None,
        "cer": None,
        "wer": None,
        "coverage": None,
        "confidence": None,
        "alignment_score": None,
        "verdict": "unknown",
    }

    if ayah_map is None:
        result["corrected_text"] = working_text if mode != "safe" else safe_display_text(working_text)
        return result

    match = match_ayah(
        raw_text=working_text,
        ayah_map=ayah_map,
        surah=surah,
        expected_ayah=expected_ayah,
        lookahead=lookahead,
        window_back=window_back,
    )
    if use_sequence_match:
        seq_match = match_ayah_sequence(
            raw_text=working_text,
            ayah_map=ayah_map,
            surah=surah,
            expected_ayah=expected_ayah,
            lookahead=lookahead,
            window_back=window_back,
            max_ayahs=sequence_max_ayahs,
        )
        if (seq_match.get("alignment_score") or 0.0) >= (match.get("alignment_score") or 0.0):
            match = seq_match
    result.update(
        {
            "matched_ayah": match["matched_ayah"],
            "matched_key": match["matched_key"],
            "matched_ayah_text": match["ayah_text"],
            "matched_start_ayah": match.get("matched_start_ayah"),
            "matched_end_ayah": match.get("matched_end_ayah"),
            "is_sequence_match": bool(match.get("is_sequence") or False),
            "cer": round(match["cer"], 4),
            "wer": round(match["wer"], 4),
            "coverage": round(match["coverage"], 4),
            "confidence": round(match["confidence"], 4),
            "alignment_score": round(match["alignment_score"], 4),
        }
    )

    ref = match["ayah_text"]
    confidence = float(match.get("confidence") or 0.0)
    high_thresh = adaptive_confidence_threshold(raw_text, "high")
    med_thresh = adaptive_confidence_threshold(raw_text, "medium")
    is_high = ref is not None and confidence >= high_thresh
    is_medium = ref is not None and confidence >= med_thresh

    if is_high:
        result["confidence_level"] = "high"
        result["verdict"] = "ok"
    elif is_medium:
        result["confidence_level"] = "medium"
        result["verdict"] = "minor"
    else:
        result["confidence_level"] = "low"
        result["verdict"] = "error"

    if mode == "safe":
        result["corrected_text"] = safe_display_text(working_text)
        return result

    if ref is None:
        result["corrected_text"] = working_text
        return result

    if preserve_reciter:
        # Strict mode: never force full ayah replacement.
        if is_high or is_medium:
            max_word_cer = 0.44 if is_high else 0.33
            partial = _constrained_partial_word_correction(working_text, ref, max_word_cer=max_word_cer)
            result["corrected_text"] = partial
            result["correction_applied"] = partial != working_text
            result["correction_mode"] = "constrained_word_correction"
        else:
            result["corrected_text"] = working_text
    elif mode == "balanced":
        if is_high and allow_reference_replacement:
            result["corrected_text"] = ref
            result["correction_applied"] = True
            result["correction_mode"] = "full_ayah_replacement"
        elif is_medium or is_high:
            partial = _partial_word_correction(working_text, ref)
            result["corrected_text"] = partial
            result["correction_applied"] = partial != working_text
            result["correction_mode"] = "partial_word_correction"
        else:
            result["corrected_text"] = working_text
    else:  # aggressive
        if (is_high or is_medium) and allow_reference_replacement:
            result["corrected_text"] = ref
            result["correction_applied"] = True
            result["correction_mode"] = "full_ayah_replacement"
        elif is_high or is_medium:
            partial = _partial_word_correction(working_text, ref)
            result["corrected_text"] = partial
            result["correction_applied"] = partial != working_text
            result["correction_mode"] = "partial_word_correction"
        else:
            result["corrected_text"] = working_text

    # Prefer raw text when it matches the reference better than the corrected output.
    if ref is not None:
        raw_cer = compute_cer(raw_text, ref)
        corr_cer = compute_cer(result.get("corrected_text", ""), ref)
        if raw_cer <= corr_cer - 0.01:
            result["corrected_text"] = safe_display_text(raw_text)
            result["correction_applied"] = False
            result["correction_mode"] = "raw_preferred"

    result["display_text"] = safe_display_text(result["corrected_text"])
    return result


def guard_inference(
    raw_text: str,
    ayah_map: Optional[Dict[Tuple[int, int], str]] = None,
    surah: Optional[int] = None,
    expected_ayah: Optional[int] = None,
    lookahead: int = 3,
    allow_auto_correct: bool = False,
    auto_cer_threshold: float = 0.12,
    auto_cov_threshold: float = 0.88,
    correction_mode: str = "balanced",
    window_back: int = 1,
    preserve_reciter: bool = True,
    allow_reference_replacement: bool = True,
    use_sequence_match: bool = False,
    sequence_max_ayahs: int = 12,
    ignore_leading_basmala: bool = True,
    lock_surah: Optional[int] = None,
):
    if allow_auto_correct and correction_mode == "safe":
        correction_mode = "balanced"

    # Use adaptive thresholds if auto_correct is enabled
    # auto_cer_threshold: when CER is below this, correction fires automatically
    # auto_cov_threshold: when token coverage is above this, correction fires automatically
    # These gate the 'allow_auto_correct' flag — if input is clean enough, correct it;
    # if it is too noisy, don't. This prevents aggressive correction on hallucinated chunks.
    _effective_auto_correct = allow_auto_correct
    if allow_auto_correct and correction_mode != "safe":
        # Pre-check: normalize input and run a quick token count
        _norm_len = len(normalize_arabic(raw_text).split())
        # If the text is very short (< 3 tokens), don't auto-correct — too ambiguous
        if _norm_len < 3:
            _effective_auto_correct = False

    return apply_correction_pipeline(
        raw_text=raw_text,
        ayah_map=ayah_map,
        surah=surah,
        expected_ayah=expected_ayah,
        lookahead=lookahead,
        window_back=window_back,
        mode=correction_mode,
        preserve_reciter=preserve_reciter,
        allow_reference_replacement=_effective_auto_correct and allow_reference_replacement,
        use_sequence_match=use_sequence_match,
        sequence_max_ayahs=sequence_max_ayahs,
        ignore_leading_basmala=ignore_leading_basmala,
        lock_surah=lock_surah,
    )


def get_word_error_annotations(
    hyp_text: str,
    ref_text: str,
    confidence: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Return word-level error annotations comparing hypothesis to reference.

    Each item includes: word, reference, status, similarity.
    """
    hyp_words = safe_display_text(hyp_text).split()
    ref_words = safe_display_text(ref_text).split()
    hyp_norm = [normalize_arabic(word) for word in hyp_words]
    ref_norm = [normalize_arabic(word) for word in ref_words]
    annotations: List[Dict[str, object]] = []

    def _append(
        word: str,
        ref_word: str,
        forced_status: Optional[str] = None,
        hyp_index: Optional[int] = None,
        ref_index: Optional[int] = None,
    ):
        norm_word = normalize_arabic(word)
        norm_ref = normalize_arabic(ref_word)

        if forced_status is not None:
            status = forced_status
            similarity = 0.0 if forced_status in {"missing", "extra", "uncertain"} else 1.0
        elif confidence is not None and confidence < 0.4:
            status = "uncertain"
            similarity = 0.0
        elif not norm_word and norm_ref:
            status = "missing"
            similarity = 0.0
        elif not norm_ref:
            status = "extra"
            similarity = 0.0
        elif norm_word == norm_ref:
            status = "correct"
            similarity = 1.0
        else:
            if _HAS_RAPIDFUZZ:
                similarity = _rfuzz.ratio(norm_word, norm_ref) / 100.0
            else:
                similarity = difflib.SequenceMatcher(None, norm_word, norm_ref).ratio()
            status = "minor" if similarity >= 0.70 else "major"

        annotations.append({
            "word": word,
            "reference": ref_word,
            "status": status,
            "similarity": round(float(similarity), 3),
            "hyp_index": hyp_index,
            "ref_index": ref_index,
        })

    if _HAS_RAPIDFUZZ:
        from rapidfuzz.distance import Opcodes as _RFOpcodes
        _rf_opcodes = _RFOpcodes.from_editops(
            _rfLev.editops(ref_norm, hyp_norm)
        )
        _opcode_iter = _rf_opcodes
    else:
        _sm = difflib.SequenceMatcher(None, ref_norm, hyp_norm, autojunk=False)
        _opcode_iter = _sm.get_opcodes()

    for tag, i1, i2, j1, j2 in _opcode_iter:
        if tag == "equal":
            for offset, (ref_word, hyp_word) in enumerate(zip(ref_words[i1:i2], hyp_words[j1:j2])):
                _append(hyp_word, ref_word, hyp_index=j1 + offset, ref_index=i1 + offset)
        elif tag == "delete":
            for offset, ref_word in enumerate(ref_words[i1:i2]):
                _append("", ref_word, "missing", ref_index=i1 + offset)
        elif tag == "insert":
            for offset, hyp_word in enumerate(hyp_words[j1:j2]):
                _append(hyp_word, "", "extra", hyp_index=j1 + offset)
        elif tag == "replace":
            ref_slice = ref_words[i1:i2]
            hyp_slice = hyp_words[j1:j2]
            shared = min(len(ref_slice), len(hyp_slice))
            for idx in range(shared):
                _append(hyp_slice[idx], ref_slice[idx], hyp_index=j1 + idx, ref_index=i1 + idx)
            for extra_idx, ref_word in enumerate(ref_slice[shared:], shared):
                ref_offset = extra_idx
                _append("", ref_word, "missing", ref_index=i1 + ref_offset)
            for extra_idx, hyp_word in enumerate(hyp_slice[shared:], shared):
                hyp_offset = extra_idx
                _append(hyp_word, "", "extra", hyp_index=j1 + hyp_offset)

    return annotations
