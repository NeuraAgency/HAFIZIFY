"""
harakaat_scoring_eval.py
-------------------------
Evaluation harness for the word + harakaat scoring scope Hamza specified
on 2026-08-03: score every word on consonant-skeleton correctness, and
ADDITIONALLY score zabar/zeer/pesh (fatha/kasra/damma) correctness only
when the word is skeleton-correct and actually carries one of those three
marks. No separate "makhraj" label, no sukun/shadda/tanween/dagger-alif in
the harakaat check.

WHY A SYNTHETIC DATASET:
No human-labeled recitation clips exist yet. This script builds ground
truth by taking real diacritized ayahs from fyp_model/all_ayat.json and
injecting KNOWN corruptions with KNOWN labels, then checking whether
classify_word() recovers those labels. This tells you the scoring logic
itself is correct and gives you a confusion matrix / accuracy / precision
/ recall / F1 you can report today.

IT IS NOT an ASR-accuracy or real-world-error-catching eval — it can't be,
without labeled recitation audio. When you have that, replace
build_synthetic_dataset() with a loader that yields
(predicted_word, reference_word, is_last_word, true_word_label,
true_harakaat_label) tuples pulled from your labeled clips, and reuse
everything from evaluate() down unchanged.

Usage:
    python harakaat_scoring_eval.py --n-ayahs 300 --seed 42
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0652\u0670]")
# zabar, zeer, pesh — the ONLY marks the harakaat check looks at
SHORT_VOWELS = {"\u064E": "fatha", "\u0650": "kasra", "\u064F": "damma"}
_DAGGER_ALIF = "\u0670"  # maad marker — deliberately excluded, see module docstring
_ALEF_VARIANTS = re.compile(r"[إأٱآ]")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AYAT_JSON = os.path.join(_BASE_DIR, "fyp_model", "all_ayat.json")


# ---------------------------------------------------------------------------
# Core scoring unit
# ---------------------------------------------------------------------------

def _normalize_skeleton(word: str) -> str:
    word = DIACRITICS_PATTERN.sub("", word)
    word = _ALEF_VARIANTS.sub("ا", word)
    word = word.replace("ى", "ي").replace("ة", "ه")
    return word


def _decompose(word: str) -> List[Tuple[str, str]]:
    """Split into (base_char, diacritic_cluster) pairs, one per base
    character — same approach as harakaat_error_detector.py."""
    pairs: List[Tuple[str, str]] = []
    base: Optional[str] = None
    cluster: List[str] = []
    for ch in word:
        if DIACRITICS_PATTERN.match(ch):
            cluster.append(ch)
        else:
            if base is not None:
                pairs.append((base, "".join(cluster)))
            base = ch
            cluster = []
    if base is not None:
        pairs.append((base, "".join(cluster)))
    return pairs


@dataclass
class WordVerdict:
    word_status: str       # "correct" | "incorrect"
    harakaat_status: str   # "correct" | "incorrect" | "not_applicable"


def classify_word(predicted_word: str, reference_word: str, is_last_word: bool) -> WordVerdict:
    """Word check always runs. Harakaat check (zabar/zeer/pesh only) runs
    only when the word is skeleton-correct AND at least one of those three
    marks appears at a comparable position."""
    pred_skel = _normalize_skeleton(predicted_word)
    ref_skel = _normalize_skeleton(reference_word)

    if pred_skel != ref_skel:
        return WordVerdict(word_status="incorrect", harakaat_status="not_applicable")

    ref_clean = reference_word.replace(_DAGGER_ALIF, "")
    pred_clean = predicted_word.replace(_DAGGER_ALIF, "")

    pred_pairs = _decompose(pred_clean)
    ref_pairs = _decompose(ref_clean)

    if len(pred_pairs) != len(ref_pairs):
        # Skeleton matched but cluster count didn't line up — can't do a
        # safe positional comparison. Word is still correct; just can't
        # judge harakaat for it.
        return WordVerdict(word_status="correct", harakaat_status="not_applicable")

    compare_range = range(len(ref_pairs) - 1) if is_last_word else range(len(ref_pairs))

    applicable = False
    mismatch = False
    for i in compare_range:
        _, ref_cluster = ref_pairs[i]
        _, pred_cluster = pred_pairs[i]
        ref_vowel = next((c for c in ref_cluster if c in SHORT_VOWELS), None)
        pred_vowel = next((c for c in pred_cluster if c in SHORT_VOWELS), None)

        if ref_vowel is None and pred_vowel is None:
            continue  # neither side has zabar/zeer/pesh here — not this check's concern

        applicable = True
        if ref_vowel != pred_vowel:
            mismatch = True

    if not applicable:
        return WordVerdict(word_status="correct", harakaat_status="not_applicable")

    return WordVerdict(word_status="correct", harakaat_status="incorrect" if mismatch else "correct")


# ---------------------------------------------------------------------------
# Synthetic labeled dataset (replace with real labeled data when available)
# ---------------------------------------------------------------------------

_ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


def _load_ayat(json_path: str = _DEFAULT_AYAT_JSON) -> Dict[Tuple[int, int], str]:
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
        if text and DIACRITICS_PATTERN.search(text):
            ayah_map[(surah, ayah)] = text
    return ayah_map


def _swap_short_vowel(word: str, rng: random.Random, is_last_word: bool) -> Optional[str]:
    """Flip one zabar/zeer/pesh to a different one of the three, at a
    position that is NOT the excluded final i'rab slot. Returns None if no
    eligible position exists."""
    pairs = _decompose(word)
    eligible = [
        i for i, (_, cluster) in enumerate(pairs)
        if any(c in SHORT_VOWELS for c in cluster) and not (is_last_word and i == len(pairs) - 1)
    ]
    if not eligible:
        return None
    idx = rng.choice(eligible)
    base, cluster = pairs[idx]
    others = [v for v in SHORT_VOWELS if v not in cluster]
    new_vowel = rng.choice(others)
    new_cluster = "".join(new_vowel if c in SHORT_VOWELS else c for c in cluster)
    pairs[idx] = (base, new_cluster)
    return "".join(b + c for b, c in pairs)


def _swap_consonant(word: str, rng: random.Random) -> Optional[str]:
    """Change one base letter (never the last, to dodge waqf handling) to
    a different Arabic letter, keeping its diacritic cluster."""
    pairs = _decompose(word)
    if len(pairs) < 2:
        return None
    idx = rng.randrange(0, len(pairs) - 1)
    base, cluster = pairs[idx]
    candidates = [c for c in _ARABIC_LETTERS if c != base]
    pairs[idx] = (rng.choice(candidates), cluster)
    return "".join(b + c for b, c in pairs)


@dataclass
class LabeledExample:
    predicted_word: str
    reference_word: str
    is_last_word: bool
    true_word_label: str
    true_harakaat_label: str


def build_synthetic_dataset(
    n_ayahs: int = 300,
    seed: int = 42,
    p_word_error: float = 0.15,
    p_harakaat_error: float = 0.25,
    ayat_json: str = _DEFAULT_AYAT_JSON,
) -> List[LabeledExample]:
    rng = random.Random(seed)
    ayat = _load_ayat(ayat_json)
    keys = list(ayat.keys())
    rng.shuffle(keys)
    keys = keys[:n_ayahs]

    examples: List[LabeledExample] = []
    for key in keys:
        ref_text = ayat[key]
        ref_words = ref_text.strip().split()
        for i, ref_word in enumerate(ref_words):
            is_last = i == len(ref_words) - 1
            roll = rng.random()

            if roll < p_word_error:
                corrupted = _swap_consonant(ref_word, rng)
                if corrupted is not None:
                    examples.append(LabeledExample(
                        predicted_word=corrupted, reference_word=ref_word, is_last_word=is_last,
                        true_word_label="incorrect", true_harakaat_label="not_applicable",
                    ))
                    continue

            if roll < p_word_error + p_harakaat_error:
                corrupted = _swap_short_vowel(ref_word, rng, is_last)
                if corrupted is not None:
                    examples.append(LabeledExample(
                        predicted_word=corrupted, reference_word=ref_word, is_last_word=is_last,
                        true_word_label="correct", true_harakaat_label="incorrect",
                    ))
                    continue

            # Unchanged — figure out the true harakaat applicability/label
            # straight from the reference word itself.
            verdict = classify_word(ref_word, ref_word, is_last)
            examples.append(LabeledExample(
                predicted_word=ref_word, reference_word=ref_word, is_last_word=is_last,
                true_word_label="correct", true_harakaat_label=verdict.harakaat_status,
            ))

    return examples


# ---------------------------------------------------------------------------
# Metrics (no external deps — plain confusion matrix / precision / recall / F1)
# ---------------------------------------------------------------------------

def _confusion_and_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    total = len(y_true)
    correct = sum(matrix[l][l] for l in labels)
    accuracy = correct / total if total else 0.0

    per_class = {}
    for l in labels:
        tp = matrix[l][l]
        fp = sum(matrix[o][l] for o in labels if o != l)
        fn = sum(matrix[l][o] for o in labels if o != l)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(matrix[l].values())
        per_class[l] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels) if labels else 0.0

    return {"matrix": matrix, "accuracy": accuracy, "per_class": per_class, "macro_f1": macro_f1, "labels": labels}


def _print_report(title: str, result: Dict):
    labels = result["labels"]
    print(f"\n=== {title} ===")
    print(f"Accuracy: {result['accuracy']:.4f}   Macro F1: {result['macro_f1']:.4f}")

    header = "true\\pred".ljust(14) + "".join(l.ljust(14) for l in labels)
    print(header)
    for t in labels:
        row = t.ljust(14) + "".join(str(result["matrix"][t][p]).ljust(14) for p in labels)
        print(row)

    print(f"\n{'class'.ljust(14)}{'precision'.ljust(12)}{'recall'.ljust(12)}{'f1'.ljust(12)}{'support'.ljust(10)}")
    for l in labels:
        m = result["per_class"][l]
        print(f"{l.ljust(14)}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1']:<12.4f}{m['support']:<10}")


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def evaluate(examples: List[LabeledExample]):
    word_true, word_pred = [], []
    harakaat_true, harakaat_pred = [], []

    for ex in examples:
        verdict = classify_word(ex.predicted_word, ex.reference_word, ex.is_last_word)

        word_true.append(ex.true_word_label)
        word_pred.append(verdict.word_status)

        # Harakaat task only scored where truth says it's applicable —
        # "not_applicable" isn't a class to confuse into, it's "this row
        # doesn't belong to this task".
        if ex.true_harakaat_label != "not_applicable":
            harakaat_true.append(ex.true_harakaat_label)
            harakaat_pred.append(verdict.harakaat_status if verdict.harakaat_status != "not_applicable" else "incorrect")

    word_result = _confusion_and_metrics(word_true, word_pred, ["correct", "incorrect"])
    harakaat_result = _confusion_and_metrics(harakaat_true, harakaat_pred, ["correct", "incorrect"])

    _print_report("WORD-LEVEL (consonant skeleton)", word_result)
    _print_report("HARAKAAT-LEVEL (zabar / zeer / pesh only, applicable words)", harakaat_result)

    print(f"\nTotal words scored: {len(examples)}")
    print(f"Harakaat-applicable subset: {len(harakaat_true)}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic word+harakaat scoring eval")
    parser.add_argument("--n-ayahs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ayat-json", type=str, default=_DEFAULT_AYAT_JSON)
    args = parser.parse_args()

    examples = build_synthetic_dataset(n_ayahs=args.n_ayahs, seed=args.seed, ayat_json=args.ayat_json)
    evaluate(examples)


if __name__ == "__main__":
    main()
