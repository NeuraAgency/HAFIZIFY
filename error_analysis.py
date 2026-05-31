"""
Qur'an ASR Error Analysis Module
--------------------------------
Evaluates raw ASR predictions against reference ayahs to detect specific word-level errors 
(missing, extra, substituted) without altering the raw input text.
Designed to be fast (using rapidfuzz) and robust (handling Arabic string edge cases).
"""

import json
import re
import difflib
import time
from typing import Dict, List, Any

# We use rapidfuzz for extremely fast Levenshtein matching over 6200+ sequences
try:
    from rapidfuzz import process, fuzz
except ImportError:
    raise ImportError("rapidfuzz is required. Install it using: pip install rapidfuzz")


class QuranASREvaluator:
    def __init__(self, reference_json_path: str):
        """
        Args:
            reference_json_path (str): Path to the all_ayat.json file containing the reference Qur'an dataset.
        """
        self._load_reference(reference_json_path)

    def _load_reference(self, path: str):
        """Loads and pre-caches the normalized text of all reference ayahs to optimize matching speed."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break

        if not isinstance(data, dict):
            raise ValueError("Reference JSON has unexpected format")

        self.ayahs: List[Dict[str, str]] = []
        self.cached_normalized_texts: List[str] = []

        for key, value in data.items():
            if not (isinstance(key, str) and "_" in key):
                continue
            try:
                surah_str, ayah_str = key.split("_", 1)
                surah_num = int(surah_str)
                ayah_num = int(ayah_str)
            except Exception:
                continue

            raw_text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
            norm_text = self.normalize_arabic_for_eval(raw_text)

            metadata = {
                "surah": surah_num,
                "ayah": ayah_num,
                "id": f"{surah_num}:{ayah_num}",
                "raw_text": raw_text,
                "normalized_text": norm_text,
            }
            self.ayahs.append(metadata)
            self.cached_normalized_texts.append(norm_text)
        
        print(f"Loaded and normalized {len(self.ayahs)} ayahs for evaluation.")

    @staticmethod
    def _has_leading_basmala(text: str) -> bool:
        tokens = text.split()
        return len(tokens) >= 4 and tokens[:4] == ["بسم", "الله", "الرحمن", "الرحيم"]

    @staticmethod
    def _strip_leading_basmala(text: str) -> str:
        tokens = text.split()
        if len(tokens) >= 4:
            return " ".join(tokens[4:]).strip()
        return ""

    def _candidate_indices(
        self,
        surah: int | None,
        expected_ayah: int | None,
        lookahead: int,
        window_back: int,
    ) -> List[int]:
        if surah is None:
            return list(range(len(self.ayahs)))

        if expected_ayah is None:
            return [i for i, a in enumerate(self.ayahs) if a["surah"] == surah]

        start = max(1, expected_ayah - max(0, window_back))
        end = expected_ayah + max(0, lookahead)
        return [
            i
            for i, a in enumerate(self.ayahs)
            if a["surah"] == surah and start <= a["ayah"] <= end
        ]

    @staticmethod
    def normalize_arabic_for_eval(text: str) -> str:
        """
        Crucial robust normalization for ASR vs Reference alignment.
        Removes diacritics and conflates letters that represent the same spoken sound.
        """
        if not text:
            return ""

        # 1. Strip diacritics (tashkeel), tatweel, and small superscript letters (often present in Tanzil JSON)
        text = re.sub(
            r"[\u0610-\u061A\u064B-\u065F\u0670\u0640\u06D6-\u06ED"
            r"\u00AB\u00BB\u200F\u200E\u202A-\u202E\uFEFF\u06DD\u06DE"
            r"۩۞۝]+", "", text
        )

        # 2. Conflate Alef forms
        text = re.sub(r"[أإآٱ]", "ا", text)

        # 3. Yaa vs Alef Maksura (often confused in ASR outputs)
        text = re.sub(r"ى", "ي", text)

        # 4. Ta Marbuta vs Haa (Often pronounced as Haa when pausing, causing ASR to output 'ه')
        text = re.sub(r"ة", "ه", text)

        # 5. Remove any remaining punctuation and reduce whitespace
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def retrieve_best_ayahs(
        self,
        norm_asr_text: str,
        top_k: int = 3,
        surah: int | None = None,
        expected_ayah: int | None = None,
        lookahead: int = 3,
        window_back: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Uses RapidFuzz to fetch the best fitting reference ayahs based on Levenshtein distance.
        RapidFuzz `process.extract` is implemented in C++ and can search 6000+ strings in ~1-5ms.
        """
        if not norm_asr_text:
            return []

        # extract() returns a list of tuples: (matched_string, similarity_score, index_in_list)
        candidate_indices = self._candidate_indices(surah, expected_ayah, lookahead, window_back)
        if not candidate_indices:
            return []

        candidate_texts = [self.cached_normalized_texts[i] for i in candidate_indices]
        results = process.extract(
            norm_asr_text,
            candidate_texts,
            scorer=fuzz.ratio,
            limit=top_k,
        )

        candidates = []
        for match_str, score, idx in results:
            ayah_idx = candidate_indices[idx]
            candidates.append({
                "ayah_data": self.ayahs[ayah_idx],
                "score": score
            })

        return candidates

    def align_words(self, asr_text: str, ref_text: str) -> List[Dict[str, str]]:
        """
        Detects missing, extra, and substituted words.
        Uses Python's difflib SequenceMatcher which implements the Ratcliff/Obershelp algorithm.
        """
        asr_tokens = asr_text.split()
        ref_tokens = ref_text.split()

        matcher = difflib.SequenceMatcher(None, ref_tokens, asr_tokens)
        errors = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            elif tag == 'delete':
                # Word exists in reference (i1:i2), but not in ASR
                for word in ref_tokens[i1:i2]:
                    errors.append({
                        "type": "missing",
                        "word": word
                    })
            
            elif tag == 'insert':
                # Word exists in ASR (j1:j2), but not in reference
                for word in asr_tokens[j1:j2]:
                    errors.append({
                        "type": "extra",
                        "word": word
                    })
            
            elif tag == 'replace':
                # Substituted words. difflib groups contiguous mismatches together.
                # We join the groups to show the chunk mismatch clearly.
                expected_phrase = " ".join(ref_tokens[i1:i2])
                got_phrase = " ".join(asr_tokens[j1:j2])
                errors.append({
                    "type": "substitution",
                    "expected": expected_phrase,
                    "got": got_phrase
                })

        return errors

    def evaluate(
        self,
        raw_asr_text: str,
        confidence_threshold: float = 85.0,
        surah: int | None = None,
        expected_ayah: int | None = None,
        lookahead: int = 3,
        window_back: int = 1,
        ignore_leading_basmala: bool = True,
    ) -> Dict[str, Any]:
        raise RuntimeError(
            "Global matching evaluator disabled. Use HybridViterbiPipeline.pipeline_from_text() instead."
        )


if __name__ == "__main__":
    # Demo: shows the word-level alignment utility that is still active
    print("Initializing Qur'an ASR Evaluator (word-alignment demo)...")

    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.join(current_dir, "fyp_model", "all_ayat.json")
    if not os.path.exists(reference_path):
        reference_path = os.path.join(current_dir, "data", "all_ayat.json")

    evaluator = QuranASREvaluator(reference_json_path=reference_path)

    # Demonstrate word-level alignment between a raw ASR output and a reference ayah
    sample_pairs = [
        (
            "الحمد لله رب العالمين",           # reference (correct)
            "الحمد لله رب العلمين",             # ASR (substitution: العلمين instead of العالمين)
        ),
        (
            "مالك يوم الدين",                   # reference
            "املك يوم",                          # ASR (substitution + missing word)
        ),
    ]

    for i, (ref, asr) in enumerate(sample_pairs, 1):
        print(f"\n===== WORD ALIGNMENT {i} =====")
        print(f"Reference : {ref}")
        print(f"ASR output: {asr}")
        norm_ref = QuranASREvaluator.normalize_arabic_for_eval(ref)
        norm_asr = QuranASREvaluator.normalize_arabic_for_eval(asr)
        errors = evaluator.align_words(norm_asr, norm_ref)
        print("Errors detected:")
        print(json.dumps(errors, indent=2, ensure_ascii=False))

    print("\nNote: Global evaluate() is disabled. Use HybridViterbiPipeline for full alignment.")

