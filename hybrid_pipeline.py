"""
Hybrid Viterbi ASR Pipeline
---------------------------
Implements a production-grade Tarteel-style ASR architecture.
Combines:
  - CTC Beam Search & KenLM 
  - N-gram feature extraction & Jaccard overlap scoring
  - Rapidfuzz Levenshtein edit distance
  - Strict Dynamic Programming (Viterbi) sequential alignment
"""

import json
import re
from fyp_model.quran_guard import normalize_arabic
import math
import os
import difflib
from typing import Dict, List, Any, Optional

try:
    from rapidfuzz import fuzz
except ImportError:
    raise ImportError("pip install rapidfuzz")

try:
    import kenlm  # type: ignore[import-not-found]
    _KENLM_AVAILABLE = True
except ImportError:
    kenlm = None
    _KENLM_AVAILABLE = False
    print("[HybridPipeline] WARNING: kenlm not installed — LM scoring disabled, using n-gram + edit distance only.")




# CRITICAL: Only strip Ta'awwuz, NOT Basmala
# Basmala is the first ayah of 113 surahs and critical for alignment
_LEADING_INVOCATIONS = (
    "اعوذ بالله من الشيطان الرجيم",
    # "بسم الله الرحمن الرحيم",  # REMOVED — it's part of ayahs
)


def _strip_leading_invocations(
    text: str,
    max_prefix_words: int = 60,
    min_ratio: int = 56,
) -> str:
    if not text:
        return text

    original_tokens = re.split(r"\s+", text.strip())
    norm_tokens = normalize_arabic(text).split()
    if not norm_tokens or not original_tokens:
        return text

    for phrase in _LEADING_INVOCATIONS:
        phrase_norm = normalize_arabic(phrase)
        phrase_tokens = phrase_norm.split()
        if not phrase_tokens:
            continue

        prefix_tokens = norm_tokens[:max_prefix_words]
        if len(prefix_tokens) < 2:
            break

        min_tokens = max(2, int(len(phrase_tokens) * 0.6))
        max_tokens = min(len(prefix_tokens), len(phrase_tokens) + 3)

        best_len = 0
        best_score = 0
        for length in range(min_tokens, max_tokens + 1):
            candidate = " ".join(prefix_tokens[:length])
            score = fuzz.ratio(candidate, phrase_norm)
            if score > best_score:
                best_score = score
                best_len = length

        if best_score >= min_ratio and best_len > 0:
            norm_tokens = norm_tokens[best_len:]
            original_tokens = original_tokens[best_len:]

    return " ".join(original_tokens).strip()


class NGramScorer:
    """Extracts Unigram, Bigram, and Trigrams to compute Jaccard similarity constraints."""
    
    @staticmethod
    def get_ngrams(text: str, n: int) -> List[str]:
        words = text.split()
        if len(words) < n:
            return [" ".join(words)] if words else []
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    @staticmethod
    def get_all_ngrams(text: str) -> List[str]:
        return (
            NGramScorer.get_ngrams(text, 1) + 
            NGramScorer.get_ngrams(text, 2) + 
            NGramScorer.get_ngrams(text, 3)
        )


class HybridViterbiPipeline:
    def __init__(self, ayah_json_path: str, kenlm_path: str):
        if _KENLM_AVAILABLE and os.path.isfile(kenlm_path):
            self.lm_model = kenlm.Model(kenlm_path)
            print(f"[HybridPipeline] KenLM model loaded from: {kenlm_path}")
        else:
            self.lm_model = None
            if _KENLM_AVAILABLE:
                print(f"[HybridPipeline] WARNING: KenLM is installed but LM file not found at: {kenlm_path}")
            else:
                print("[HybridPipeline] WARNING: kenlm not installed — LM scoring disabled.")
        self._load_reference(ayah_json_path)
        
    def _load_reference(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("tafsir", "all_ayat", "verses", "data"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break

        self.ayahs = []
        self.ayah_ngrams = []
        
        idx = 0
        for key, value in data.items():
            if not (isinstance(key, str) and "_" in key):
                continue
            try:
                surah, ayah = map(int, key.split("_", 1))
            except Exception:
                continue

            raw_text = str(value.get("text", "")) if isinstance(value, dict) else str(value or "")
            norm = normalize_arabic(raw_text)
            
            self.ayahs.append({
                "global_idx": idx,
                "surah": surah,
                "ayah": ayah,
                "id": f"{surah}:{ayah}",
                "raw_text": raw_text,
                "norm_text": norm
            })
            self.ayah_ngrams.append(set(NGramScorer.get_all_ngrams(norm)))
            idx += 1
            
        # Build inverted trigram index for fast candidate pre-selection
        # Key: ngram string → list of ayah global_idx values
        # At query time: look up chunk ngrams, tally hit counts per ayah,
        # take top-30 by count, then only run fuzz.ratio on those 30.
        import collections as _collections
        self._ngram_index: dict = _collections.defaultdict(list)
        for idx, ayah in enumerate(self.ayahs):
            tokens = ayah["norm_text"].split()
            # Index bigrams AND trigrams — unigrams are too common to be useful
            for n in (2, 3):
                for j in range(len(tokens) - n + 1):
                    gram = " ".join(tokens[j : j + n])
                    self._ngram_index[gram].append(idx)
        print(
            f"[HybridPipeline] Inverted index built: "
            f"{len(self._ngram_index)} unique bigrams+trigrams."
        )
        print(f"[HybridPipeline] Loaded {len(self.ayahs)} ayahs into sequential space.")

    def _get_candidate_indices(self, chunk_ngrams: set, top_k: int = 30) -> list:
        """
        Use the inverted index to find the top-k ayah indices most likely to
        match the given chunk, based on weighted n-gram hit counts.

        This is O(|chunk_ngrams|) instead of O(N_ayahs) and runs in ~1ms.
        Only the returned indices are passed to the expensive _score_chunk_against_ayah.

        Args:
            chunk_ngrams: set of all n-gram strings from the chunk
            top_k: how many candidate ayahs to return

        Returns:
            List of ayah global_idx integers, sorted by hit weight descending.
        """
        hit_counts: dict = {}
        for gram in chunk_ngrams:
            if gram not in self._ngram_index:
                continue
            # Weight by n-gram length: trigrams=3, bigrams=2
            weight = len(gram.split())
            for ayah_idx in self._ngram_index[gram]:
                hit_counts[ayah_idx] = hit_counts.get(ayah_idx, 0) + weight

        if not hit_counts:
            return []

        # Sort by hit weight descending, return top_k
        sorted_candidates = sorted(hit_counts, key=hit_counts.get, reverse=True)
        return sorted_candidates[:top_k]

    def _score_chunk_against_ayah(self, chunk_text: str, chunk_ngrams: set, ayah_idx: int) -> float:
        """
        Emission Probability:
        final_score = 0.5 * KenLM beam score + 0.3 * n-gram overlap score + 0.2 * edit distance score
        """
        ref_ayah = self.ayahs[ayah_idx]
        ref_ngrams = self.ayah_ngrams[ayah_idx]
        
        # 1. N-Gram Overlap Score (0.0 to 1.0)
        # Uses Overlap (Intersection/Input) instead of Jaccard (Intersection/Union)
        # so that long ayahs are not unfairly penalized.
        if not chunk_ngrams:
            overlap = 0.0
        else:
            intersection = len(chunk_ngrams.intersection(ref_ngrams))
            overlap = intersection / len(chunk_ngrams) if chunk_ngrams else 0.0
            
        # Optimization: shortcircuit if 0 overlap
        if overlap == 0:
            return 0.0

        # 2. Levenshtein Edit Distance (using ratio for tighter matching)
        edit_score = fuzz.ratio(chunk_text, ref_ayah['norm_text']) / 100.0
        
        # 3. KenLM Score evaluation
        # we score the chunk itself to see how "Quranic" the phrase behaves within the 5-gram space
        if self.lm_model:
            lm_log10_prob = self.lm_model.score(chunk_text, bos=False, eos=False)
            words = len(chunk_text.split()) or 1
            avg_prob_per_word = lm_log10_prob / words

            # Calibrated mapping: −2.0 per word → 1.0 (excellent Quran match)
            #                     −5.0 per word → 0.0 (poor / non-Quran)
            # Linear interpolation between those two anchors.
            _LM_BEST  = -2.0   # avg log10 prob for a near-perfect Quran phrase
            _LM_WORST = -5.0   # avg log10 prob for unrelated text
            lm_normalized = max(
                0.0,
                min(1.0, (avg_prob_per_word - _LM_WORST) / (_LM_BEST - _LM_WORST))
            )
            
            # Weighted aggregate
            final_score = (0.5 * lm_normalized) + (0.3 * overlap) + (0.2 * edit_score)
        else:
            # Fallback when KenLM is disabled: purely use overlap and edit distance
            final_score = (0.6 * overlap) + (0.4 * edit_score)
        return final_score

    def viterbi_align(
        self,
        asr_text: str,
        chunk_size: int = 5,
        step: int = 3,
        start_surah: Optional[int] = None,
        lock_surah: Optional[int] = None,
        strip_invocations: bool = True,
    ) -> Dict[str, Any]:
        """
        Chunks the ASR text and routes it through a Viterbi DP grid against the structured Qur'an corpus to optimally 
        align ayahs while penalizing hallucinated jumps.
        Includes CONFIDENCE-BASED SURAH TRACKING with CONSISTENT DELAYED LOCKING.
        """
        if strip_invocations:
            asr_text = _strip_leading_invocations(asr_text)

        norm_asr = normalize_arabic(asr_text)
        words = norm_asr.split()
        
        if not words:
            return self._empty_result(asr_text)
            
        # Step 1: Chunking
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        # Time dimension T = number of chunks
        T = len(chunks)
        if T == 0:
            return self._empty_result(asr_text)

        # Step 2: Extract top-K emissions per chunk to restrict DP state space to reasonable candidates
        K_CANDIDATES = 20
        emissions = [] # list of lists: [{'idx': int, 'score': float}, ...]
        raw_scores_per_chunk = []
        
        # --- SURAH TRACKING CONSTANTS & STATE ---
        CONF_THRESHOLD = 0.6
        AVG_THRESHOLD = 0.7
        MARGIN_THRESHOLD = 0.25
        HISTORY_SIZE = 5

        history_buffer = []  
        lock_state = "INACTIVE"
        current_surah = None
        if lock_surah is not None:
            lock_state = "ACTIVE"
            current_surah = lock_surah
        
        for t, chunk in enumerate(chunks):
            chunk_ngrams = set(NGramScorer.get_all_ngrams(chunk))
            scores = []
            
            # FAST PATH: use inverted index to get top-30 candidates
            # instead of scanning all 6,236 ayahs
            candidate_indices = self._get_candidate_indices(chunk_ngrams, top_k=30)

            # FALLBACK: if index returns no hits (very short chunk, noise, etc.),
            # fall back to scanning the full ayah list.
            # This should rarely happen for real recitation input.
            if not candidate_indices:
                candidate_indices = list(range(len(self.ayahs)))

            for idx in candidate_indices:
                s = self._score_chunk_against_ayah(chunk, chunk_ngrams, idx)
                if s > 0:
                    scores.append({'idx': idx, 'score': s})
            
            # Sort and take top K
            scores.sort(key=lambda x: x['score'], reverse=True)

            if t == 0 and start_surah is not None:
                for s_dict in scores:
                    if self.ayahs[s_dict['idx']]['surah'] == start_surah:
                        s_dict['score'] += 0.5
                scores.sort(key=lambda x: x['score'], reverse=True)

            raw_scores_per_chunk.append({s['idx']: s['score'] for s in scores[:30]})
            
            if lock_surah is None:
                # --- STEP 1, 2, 3 & 6: SURAH TRACKING ---
                if scores:
                    top_idx = scores[0]['idx']
                    top_score = scores[0]['score']
                    top_surah = self.ayahs[top_idx]['surah']
                    
                    # Find second best surah score for margin
                    second_best_surah_score = 0.0
                    for s_dict in scores:
                        s_surah = self.ayahs[s_dict['idx']]['surah']
                        if s_surah != top_surah:
                            second_best_surah_score = s_dict['score']
                            break
                            
                    margin = top_score - second_best_surah_score
                    
                    # Step 2: Maintain history buffer
                    if top_score >= CONF_THRESHOLD:
                        history_buffer.append({
                            'surah': top_surah, 
                            'score': top_score,
                            'margin': margin
                        })
                        if len(history_buffer) > HISTORY_SIZE:
                            history_buffer.pop(0)
                
                # Step 6: UNLOCK Mechanism
                if lock_state == "ACTIVE":
                    if not scores or scores[0]['score'] < (CONF_THRESHOLD - 0.2):
                        lock_state = "INACTIVE"
                    else:
                        # Check if mismatch increased significantly (other surah dominating)
                        recent_other_surah = [h for h in history_buffer[-3:] if h['surah'] != current_surah]
                        if len(recent_other_surah) >= 2 and all(h['score'] >= AVG_THRESHOLD for h in recent_other_surah):
                            lock_state = "INACTIVE"
                
                # Step 3: LOCK Mechanism
                if lock_state == "INACTIVE" and len(history_buffer) >= 3:
                    surah_counts = {}
                    for h in history_buffer:
                        s = h['surah']
                        if s not in surah_counts:
                            surah_counts[s] = {'count': 0, 'sum_score': 0.0, 'sum_margin': 0.0}
                        surah_counts[s]['count'] += 1
                        surah_counts[s]['sum_score'] += h['score']
                        surah_counts[s]['sum_margin'] += h['margin']
                        
                    for s, stats in surah_counts.items():
                        if stats['count'] >= 3:
                            avg_score = stats['sum_score'] / stats['count']
                            avg_margin = stats['sum_margin'] / stats['count']
                            if avg_score >= AVG_THRESHOLD and avg_margin >= MARGIN_THRESHOLD:
                                lock_state = "ACTIVE"
                                current_surah = s
                                break

            if lock_surah is not None:
                locked_scores = [
                    s for s in scores if self.ayahs[s['idx']]['surah'] == lock_surah
                ]
                if locked_scores:
                    scores = locked_scores
                else:
                    for s_dict in scores:
                        if self.ayahs[s_dict['idx']]['surah'] != lock_surah:
                            s_dict['score'] -= 1.5
                scores.sort(key=lambda x: x['score'], reverse=True)
            else:
                # Step 4: SOFT LOCK PENALTY/BOOST
                if lock_state == "ACTIVE" and current_surah is not None:
                    for s_dict in scores:
                        s_surah = self.ayahs[s_dict['idx']]['surah']
                        if s_surah == current_surah:
                            s_dict['score'] += 0.2  # Soft boost
                        else:
                            s_dict['score'] -= 0.2  # Soft penalty
                            
                    # Re-sort after boosting/penalizing
                    scores.sort(key=lambda x: x['score'], reverse=True)
            
            candidate_map = {s['idx']: s['score'] for s in scores[:K_CANDIDATES]}
            
            # Viterbi continuity safety: ALWAYS pad the state-space with logical next steps,
            # so the grid doesn't force a random jump if an intermediate chunk is completely unintelligible.
            if t > 0:
                for prev_idx in emissions[t-1]:
                    p_idx = prev_idx['idx']
                    if p_idx not in candidate_map:
                        candidate_map[p_idx] = 0.0
                    if p_idx + 1 < len(self.ayahs) and (p_idx + 1) not in candidate_map:
                        candidate_map[p_idx + 1] = 0.0
            elif not candidate_map:
                # Failsafe for very first chunk
                candidate_map[0] = 0.0

            emissions.append([{'idx': idx, 'score': sc} for idx, sc in candidate_map.items()])

        # Step 3: Dynamic Programming (Viterbi) Tracker Maps
        dp = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]
        
        # Init base cases (T=0)
        for cand in emissions[0]:
            dp[0][cand['idx']] = cand['score']
            backpointer[0][cand['idx']] = None
            
        # Step 4: Run DP Transitions
        for t in range(1, T):
            for cand in emissions[t]:
                curr_idx = cand['idx']
                curr_emission_score = cand['score']
                
                best_score = -float('inf')
                best_prev = None
                
                for prev_idx, prev_cum_score in dp[t-1].items():
                    # Step 5: SEQUENTIAL ALIGNMENT CONSTRAINT
                    transition_penalty = -5.0 # Severe penalty for random jump
                    
                    prev_surah = self.ayahs[prev_idx]['surah']
                    curr_surah = self.ayahs[curr_idx]['surah']
                    
                    if curr_surah != prev_surah:
                        # Allow transition ONLY if end of surah reached OR lock_state invalid
                        is_end_of_surah = False
                        if prev_idx + 1 < len(self.ayahs):
                            if self.ayahs[prev_idx + 1]['surah'] != prev_surah:
                                is_end_of_surah = True
                        else:
                            is_end_of_surah = True
                            
                        if is_end_of_surah and self.ayahs[curr_idx]['ayah'] == 1:
                            transition_penalty = -1.0 # Logical surah boundary transition
                        elif lock_state == "INACTIVE":
                            transition_penalty = -2.0 # More forgiving if no surah lock
                        else:
                            transition_penalty = -10.0 # Strictly penalize surah jumps when locked
                    else:
                        # Within the same surah sequence tracking
                        if curr_idx == prev_idx:
                            transition_penalty = 0.5 
                        elif curr_idx == prev_idx + 1:
                            transition_penalty = 1.0 
                        elif curr_idx == prev_idx + 2:
                            transition_penalty = -0.5
                        elif curr_idx == prev_idx + 3:
                            transition_penalty = -1.0
                        elif curr_idx < prev_idx:
                            transition_penalty = -5.0 # Prevent backward jumps
                        else:
                            transition_penalty = -5.0 # Prevent massive skipping
                    
                    candidate_score = prev_cum_score + transition_penalty + curr_emission_score
                    
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_prev = prev_idx
                
                dp[t][curr_idx] = best_score
                backpointer[t][curr_idx] = best_prev

            # Beam pruning: keep only the top-30 states by cumulative score.
            # This prevents the state space from growing unboundedly over long
            # sessions where every chunk pads in extra continuity states.
            # Pruning to B=30 has negligible accuracy impact because:
            #   - Low-scoring paths were already unlikely to be the optimal path
            #   - The backpointer for pruned states is already saved, so the
            #     optimal path up to t is preserved in the survivors
            _BEAM_SIZE = 30
            if len(dp[t]) > _BEAM_SIZE:
                _top_keys = sorted(dp[t], key=dp[t].__getitem__, reverse=True)[:_BEAM_SIZE]
                _keep = set(_top_keys)
                dp[t] = {k: v for k, v in dp[t].items() if k in _keep}
                backpointer[t] = {k: v for k, v in backpointer[t].items() if k in _keep}
                
        # Step 5: Backtrack optimal path
        best_end_idx = max(dp[T-1].keys(), key=lambda idx: dp[T-1][idx])
        
        path = []
        curr = best_end_idx
        for t in range(T-1, -1, -1):
            path.append(curr)
            curr = backpointer[t][curr]
            
        path.reverse()
        
        # Collapse continuous path indices
        unique_path_idxs = []
        for idx in path:
            if not unique_path_idxs or unique_path_idxs[-1] != idx:
                unique_path_idxs.append(idx)
                
        # Segment the original decoded text to the mapped ayahs
        orig_words = asr_text.split()
        ayah_to_words = {idx: [] for idx in unique_path_idxs}
        
        for t, idx in enumerate(path):
            start_idx = t * step
            if t < T - 1:
                end_idx = start_idx + step
            else:
                end_idx = len(orig_words) # Last chunk consumes the rest
            ayah_to_words[idx].extend(orig_words[start_idx:end_idx])
                
        # Formulate strict JSON output
        aligned_ayahs = []
        total_confidence = 0
        
        for idx in unique_path_idxs:
            ayah = self.ayahs[idx]
            
            # Reconstruct the decoded segment for this specific ayah
            seg_text = " ".join(ayah_to_words.get(idx, [])).strip()
            if not seg_text:
                continue
                
            # Compute independent confidence for this specific ayah match
            chunks_for_this_ayah = [t for t, p_idx in enumerate(path) if p_idx == idx]
            ayah_score_sum = 0
            for t in chunks_for_this_ayah:
                raw_score = raw_scores_per_chunk[t].get(idx, None)
                if raw_score is not None:
                    ayah_score_sum += raw_score
                    
            if chunks_for_this_ayah:
                avg_emission = ayah_score_sum / len(chunks_for_this_ayah)
                confidence = min(max(avg_emission * 100, 0), 100)
            else:
                confidence = 0.0
                
            total_confidence += confidence
            
            aligned_ayahs.append({
                "surah_ayah_id": ayah["id"],
                "text": seg_text,
                "confidence": round(confidence, 1)
            })
            
        avg_confidence = total_confidence / len(aligned_ayahs) if aligned_ayahs else 0

        status = "high" if avg_confidence > 75 else "low"
        
        # Step 7: FINAL OUTPUT
        return {
            "decoded_text": asr_text,
            "aligned_ayahs": aligned_ayahs,
            "current_surah": current_surah if current_surah is not None else "",
            "lock_state": lock_state,
            "alignment_confidence": status,
            "final_confidence": status,
            "alignment_path": path
        }

    def _segment_text(self, asr_text: str, chunk_size: int = 5, step: int = 3) -> List[str]:
        norm_asr = normalize_arabic(asr_text)
        words = norm_asr.split()
        if not words:
            return []

        segments = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + chunk_size]).strip()
            if chunk:
                segments.append(chunk)
        return segments

    def match_chunks(self, segments: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        candidates = []
        for chunk in segments:
            chunk_ngrams = set(NGramScorer.get_all_ngrams(chunk))
            scores = []

            candidate_indices = self._get_candidate_indices(chunk_ngrams, top_k=30)
            if not candidate_indices:
                candidate_indices = list(range(len(self.ayahs)))

            for idx in candidate_indices:
                s = self._score_chunk_against_ayah(chunk, chunk_ngrams, idx)
                if s > 0:
                    scores.append({
                        "idx": idx,
                        "score": s,
                        "surah": self.ayahs[idx]["surah"],
                        "ayah": self.ayahs[idx]["ayah"],
                        "id": self.ayahs[idx]["id"],
                    })
            scores.sort(key=lambda x: x["score"], reverse=True)
            candidates.append(scores[:top_k])
        return candidates

    def align_sequence(
        self,
        candidates: List[List[Dict[str, Any]]],
        asr_text: str,
        start_surah: Optional[int] = None,
        strip_invocations: bool = True,
        lock_surah: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        _ = candidates
        result = self.viterbi_align(
            asr_text,
            start_surah=start_surah,
            lock_surah=lock_surah,
            strip_invocations=strip_invocations,
        )
        return result.get("aligned_ayahs", [])

    def pipeline_from_text(
        self,
        asr_text: str,
        start_surah: Optional[int] = None,
        strip_invocations: bool = True,
        lock_surah: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = self.viterbi_align(
            asr_text,
            start_surah=start_surah,
            lock_surah=lock_surah,
            strip_invocations=strip_invocations,
        )
        return {"aligned_ayahs": result.get("aligned_ayahs", [])}
        
    def _empty_result(self, raw: str) -> Dict[str, Any]:
        return {
            "decoded_text": raw,
            "aligned_ayahs": [],
            "current_surah": "",
            "lock_state": "INACTIVE",
            "alignment_confidence": "low",
            "final_confidence": "low",
            "alignment_path": []
        }

if __name__ == "__main__":
    print("Initializing Hybrid Viterbi ASR Pipeline...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_path = os.path.join(current_dir, "fyp_model", "all_ayat.json")
    lm_path = os.path.join(current_dir, "quran_5gram.arpa")
    
    pipeline = HybridViterbiPipeline(ayah_json_path=reference_path, kenlm_path=lm_path)
    
    # Test case involving a sequence that hits surah 1:1 and tracks accurately to 1:2
    sample_text = "بسم الله الرحمن الحمد لله رب العالمين املك يوم الدين"
    
    print("\n--- Running Viterbi Alignment ---")
    result = pipeline.viterbi_align(sample_text)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
