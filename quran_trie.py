"""
Quran Prefix Trie — Constrained Decoding Support
Builds a token-level prefix trie from all_ayat.json using the
faster-whisper model's built-in tokenizer.
"""
import json
import os
import pickle
from typing import Optional

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quran_trie_cache.pkl")


class QuranTrie:
    def __init__(self):
        # Full trie: token_id -> {...} nested dict, "__end__" marks ayah boundary
        self.full_trie = {}
        # Per-surah tries: {surah_num: trie_dict}
        self.surah_tries = {}
        # Per-ayah hotwords: {(surah, ayah): [word1, word2, ...]}
        self.ayah_hotwords = {}
        # Per-surah hotwords: {surah: [word1, word2, ...]}
        self.surah_hotwords = {}

    def build(self, ayah_map: dict, tokenizer) -> None:
        """
        Build trie from ayah_map.
        ayah_map keys are (surah, ayah) tuples, values are Arabic text strings.
        tokenizer is the faster-whisper model's hf_tokenizer.
        """
        self.full_trie = {}
        self.surah_tries = {}
        self.ayah_hotwords = {}
        self.surah_hotwords = {}

        for (surah, ayah), text in ayah_map.items():
            if not text:
                continue

            # Build hotwords (word level)
            words = text.strip().split()
            self.ayah_hotwords[(surah, ayah)] = words
            self.surah_hotwords.setdefault(surah, set()).update(words)

            # Tokenize and build trie
            try:
                encoding = tokenizer.encode(text, add_special_tokens=False)
                token_ids = encoding.ids if hasattr(encoding, 'ids') else list(encoding)
            except Exception:
                continue

            # Insert into full trie
            node = self.full_trie
            for tid in token_ids:
                if tid not in node:
                    node[tid] = {}
                node = node[tid]
            node["__end__"] = True

            # Insert into per-surah trie
            if surah not in self.surah_tries:
                self.surah_tries[surah] = {}
            node = self.surah_tries[surah]
            for tid in token_ids:
                if tid not in node:
                    node[tid] = {}
                node = node[tid]
            node["__end__"] = True

        # Convert surah hotwords sets to sorted lists
        self.surah_hotwords = {s: sorted(list(words)) for s, words in self.surah_hotwords.items()}
        print(f"[QuranTrie] Built trie: {len(ayah_map)} ayahs, {len(self.surah_tries)} surahs")

    def get_hotwords(self, surah: Optional[int] = None, ayah: Optional[int] = None) -> list:
        """Return hotword list for constrained decoding.

        Deliberately returns [] (no hotwords) rather than the flattened
        full-Quran vocabulary when neither surah nor ayah is known yet (e.g.
        before surah-lock in a live session). faster-whisper's `hotwords`
        works by boosting decoder logits for the given tokens -- boosting
        essentially the entire vocabulary at once isn't a useful constraint,
        it's noise applied to nearly every token, and was measurably
        corrupting early-chunk decode quality (especially on smaller models
        like whisper-base, which has much less capacity to resist that bias
        than whisper-large-v3/turbo). See realtime_streamer.py's
        _decode_chunk() caller: ctx_surah/ctx_ayah are both None until
        surah-lock happens, which itself needs coherent decoded text to
        detect against -- so this used to create a self-reinforcing failure
        where garbage hotwords produced garbage decodes that could never
        accumulate enough signal to lock a surah and stop being garbage.
        """
        if surah is not None and ayah is not None:
            return self.ayah_hotwords.get((surah, ayah), [])
        if surah is not None:
            return self.surah_hotwords.get(surah, [])
        return []

    def get_initial_prompt(self, surah: Optional[int], ayah: Optional[int], ayah_map: dict) -> str:
        """Build initial_prompt string for Whisper decoder context."""
        if surah is None:
            return "بسم الله الرحمن الرحيم"
        # Get up to 2 preceding ayahs as context
        prompts = []
        start = max(1, (ayah or 1) - 2)
        end = (ayah or 1)
        for a in range(start, end):
            text = ayah_map.get((surah, a))
            if text:
                prompts.append(text)
        return " ".join(prompts) if prompts else "بسم الله الرحمن الرحيم"

    def save_cache(self) -> None:
        try:
            with open(_CACHE_FILE, "wb") as f:
                pickle.dump(self, f)
            print(f"[QuranTrie] Cache saved to {_CACHE_FILE}")
        except Exception as e:
            print(f"[QuranTrie] Cache save failed: {e}")

    @classmethod
    def load_cache(cls) -> Optional["QuranTrie"]:
        if not os.path.isfile(_CACHE_FILE):
            return None
        try:
            with open(_CACHE_FILE, "rb") as f:
                obj = pickle.load(f)
            print(f"[QuranTrie] Loaded cache from {_CACHE_FILE}")
            return obj
        except Exception as e:
            print(f"[QuranTrie] Cache load failed: {e}")
            return None