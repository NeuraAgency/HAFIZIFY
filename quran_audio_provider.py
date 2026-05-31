import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


API_BASE = "https://api.quran.com/api/v4"
AUDIO_BASE = "https://verses.quran.foundation"
DEFAULT_RECITATION_ID = 7


class QuranAudioProvider:
    """Fetch and cache Quran.com ayah audio plus word timing segments."""

    def __init__(
        self,
        cache_dir: str = "audio_cache",
        recitation_id: Optional[int] = None,
    ):
        self.cache_dir = cache_dir
        self.recitation_id = int(
            recitation_id or os.environ.get("QURAN_RECITATION_ID") or DEFAULT_RECITATION_ID
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_ayah_audio(
        self,
        surah: int,
        ayah: int,
    ) -> Optional[Dict[str, Any]]:
        cache_key = f"{int(surah):03d}_{int(ayah):03d}"
        reciter_dir = os.path.join(self.cache_dir, str(self.recitation_id))
        os.makedirs(reciter_dir, exist_ok=True)

        audio_path = os.path.join(reciter_dir, f"{cache_key}.mp3")
        meta_path = os.path.join(reciter_dir, f"{cache_key}.json")

        if os.path.isfile(audio_path) and os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["audio_path"] = audio_path
            return meta

        ayah_key = f"{int(surah)}:{int(ayah)}"
        api_url = (
            f"{API_BASE}/recitations/{self.recitation_id}/by_ayah/"
            f"{urllib.parse.quote(ayah_key)}?fields=url,segments,duration,verse_key"
        )

        with urllib.request.urlopen(api_url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        audio_files = payload.get("audio_files") or []
        if not audio_files:
            return None

        meta = audio_files[0]
        audio_url = str(meta.get("url") or "")
        if audio_url and not audio_url.startswith(("http://", "https://")):
            audio_url = f"{AUDIO_BASE}/{audio_url.lstrip('/')}"
        if not audio_url:
            return None

        urllib.request.urlretrieve(audio_url, audio_path)
        meta["audio_url"] = audio_url
        meta["audio_path"] = audio_path

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return meta

    @staticmethod
    def segment_range_ms(
        segments: List[Any],
        start_word_index: Optional[int],
        end_word_index: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        if start_word_index is None or end_word_index is None:
            return None

        # Quran.com segments are commonly [word_index, start_ms, end_ms].
        # Treat correction word indexes as 0-based and accept 1-based segment indexes.
        wanted_start = int(start_word_index) + 1
        wanted_end = int(end_word_index) + 1
        selected = []

        for segment in segments or []:
            if not isinstance(segment, (list, tuple)) or len(segment) < 3:
                continue
            word_index, start_ms, end_ms = segment[:3]
            try:
                word_index = int(word_index)
                start_ms = int(start_ms)
                end_ms = int(end_ms)
            except Exception:
                continue
            if wanted_start <= word_index <= wanted_end:
                selected.append((start_ms, end_ms))

        if not selected:
            return None

        return min(s for s, _e in selected), max(e for _s, e in selected)
