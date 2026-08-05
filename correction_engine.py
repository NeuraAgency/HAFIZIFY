"""
Quran Interactive Correction Engine
States: LISTENING → CORRECTING → VERIFYING → CONFIRMED/SKIPPED
"""
import asyncio
import difflib
import threading
import time
import edge_tts
import os
import tempfile
import pygame

try:
    from rapidfuzz import fuzz as _rfuzz_ce
    _CE_HAS_RAPIDFUZZ = True
except ImportError:
    _rfuzz_ce = None
    _CE_HAS_RAPIDFUZZ = False

from fyp_model.quran_guard import normalize_arabic
from quran_audio_provider import QuranAudioProvider

VOICE = "ar-SA-HamedNeural"


def _words_close(a: str, b: str) -> bool:
    """True if two normalized Arabic words are identical or a close ASR
    near-miss (e.g. a single letter the model consistently confuses) —
    same tolerance already used in realtime_streamer._token_close."""
    if not a or not b:
        return False
    if a == b:
        return True
    if _CE_HAS_RAPIDFUZZ:
        return _rfuzz_ce.ratio(a, b) >= 72.0
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.72


class CorrectionEngine:
    def __init__(self, on_state_change=None):
        self.state = "LISTENING"  # LISTENING, CORRECTING, VERIFYING
        self._min_trigger_confidence = 0.30  # ignore low-confidence/garbled chunks — not a real mistake
        self.current_ayah = None
        self.current_surah = None
        self.correction_attempts = 0
        self.on_state_change = on_state_change  # callback for UI update
        
        # Full session tracking
        self.error_history = []
        self.skipped_ayahs = []
        self.total_ok = 0
        self.total_errors = 0
        self.total_minor = 0

        self._pending_wrong_words = []
        self._pending_corrections = []
        # Which kind of correction is currently pending: "word" (consonant/
        # missing/extra word mistake — the original flow) or "harakaat"
        # (vowel-only mistake — masterplan.md §10). Determines how
        # _handle_verifying() checks whether the reciter actually fixed it:
        # word corrections check consonant-level text match; harakaat
        # corrections must check fresh diacritics, since a text match alone
        # would pass on the very first retry (the consonants were already
        # right — that's what made it a harakaat error and not a word one).
        self._correction_kind = "word"
        # Normalized (diacritic-stripped) reference words currently pending
        # a harakaat fix. Deliberately NOT touched by consume_pending_match()
        # (which only reads/writes _pending_wrong_words) — kept as an
        # independent set so the two verification paths can't interfere with
        # each other even though both may be "pending" in some form.
        self._pending_harakaat_ref_words: set = set()
        self._state_lock = threading.RLock()
        self._correction_id = 0
        
        # TTS setup
        self._audio_enabled = True
        try:
            pygame.mixer.init()
        except pygame.error as e:
            print(f"[CorrectionEngine] WARNING: Audio init failed ({e}). TTS disabled.")
            self._audio_enabled = False
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            try:
                pygame.mixer.init()
            except pygame.error:
                pass

        self._tts_lock = threading.Lock()
        self._quran_audio = QuranAudioProvider(
            cache_dir=os.path.join(os.path.dirname(__file__), "audio_cache")
        )

    # ─── TTS ───────────────────────────────────────────────
    def speak(self, text: str):
        if not self._audio_enabled:
            return

        with self._tts_lock:
            tmp_path = None
            try:
                async def _speak():
                    communicate = edge_tts.Communicate(text, VOICE)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        _tmp = f.name
                    await communicate.save(_tmp)
                    return _tmp

                import asyncio as _asyncio
                _loop = _asyncio.new_event_loop()
                try:
                    tmp_path = _loop.run_until_complete(_speak())
                finally:
                    _loop.close()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)

                pygame.mixer.music.unload()
                pygame.time.wait(100)
            except Exception as exc:
                print(f"[CorrectionEngine] TTS unavailable, continuing silently: {exc}")
            finally:
                if tmp_path:
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, FileNotFoundError):
                        pass

    # ─── Main Entry Point ──────────────────────────────────
    def process_verdict(
        self,
        verdict: str,
        raw_asr: str,
        correct_ayah_text: str,
        ayah_num: int,
        surah_num: int,
        wrong_words: list[str] | None = None,
        correction_spans: list[dict] | None = None,
        confidence: float = 1.0,
        harakaat_errors: list | None = None,
    ) -> dict:
        """
        Call this after every chunk is processed.
        Returns dict with action to take.

        harakaat_errors (masterplan.md §4.4 / Phase 5): optional, only ever
        populated by Combined Mode chunks. When None (standard mode, or a
        Combined Mode chunk with no vowel slips), behavior here is 100%
        unchanged from before this param existed.
        """
        with self._state_lock:
            self.current_ayah = ayah_num
            self.current_surah = surah_num

            if self.state == "LISTENING":
                return self._handle_listening(
                    verdict, raw_asr, correct_ayah_text, wrong_words, correction_spans,
                    confidence=confidence, harakaat_errors=harakaat_errors,
                )
            
            elif self.state == "VERIFYING":
                return self._handle_verifying(
                    verdict, raw_asr, correct_ayah_text, wrong_words,
                    confidence=confidence, harakaat_errors=harakaat_errors,
                )
            
            return {"action": "continue", "state": self.state}

    # ─── LISTENING state ───────────────────────────────────
    def _handle_listening(
        self,
        verdict,
        raw_asr,
        correct_ayah_text,
        wrong_words,
        correction_spans=None,
        confidence: float = 1.0,
        harakaat_errors: list | None = None,
    ) -> dict:
        if verdict == "ok":
            self.total_ok += 1
            if harakaat_errors:
                return self._handle_harakaat_error(harakaat_errors)
            self._notify("LISTENING")
            return {"action": "continue", "state": "LISTENING"}

        elif verdict == "minor":
            self.total_minor += 1
            if harakaat_errors:
                return self._handle_harakaat_error(harakaat_errors)
            self._notify("LISTENING")
            return {"action": "warn", "state": "LISTENING",
                    "message": "تحسين بسيط مطلوب"}

        elif verdict == "error":
            self.total_errors += 1
            # Ignore low-confidence/garbled chunks — not a real recitation mistake
            if confidence < self._min_trigger_confidence:
                self._notify("LISTENING")
                return {"action": "warn", "state": "LISTENING",
                        "message": f"خطأ محتمل"}

            self._pending_corrections = self._build_pending_corrections(
                wrong_words or [],
                correction_spans or [],
            )
            self.correction_attempts = 0
            self._correction_id += 1
            correction_id = self._correction_id
            self.state = "CORRECTING"
            self._correction_kind = "word"
            self._notify("CORRECTING")

            self._pending_wrong_words = [c["text"] for c in self._pending_corrections if c.get("text")]
            
            self.error_history.append({
                "ayah": self.current_ayah,
                "surah": self.current_surah,
                "wrong_text": raw_asr,
                "correct_text": correct_ayah_text,
                "wrong_words": list(self._pending_wrong_words),
                "attempts": 0,
                "resolved": False,
                "skipped": False,
            })
            
            threading.Thread(
                target=self._speak_words_and_verify,
                args=(list(self._pending_corrections), correction_id),
                daemon=True
            ).start()
            
            return {"action": "pause", "state": "CORRECTING",
                    "message": "خطأ — صحح الكلمة"}

    # ─── VERIFYING state ───────────────────────────────────
    def _handle_verifying(
        self, verdict, raw_asr, correct_ayah_text, wrong_words,
        confidence: float = 1.0, harakaat_errors: list | None = None,
    ) -> dict:
        # Ignore low-confidence/garbled chunks (background noise, ASR
        # hallucinations on silence, etc.) — they aren't a real correction
        # attempt, so don't count them as "still wrong" and don't replay TTS.
        if confidence < self._min_trigger_confidence:
            return {"action": "continue", "state": "VERIFYING",
                    "message": "بانتظار المحاولة"}

        if self._correction_kind == "harakaat":
            return self._handle_verifying_harakaat(verdict, harakaat_errors)

        if verdict in ("ok", "minor"):
            self.total_ok += 1
            if self.error_history:
                self.error_history[-1]["resolved"] = True
                self.error_history[-1]["attempts"] = self.correction_attempts

            self._correction_id += 1
            self.correction_attempts = 0
            self._pending_wrong_words = []
            self._pending_corrections = []
            self.state = "LISTENING"
            self._notify("LISTENING")

            threading.Thread(
                target=self.speak,
                args=("أحسنت",),
                daemon=True
            ).start()

            return {"action": "continue", "state": "LISTENING",
                    "message": "أحسنت — تابع"}

        # Still wrong — replay the correction and keep listening. The reciter
        # cannot move on to the next ayah until this one is said correctly.
        self.correction_attempts += 1
        if self.error_history:
            self.error_history[-1]["attempts"] = self.correction_attempts

        self._correction_id += 1
        correction_id = self._correction_id
        self.state = "CORRECTING"
        self._notify("CORRECTING")
        threading.Thread(
            target=self._speak_words_and_verify,
            args=(list(self._pending_corrections), correction_id),
            daemon=True
        ).start()

        return {"action": "retry", "state": "CORRECTING",
                "attempts": self.correction_attempts,
                "message": "حاول مرة أخرى"}

    def _handle_verifying_harakaat(self, verdict, harakaat_errors: list | None) -> dict:
        """Verification for a pending harakaat (vowel-only) correction.

        Deliberately does NOT reuse consume_pending_match()/the word-level
        text check above — that check runs after normalize_arabic(), which
        strips all diacritics, so it would report "fixed" the instant the
        reciter says the same consonants again regardless of whether the
        vowel is now right (it already was consonant-correct before this
        correction even started — that's what made it a harakaat error
        rather than a word error). Instead this compares the FRESH
        harakaat_errors computed for this new chunk (by
        harakaat_error_detector, re-run on every chunk regardless of qari
        state) against the specific reference word(s) this correction is
        for. A word only counts as fixed when it no longer shows up as a
        harakaat_error at all in this chunk.
        """
        current_norm_words = {
            normalize_arabic(h.get("reference") or h.get("predicted") or "")
            for h in (harakaat_errors or [])
            if (h.get("reference") or h.get("predicted"))
        }
        still_wrong = self._pending_harakaat_ref_words & current_norm_words

        if not still_wrong:
            self.total_ok += 1
            if self.error_history:
                self.error_history[-1]["resolved"] = True
                self.error_history[-1]["attempts"] = self.correction_attempts

            self._correction_id += 1
            self.correction_attempts = 0
            self._pending_wrong_words = []
            self._pending_corrections = []
            self._pending_harakaat_ref_words = set()
            self._correction_kind = "word"
            self.state = "LISTENING"
            self._notify("LISTENING")

            threading.Thread(target=self.speak, args=("أحسنت",), daemon=True).start()

            return {"action": "continue", "state": "LISTENING",
                    "message": "أحسنت — تابع"}

        # Still wrong — replay the correction for whichever flagged word(s)
        # remain wrong (a multi-word harakaat correction can be partially
        # fixed; only replay the ones still pending).
        self.correction_attempts += 1
        if self.error_history:
            self.error_history[-1]["attempts"] = self.correction_attempts

        self._pending_corrections = [
            c for c in self._pending_corrections
            if normalize_arabic(c.get("text", "")) in still_wrong
        ]

        self._correction_id += 1
        correction_id = self._correction_id
        self.state = "CORRECTING"
        self._notify("CORRECTING")
        threading.Thread(
            target=self._speak_words_and_verify,
            args=(list(self._pending_corrections), correction_id),
            daemon=True
        ).start()

        return {"action": "retry", "state": "CORRECTING",
                "attempts": self.correction_attempts,
                "message": "حاول التشكيل مرة أخرى"}

    # ─── Harakaat correction (masterplan.md §10) ───────────
    def _handle_harakaat_error(self, harakaat_errors: list) -> dict:
        """Full CORRECTING/VERIFYING interrupt for a vowel-only (harakaat)
        mistake — Combined Mode only. Uses the same audio-correction flow
        as a word/consonant error (mic pauses, the correct ayah audio plays,
        then it waits for a verified fix before resuming) rather than the
        old lighter HARAKAAT_HINT sub-state this replaces: per Hamza, a
        harakaat mistake should be corrected and verified, not just noted.

        Plays the real Quran recitation audio for the specific flagged
        word(s) (via _play_quran_correction, same as the word-error path) —
        that audio inherently carries the correct diacritics/pronunciation,
        which is a better teaching signal here than synthesized TTS would
        be. Uses each flagged word's ref_index (position in the REFERENCE
        ayah, not the predicted text — see WordAnnotation.ref_index in
        harakaat_error_detector.py) to select the right audio segment.
        """
        corrections = []
        for h in harakaat_errors:
            ref_word = h.get("reference") or h.get("predicted")
            if not ref_word:
                continue
            ref_idx = h.get("ref_index")
            corrections.append({
                "text": ref_word,
                "surah": self.current_surah,
                "ayah": self.current_ayah,
                "ref_word_start": ref_idx,
                "ref_word_end": ref_idx,
            })
        self.correction_attempts = 0
        self._correction_id += 1
        correction_id = self._correction_id
        self.state = "CORRECTING"
        self._correction_kind = "harakaat"
        self._notify("CORRECTING")

        self._pending_corrections = corrections
        self._pending_wrong_words = [c["text"] for c in corrections]
        self._pending_harakaat_ref_words = {normalize_arabic(c["text"]) for c in corrections}

        self.error_history.append({
            "ayah": self.current_ayah,
            "surah": self.current_surah,
            "wrong_text": "",
            "correct_text": "",
            "wrong_words": list(self._pending_wrong_words),
            "attempts": 0,
            "resolved": False,
            "skipped": False,
            "kind": "harakaat",
        })

        threading.Thread(
            target=self._speak_words_and_verify,
            args=(list(self._pending_corrections), correction_id),
            daemon=True
        ).start()

        return {"action": "pause", "state": "CORRECTING",
                "message": "خطأ في التشكيل — استمع وأعد",
                "harakaat_errors": harakaat_errors}

    def is_harakaat_correction(self) -> bool:
        """True while CORRECTING/VERIFYING is for a harakaat (vowel-only)
        mistake rather than a word/consonant one. realtime_streamer.py's
        process_chunk_combined() checks this during VERIFYING to decide
        whether to take its normal diacritic-blind consume_pending_match()
        shortcut (wrong for harakaat verification — see
        _handle_verifying_harakaat's docstring) or run the full word/
        harakaat scoring path so fresh diacritics are actually computed for
        this handler to check."""
        return self._correction_kind == "harakaat"

    # ─── Helpers ───────────────────────────────────────────
    def _speak_words_and_verify(self, corrections: list[dict], correction_id: int):
        try:
            spoken = set()
            for correction in corrections:
                word = str(correction.get("text") or "").strip()
                # Use one playback mechanism only. Quran-audio responses can
                # lack word timings and replay a whole ayah for every error;
                # Edge TTS always says only the intended reference word.
                word_key = normalize_arabic(word)
                if word and word_key and word_key not in spoken:
                    spoken.add(word_key)
                    self.speak(word)
        except Exception as exc:
            print(f"[CorrectionEngine] Correction audio failed, continuing silently: {exc}")

        # Brief pause after the correction finishes playing, before we start
        # listening again, so the reciter isn't cut off mid-breath.
        time.sleep(0.2)

        with self._state_lock:
            if correction_id != self._correction_id or self.state != "CORRECTING":
                return
            self.state = "VERIFYING"
            self._notify("VERIFYING")

    def _build_pending_corrections(self, wrong_words: list, correction_spans: list[dict]) -> list[dict]:
        corrections: list[dict] = []
        max_len = max(len(wrong_words), len(correction_spans)) if (wrong_words or correction_spans) else 0

        for idx in range(max_len):
            word = wrong_words[idx] if idx < len(wrong_words) else ""
            span = correction_spans[idx] if idx < len(correction_spans) and isinstance(correction_spans[idx], dict) else {}
            text = str(span.get("text") or word or "").strip()
            if not text:
                continue
            corrections.append({
                "text": text,
                "surah": self.current_surah,
                "ayah": self.current_ayah,
                "ref_word_start": span.get("ref_word_start"),
                "ref_word_end": span.get("ref_word_end"),
            })

        return corrections

    def _play_quran_correction(self, correction: dict) -> bool:
        if not self._audio_enabled:
            return False
        surah = correction.get("surah")
        ayah = correction.get("ayah")
        if not surah or not ayah:
            return False

        try:
            meta = self._quran_audio.get_ayah_audio(int(surah), int(ayah))
            if not meta:
                return False

            audio_path = meta.get("audio_path")
            if not audio_path or not os.path.isfile(audio_path):
                return False

            segment_range = self._quran_audio.segment_range_ms(
                meta.get("segments") or [],
                correction.get("ref_word_start"),
                correction.get("ref_word_end"),
            )

            with self._tts_lock:
                pygame.mixer.music.load(audio_path)
                if segment_range:
                    start_ms, end_ms = segment_range
                    duration_ms = max(250, end_ms - start_ms)
                    pygame.mixer.music.play(start=start_ms / 1000.0)
                    pygame.time.wait(duration_ms + 120)
                    pygame.mixer.music.stop()
                else:
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                pygame.mixer.music.unload()
            return True
        except Exception as exc:
            print(f"[CorrectionEngine] Quran audio fallback to TTS: {exc}")
            return False

    def _notify(self, state: str):
        if self.on_state_change:
            self.on_state_change(state)

    def get_pending_corrections(self) -> list[str]:
        with self._state_lock:
            return list(self._pending_wrong_words)

    def consume_pending_match(self, recited_text: str) -> bool:
        """Check the just-recited text against the specific phrase(s) that
        were flagged wrong, using the same near-miss tolerance the normal
        listening pipeline already applies — an exact-only comparison would
        reject a correct retry any time the ASR mishears the same letter
        again, which happens often (e.g. ص heard as س)."""
        with self._state_lock:
            norm_recited = normalize_arabic(recited_text)
            if not norm_recited:
                return False

            recited_words = norm_recited.split()
            new_pending: list[str] = []
            matched = False

            def _sequence_close(a_words: list[str], b_words: list[str]) -> bool:
                if not a_words or len(a_words) != len(b_words):
                    return False
                return all(_words_close(a, b) for a, b in zip(a_words, b_words))

            for phrase in self._pending_wrong_words:
                norm_phrase = normalize_arabic(phrase)
                phrase_words = norm_phrase.split()

                phrase_in_recited = any(
                    _sequence_close(phrase_words, recited_words[i : i + len(phrase_words)])
                    for i in range(0, max(0, len(recited_words) - len(phrase_words)) + 1)
                ) if phrase_words else False

                if _sequence_close(phrase_words, recited_words) or phrase_in_recited:
                    matched = True
                    continue

                remaining_words: list[str] = []
                for word in phrase.split():
                    norm_word = normalize_arabic(word)
                    if any(_words_close(norm_word, rw) for rw in recited_words):
                        matched = True
                        continue
                    remaining_words.append(word)

                if remaining_words:
                    new_pending.append(" ".join(remaining_words))

            if matched:
                self._pending_wrong_words = new_pending
                remaining_norm = {normalize_arabic(text) for text in new_pending}
                self._pending_corrections = [
                    correction
                    for correction in self._pending_corrections
                    if normalize_arabic(correction.get("text", "")) in remaining_norm
                ]

            return matched and not self._pending_wrong_words

    # ─── Session Summary ───────────────────────────────────
    def get_session_summary(self) -> dict:
        return {
            "total_ok": self.total_ok,
            "total_minor": self.total_minor,
            "total_errors": self.total_errors,
            "error_history": self.error_history,
            "skipped_ayahs": self.skipped_ayahs,
            "accuracy": round(self.total_ok / max(1, self.total_ok + self.total_errors) * 100, 1) if (self.total_ok + self.total_errors) > 0 else 0.0
        }

    def reset(self):
        self.__init__(self.on_state_change)
