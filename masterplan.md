# Hafizify — Combined Model Mode: Master Implementation Plan

> Status: PLANNING — no code written yet.
> Written by Claude (chat) after reviewing `system.md`, `groq_transcriber.py`,
> `tajweed_model.py`, `fyp_model/quran_guard.py`, `error_analysis.py`.
> Goal of this doc: survive context/token loss. Any future session (Claude or
> Cline/DeepSeek) should be able to pick this up cold and keep going.

## 0. Non-negotiable constraints (from Hamza, 2026-08-01)

- **Do not touch the existing pipeline.** Standard mode (single offline
  `whisper-base-quran-lora-ct2`) must keep working exactly as documented in
  `system.md` with zero behavior change.
- **Do not touch VAD settings.** `session_manager.py`'s `_VAD_*` constants and
  `RecitationSession` chunking logic are off-limits. We *reuse* the chunks it
  already produces — we don't re-tune or re-implement VAD.
- Everything new is **opt-in**, gated behind one new UI toggle. If the user
  doesn't select "Combined Model", nothing about their session changes.
- New work lands in **new files** wherever possible. Existing files get small,
  additive, guarded edits only (new optional params with safe defaults, new
  `if combined_mode:` branches) — never rewrites of existing logic.

## 1. What "Combined Model" mode actually is

A second, optional ASR path for the Live Recitation tab that runs **two
models per VAD chunk** instead of one:

1. **Groq cloud Whisper** (`whisper-large-v3`, via the existing
   `groq_transcriber.py` — already has safe env-based API key handling, reuse
   it as-is, do not duplicate the client).
2. **Local diacritized model** — `whisper-l-v3-turbo-quran-lora-dataset-mix/`
   (already sitting on disk in this project, confirmed present, not currently
   wired into `app.py`). This is a heavier HF Transformers model (not the
   lightweight CT2 CPU model) and is the one that actually outputs harakaat
   (tashkeel/diacritics) — the default CT2 model's output is likely
   undiacritized like Groq's, which is *why* this second model is needed at
   all.

The two outputs get merged word-by-word: Groq supplies the more reliable
consonant backbone, the local model supplies the diacritics, via
character-position vowel injection (this logic already exists — as a rough
draft — in the Colab script you pasted earlier; it needs cleanup, not a
rewrite from scratch).

**Trade-off to surface in the UI, not hide:** this mode needs internet (Groq)
and a heavier local model, which sits outside the documented
"CPU-only / IoT, no GPU" design goal in `system.md §13` for the *default*
pipeline. That's fine — it's explicitly a secondary, opt-in mode, and
Hamza will run it on an RTX 2060 Super during development, later moving to a
GPU server for deployment — so real-time performance is not a blocker the
way it would be on CPU. Still worth a short UI note ("Combined Mode:
requires internet + GPU, more accurate diacritics") so it's clear this path
has different hardware requirements than the default offline mode, for the
FYP writeup's sake.

**GPU note for Phase 1:** load `whisper-l-v3-turbo-quran-lora-dataset-mix`
with `device_map="auto"` / `torch_dtype=torch.float16` when CUDA is
available (as in the original Colab script) — the RTX 2060 Super (8GB VRAM)
handles the turbo model fine; just watch VRAM headroom if it ever runs
alongside the CT2 model in the same process during testing.

## 2. New capability this unlocks: harakaat-level error detection

Today, `fyp_model/quran_guard.py::normalize_arabic()` **strips all diacritics
before any comparison**. That means the entire existing correction/guard/
Viterbi/Qari stack is structurally blind to vowel-level mistakes — a reciter
saying `اهدِنا` with the wrong short vowel on a letter, or dropping a shadda,
currently cannot be detected at all, only full word substitutions
(`correction_engine.py`'s error gate) and consonant-level ayah mismatches are.

Combined Mode is the first pipeline that produces diacritized ASR output, so
it's the first point where a **harakaat error detector** becomes possible.
This is being built as a **separate, new module**, not bolted into
`quran_guard.py`, and it only ever runs when Combined Mode is selected.

**Phase 0 result (confirmed):** `fyp_model/all_ayat.json` — 6,235 ayahs,
6,215 of them (99.7%) carry full Uthmani tashkeel in `text`, e.g.
`1_3 -> "الرَّحْمَٰنِ الرَّحِيمِ"`. We have a diacritized reference for free,
no separate corpus needed. Two things to handle in the detector:
- `1_1` (the Basmala) has **no key at all** in this file (consistent with
  `surah_detector.py` deliberately excluding it from its index). The
  harakaat detector must skip comparison gracefully when the expected ayah
  is 1:1 rather than erroring.
- ~20 ayahs (0.3%) have no diacritics in this file. Detector should treat a
  bare reference (no diacritics found) as "skip harakaat check" for that
  ayah rather than flagging every word as a harakaat error.

## 3. New files (all additive, zero risk to existing code)

| File | Purpose |
|---|---|
| `hybrid_diacritic_pipeline.py` | Runs Groq + local turbo model on one audio chunk, merges into diacritized text. Cleaned-up version of the Colab logic. |
| `harakaat_error_detector.py` | Word-aligns combined diacritized output against the expected ayah's diacritized reference text; classifies each word as `ok` / `harakaat_error` (vowel-only mismatch) / `makhraj_error` (consonant mismatch, already covered elsewhere but flagged here too for consistency). |
| `masterplan.md` | This file. |

### 3.1 `hybrid_diacritic_pipeline.py` — carried over from your Colab script, with fixes

Reuse: `smart_normalize_word`, `inject_vowels_by_character_position`,
`run_hybrid_combination_logic` (the core merge algorithm — it works, 92%
smart-accuracy proved that).

Fix while porting (small, contained, doesn't change intended behavior):
- **RULE 1 dead branch.** In your script:
  ```python
  patched_word = "اهْ" + patched_word[2:] if patched_word.startswith("اِهْ") else "اهْ" + patched_word[2:] if groq_word.startswith("اهد") else "ا" + patched_word[2:]
  ```
  Both truthy branches return the same value — the first `if` never matters.
  Collapse to `"اهْ" + patched_word[2:] if groq_word.startswith("اهد") else "ا" + patched_word[2:]`.
- **Hardcoded `"مالك"` global override.** Currently force-injects one fixed
  diacritization regardless of context. Since the live app *always* knows the
  expected ayah (via `SurahLockManager` / `current_ayah`), replace this
  special case with a lookup against the actual reference diacritics for that
  word at that position instead of a hardcoded string — makes it correct for
  every ayah containing `مالك`-family words, not just Al-Fatiha.
- **Remove the hardcoded `GROQ_API_KEY` entirely** — this module calls the
  existing `groq_transcriber.get_groq_transcriber()` singleton instead of
  instantiating its own `Groq()` client. One key source, already `.env`-based.
- Local model loading: **lazy-load only when Combined Mode is first used in a
  session**, not at app startup — keeps standard-mode startup time/memory
  exactly as it is today for anyone who never touches this toggle.

### 3.2 `harakaat_error_detector.py`

```
detect_harakaat_errors(predicted_diacritized_text, reference_diacritized_text) -> list[WordAnnotation]
```
Per word (reusing the word-alignment pattern already in `error_analysis.py`'s
`align_words`, which is a good fit):
- strip diacritics from both → if the *consonant skeleton* differs, it's a
  `makhraj_error` (existing guard machinery already flags this at the ayah
  level; we just tag it consistently here too)
- if consonant skeleton matches but diacritics differ, it's a
  `harakaat_error` — new category
- else `ok`

Output is a plain list of small dicts (word, type, expected, got) — easy to
hand to both the display layer and Qari Mode without either needing to know
how the detection works internally.

## 4. Integration points (additive edits only)

### 4.1 `session_manager.py` — VAD chunks (READ-ONLY reuse, no edits planned)
`RecitationSession` already emits `ChunkResult`-bound audio chunks with
pre/post context handling. Combined Mode consumes the **same emitted chunk
audio** the standard pipeline gets — we do not add a second VAD pass or touch
`_VAD_*` constants. The only touch to this file, if unavoidable, is adding
two **optional** fields to the `ChunkResult` dataclass:
```python
harakaat_errors: list | None = None
harakaat_error_count: int = 0
```
Defaults mean standard-mode sessions serialize to `results.json` exactly as
before (`null` / `0`), no schema break.

### 4.2 `realtime_streamer.py` — new sibling function, not a rewrite
Do **not** modify `process_chunk()`. Add a new function
`process_chunk_combined()` alongside it that:
1. takes the same chunk audio `process_chunk()` receives
2. calls `hybrid_diacritic_pipeline` instead of the single faster-whisper decode
3. runs the result through the *existing* guard pipeline (`quran_guard.py`)
   for ayah matching/correction, unchanged
4. additionally runs `harakaat_error_detector` and attaches the result

The session worker thread picks `process_chunk()` or
`process_chunk_combined()` based on the new UI toggle's value — a single
dispatch `if`, not a rewrite of the worker loop.

### 4.3 `app.py` — one new UI control
Add a single control to the Live Recitation tab (radio or checkbox):
`"ASR Engine: Standard (offline, fast) | Combined (Groq + Local, harakaat-aware, needs internet)"`.
Default: **Standard** — an existing user who never touches this control gets
byte-identical behavior to today.

### 4.4 `correction_engine.py` (Qari Mode) — additive param, new soft state
`process_verdict()` gains one new optional kwarg:
```python
harakaat_errors: list | None = None
```
When `None` (standard mode, or combined mode with no vowel errors this
chunk), behavior is **100% unchanged** — the existing consecutive-error gate,
`CORRECTING` state, and TTS flow for word-level (makhraj) errors stay exactly
as documented in `system.md §10`.

When harakaat errors are present, add a lighter-weight sibling reaction —
**does not use the strict "2 consecutive errors" gate**, since a single vowel
slip is real and worth a gentle nudge, but shouldn't trigger a full
`CORRECTING` interrupt the way a wrong word does. Suggest a new minimal
sub-state (name it something like `HARAKAAT_HINT`) that plays a short,
distinct audio cue (different from the full correction TTS) and immediately
returns to `LISTENING` — no multi-attempt verification loop, since a diacritic
slip is a review note, not a stop-and-retry event.

### 4.5 `live_display_formatter.py` — new color, additive branch
Add a third highlight color (e.g. amber) for `harakaat_error` words, distinct
from the existing red (wrong word) / green (correct). New rendering branch
only fires when `harakaat_errors` is non-empty on a `ChunkResult` — standard
mode chunks never populate that field, so existing HTML output is unaffected.

## 5. Explicit "do not touch" list

- `session_manager.py` `_VAD_*` constants, `_CTX_PRE_S` / `_CTX_POST_S`
- `fyp_model/quran_guard.py` existing `_PHRASE_FIXES` / `_TOKEN_FIXES` tables
  and `normalize_arabic()` (harakaat detector is a separate comparison path,
  not a replacement for the existing text-normalized guard matching)
- `hybrid_pipeline.py` (the Viterbi/KenLM alignment pipeline — unrelated,
  different "hybrid" than this plan's hybrid, don't confuse the two)
- `realtime_streamer.py::process_chunk()` body
- `correction_engine.py`'s existing consecutive-error gate logic for
  makhraj/word errors
- `MODEL_CHOICES` / single-offline-model startup path in `app.py`

## 6. Build order (phases — each independently testable)

- **Phase 0 — Audit (do first, cheap):** confirm whether `all_ayat.json`
  `text` fields carry tashkeel. Spot-check via a short Python one-liner on
  one ayah (e.g. `1_1`), not a full file read.
- **Phase 1:** Build `hybrid_diacritic_pipeline.py` standalone. Test via a
  small CLI script against a sample WAV, compare to a known ground truth
  (same pattern as your Colab script's WER/CER printout).
- **Phase 2:** Build `harakaat_error_detector.py` standalone with a handful of
  hand-written correct/incorrect diacritic pairs as unit tests.
- **Phase 3:** Wire `process_chunk_combined()` into `realtime_streamer.py` as
  a new sibling function + dispatch.
- **Phase 4:** Add the UI toggle in `app.py`.
- **Phase 5:** Extend `correction_engine.py` with the optional
  `harakaat_errors` param and `HARAKAAT_HINT` sub-state.
- **Phase 6:** Extend `live_display_formatter.py` with the amber highlight
  branch.
- **Phase 7:** End-to-end test: same recording, run once in Standard mode and
  once in Combined mode, diff the two `results.json` outputs to confirm
  Standard mode is byte-for-byte unaffected.
- **Phase 8:** Update `system.md` and `Hafizify.md` — add Combined Mode to
  the architecture doc and bug/change log table, same style as the existing
  v1.1.0 → v1.2.0 log.

## 7. Known risk / things to watch

- Groq call is per-chunk over the network — latency budget needs checking
  against the VAD chunk cadence (chunks arrive faster than a round trip may
  return on a slow connection). May need a short async queue rather than a
  blocking call inside the worker thread.
- The turbo local model is much heavier than the CT2 model `system.md`
  specifically chose for CPU-only real-time use. Primary dev target is the
  RTX 2060 Super, later a GPU server, so real-time isn't a concern there —
  but the loader should still auto-detect CUDA and gracefully fall back to
  CPU (`torch_dtype=torch.float16 if torch.cuda.is_available() else
  torch.float32`, `device_map="auto"`) so the app doesn't hard-crash on a
  machine without a GPU; it'll just be slow there, which is an acceptable
  degrade for an opt-in mode.
- Groq API key: the one in your pasted Colab script is exposed in this chat
  now — rotate it in the Groq console. The `.env`-based key in
  `groq_transcriber.py` is the one to keep using; don't reintroduce a
  hardcoded key anywhere in the new files.

## 8. Changelog — deviations / fixes made after this plan was written

**2026-08-02 — Phase 5 completed + a Phase 3/4 gap closed (Claude, chat).**

On resuming work at Phase 5, found that `correction_engine.py` already had
the `harakaat_errors` kwarg wired into `process_verdict()` /
`_handle_listening()` (per §4.4) from an earlier session, including two
call sites invoking `self._handle_harakaat_hint(harakaat_errors)` — but
that method was never actually defined. This was a live crash bug: the
first Combined Mode chunk with a harakaat-only mistake during a Qari
session would raise `AttributeError` and kill the worker thread.

Fixes made:
1. **`correction_engine.py`** — added the missing `_handle_harakaat_hint()`
   method. No consecutive-error gate; plays a short distinct spoken cue
   (`"انتبه للتشكيل"`, backgrounded via the existing `speak()`/edge_tts
   path so it doesn't block); never touches `self.state` (stays
   `LISTENING`, no `CORRECTING`/`VERIFYING` loop); returns
   `{"action": "hint", ...}`. Nothing else in the file changed.
2. **A second gap, not in the original phase list:** `process_chunk_combined()`
   never called `self.correction_engine.process_verdict()` at all, and
   `app.py`'s worker dispatch didn't even pass `qari_mode` through on the
   `asr_engine == "combined"` path — so even with `_handle_harakaat_hint`
   fixed, nothing could reach it in Combined Mode. Closed this by:
   - `realtime_streamer.py`: added `qari_mode: bool = False` param to
     `process_chunk_combined()`; added a new module-level helper
     `_apply_qari_word_scoring()` (a fresh copy of process_chunk()'s
     qari_mode word-scoring logic, not a refactor — `process_chunk()`'s
     body is still untouched, per §5's "do not touch" list) plus the same
     `VERIFYING` / `consume_pending_match` branch process_chunk() has;
     at the end of the function, when `qari_mode` is True, calls
     `self.correction_engine.process_verdict(...)` with
     `harakaat_errors=guard_result.get("harakaat_errors")`, mirroring
     process_chunk()'s qari_mode dispatch (`pause`/`continue`/`skip` →
     `_vad_paused`).
   - `app.py`: the worker's `asr_engine == "combined"` branch now passes
     `qari_mode=qari_mode` into `process_chunk_combined()`.
3. Confirmed §4.1 (`ChunkResult.harakaat_errors` / `harakaat_error_count`
   fields, `register_chunk_result` populating them) was already done —
   no change needed there.

Not done, still open: the `strict_correction`-driven `lookahead`/
`window_back` narrowing that `process_chunk()` applies to `guard_inference`
while `CORRECTING`/`VERIFYING` was intentionally NOT ported into
`process_chunk_combined()` — the queue worker already drops chunks queued
while state is `CORRECTING` for both engines, and `VERIFYING` is handled
explicitly via `consume_pending_match`, so the narrowing is a minor
real-time-decode precision tweak, not a correctness requirement. Revisit
if Combined + Qari Mode testing (Phase 7) shows it's needed.

**2026-08-02 (later same day) — Phase 1/2 correctness bug found + fixed
(Claude, chat), reported by Hamza with a live bad output.**

Hamza noticed Combined mode's output for a Fatiha chunk ("Groq: الحمد لله
رب العالمين" / "Local: من حبل اللّه ربّ العالمين" → "Combined:
من للهِ ربّ العالمين") wasn't taking words from Groq + harakaat from local
as designed — it was dropping/mangling Groq words entirely.

Root cause in `hybrid_diacritic_pipeline.py::run_hybrid_combination_logic()`:
the per-word loop only had a real alignment (`local_word_mapping`, built
from difflib's `"equal"` opcodes) for words difflib confidently matched
between Groq and local. For every other word, it fell back to
`local_words[idx]` — pairing by *raw index position*, not by actual
alignment. Any time Groq/local word counts or wording diverged for a
stretch (routine, since they're two different models), this silently
paired unrelated words: either overwriting a Groq word wholesale with an
unrelated local word (skeleton-length mismatch → fallback returns the
local word verbatim), or splicing one word's consonants with a different
word's diacritics (skeleton-length coincidentally equal).

Agreed fix direction (Hamza): Groq's words are the backbone and are never
dropped/overwritten. Harakaat is only pasted on when there's a confident
local-model match for that specific word; on a mismatch, leave the Groq
word bare rather than guess. Downstream, a bare (non-diacritized) word
should only ever be checked for word/makhraj correctness — never flagged
as a harakaat mistake just because it has no vowels to compare; a word
that does carry harakaat gets checked for both.

Fixes made:
1. **`hybrid_diacritic_pipeline.py`** — removed the `elif idx < len(local_words):`
   index-guess fallback in `run_hybrid_combination_logic()` entirely. A
   Groq word now only gets `inject_vowels_by_character_position()` applied
   if `idx in local_word_mapping` (a genuine difflib `"equal"`-block
   match); otherwise it passes through unchanged. No behavior change to
   `inject_vowels_by_character_position()` itself or to the already-correct
   equal-block path.
2. **`harakaat_error_detector.py`** — added `harakaat_checked: bool = True`
   to `WordAnnotation`. In `detect_harakaat_errors()`'s `"equal"`-tag
   branch, a predicted word with no diacritics at all now gets
   `status="ok", harakaat_checked=False` (word/skeleton already matched to
   get into this branch — no harakaat claim is made either way) instead of
   being counted as `harakaat_error` just because it differs textually
   from a diacritized reference word. Words that do carry diacritics keep
   the existing ok/harakaat_error comparison, unchanged.

Not done yet: downstream consumers of `WordAnnotation` (display layer /
`live_display_formatter.py`'s amber-highlight branch, §4.5 — not yet
built) should read `harakaat_checked` and NOT amber-highlight a word where
it's `False`, since that's an intentional "unknown", not a flagged error.
Flag this for whoever builds Phase 6.

**2026-08-02 (later same day, second round) — fuzzy same-slot matching
added to `hybrid_diacritic_pipeline.py` (Claude, chat), per Hamza.**

The fix above was too strict in one specific, real case: local model gets
the *consonants* wrong (hallucinated letter) but the *vowel pattern*
right — e.g. Groq `اهدنا`, local `إِهْتِنَا` (one wrong letter, ت
for د, but the exact same vowel sequence in the same positions). Since
difflib's `"equal"` opcode requires the whole normalized word to match,
this pair landed in a `"replace"` block and, after the first fix, was left
bare — throwing away a genuinely correct vowel pattern along with the one
wrong letter.

Added a second, narrower matching tier inside `run_hybrid_combination_logic()`:
within a `"replace"` opcode block where Groq and local have the *same word
count* (a plausible 1:1 substitution, not an insertion/deletion drift), a
pair is now also treated as a confident match — and gets vowel injection
— if `_is_close_enough()` says so: same normalized length (required
anyway for character-position injection) and a small edit distance (1 for
words ≤ 6 letters, 2 for longer; words under 3 letters require an exact
match, since a 1-letter edit on a very short word is usually just a
different word, not ASR noise). This is safe by construction —
`inject_vowels_by_character_position()` always keeps Groq's letters as the
backbone and only ever borrows the *diacritic* pattern from the local
word, so a wrong local letter can never leak into the output; fuzzy-
matching only changes whether we use the vowels or leave the word bare.
No change to the exact-match (`"equal"`) path, which still takes priority.

**2026-08-02 (later same day, third round) — two false-positive sources
fixed in `harakaat_error_detector.py` (Claude, chat), per Hamza, from a
live log showing `harakaat_errors=2` on a chunk that was actually correct.**

Both mismatches in the flagged chunk (`الرَّحْمَنِ الرَّحِيمَ` predicted vs
`الرَّحْمَٰنِ الرَّحِيمِ` reference) were false positives, not real recitation
mistakes:

1. **Dagger-alif (ٰ).** The reference carries it (`مَٰ`), but neither
   Groq nor the local Whisper model can ever produce that character — it's
   a Quranic-orthography convention, not something ASR output uses. Any
   ayah containing one would falsely flag forever, regardless of correct
   recitation.
2. **Last word of the ayah's case-ending vowel.** `الرَّحِيمَ` (predicted,
   fatha) vs `الرَّحِيمِ` (reference, kasra) — could be a real slip, but
   final-word case endings legitimately change under pausal recitation
   (waqf) and are the least ASR-reliable vowel position generally, so
   flagging them outright is unreliable.

Hamza's direction: fix #1 outright (never a real signal). For #2, be
narrow — only skip the FINAL diacritic mark of the LAST word of the ayah,
nothing more; every other diacritic in that same word, and every other
word, still gets checked normally.

Fixes made, both scoped to the `"equal"`-tag comparison in
`detect_harakaat_errors()`:
- Added `_strip_dagger_alif()` — strips `\u0670` from the reference word
  before the equality check (predicted text never has it, so this is a
  one-sided but symmetric-safe strip).
- Added `_strip_last_diacritic_cluster()` — reuses
  `decompose_diacritic_clusters()` to rebuild a word with the diacritic
  cluster on its FINAL base character removed, keeping every other
  letter's diacritics intact. Applied to both predicted and reference word
  only when `ref_idx == len(ref_words) - 1` (this word is literally the
  last word of the ayah's reference text).
- `WordAnnotation` gained `final_vowel_skipped: bool = False`, set True on
  annotations for that last word, so downstream consumers (display layer,
  eval/logging) can see when a word's ending wasn't fully checked rather
  than silently trusting a pass.
- No change to non-last words, no change to the `makhraj_error` /
  `replace`/`insert`/`delete` branches, no change to
  `hybrid_diacritic_pipeline.py`.

**2026-08-02 (later same day, fourth round) — `api/` (mobile FastAPI
layer) wired up for Combined Mode (Claude, chat), per Hamza.**

Audit finding: `api/main.py` and `api/formatters.py` predated Combined
Mode entirely — `StartSessionRequest` had no `asr_engine` field, the WS
stream and `/transcribe` only ever called the standard decode path, and
`chunk_result_to_json()` never read `ChunkResult.harakaat_errors` /
`harakaat_error_count` even though `session_manager.py` had carried those
fields since §4.1. A mobile client had no way to request Combined Mode or
see harakaat data even if it could.

Fixes made:
1. **`realtime_streamer.py`** — added `self._last_qari_action` (init to
   `None` in `__init__`), set at the end of `process_chunk_combined()`'s
   qari_mode branch to whatever `correction_engine.process_verdict()`
   returned. Needed because the API's existing qari relay only inferred a
   `"pause"` action from `get_pending_corrections()`, which stays empty for
   a `"hint"` (harakaat-only) action — there was no way to see that action
   fired at all without capturing the real returned dict. `process_chunk()`
   (standard) itself is untouched.
2. **`api/main.py`**:
   - `StartSessionRequest` gained `asr_engine: str = "standard"` (falls
     back to `"standard"` on any other value, same tolerance as app.py's
     `_parse_asr_engine()`). `start_session()` stores it in a new
     `_active_session_asr_engine` global and, on `"combined"`,
     synchronously calls `hybrid_diacritic_pipeline.preload_local_pipeline()`
     before returning — mirrors `start_live_session()`'s preload in app.py
     so the first WS chunk isn't slowed by a cold model load. Also resets
     `correction_engine` + `_vad_paused` + `_last_qari_action` when
     `qari_mode` is on, matching app.py's Qari Mode session-start reset.
   - Added `_process_one_chunk()` — single dispatch point (standard vs
     `process_chunk_combined`) used by both the WS loop and the flush loop
     in `_finalize_active_session()`, so they can't drift out of sync on
     which engine a session is actually using.
   - WS stream: when `asr_engine == "combined"` and `qari_mode` is on, the
     qari-action relay now reads `rt_streamer._last_qari_action` (covers
     both `"pause"` and the new `"hint"`) instead of only synthesizing a
     `"pause"` guess from pending corrections; standard mode keeps the
     original pending-corrections-based relay unchanged.
   - `/transcribe` gained `asr_engine: str = Form("standard")`. On
     `"combined"`, decodes via `hybrid_diacritic_pipeline.run_combined_transcription()`
     instead of `rt_streamer._decode_raw()`, then runs
     `harakaat_error_detector.detect_harakaat_errors()` against the matched
     ayah and includes `harakaat_errors` / `harakaat_error_count` in the
     response (`[]` / `0` on standard calls, never omitted).
3. **`api/formatters.py`** — `chunk_result_to_json()` now includes
   `harakaat_errors` (defaulting `[]`, never `null`) and
   `harakaat_error_count` read straight off `ChunkResult`.
   `qari_action_to_json()` now also passes through `harakaat_errors` from
   the correction-engine action dict, alongside the existing
   `wrong_words`.
4. **`api/README.md`** — updated to document `asr_engine` on both
   `/transcribe` and `/session/start`, the new response/message fields, the
   `"hint"` qari-action, and a note in the mobile-pipeline walkthrough.

Not done: no mobile-side (Expo) code exists yet to actually consume any of
this — this round only closes the server-side gap so the API can serve a
Combined Mode client whenever that's built.

**2026-08-02 (later same day, fifth round) — Docker build: removed kenlm
entirely instead of fixing its build (Claude, chat), per Hamza's "do we
even use kenlm, it never loads" catch.**

Previous two rounds (same day) had been chasing a `kenlm` from-source build
failure (`Python.h: No such file or directory`) by adding
`build-essential`/`cmake`/`python3-dev`/`python3.10-dev`/`libeigen3-dev` to
the `Dockerfile`. Hamza correctly pushed back: is kenlm even reachable code
in this image? Traced it — no. Its only two call sites are
`hybrid_pipeline.py`'s `HybridViterbiPipeline` (used only by
`realtime_streamer.py`'s `decode_full_session()`, which `api/main.py`
never calls — and `hybrid_pipeline.py` isn't even in the `Dockerfile`'s
`COPY` list) and `fyp_model/beam_decoder.py` (only used for the `wav2vec2`
model type, which nothing in `api/`'s model registry/defaults ever
selects). Both already handle kenlm being absent via `try/except`. It was
never going to load in this container regardless of whether the build
succeeded.

Fix: removed `kenlm` from `requirements-server.txt` (commented, explaining
why, so a future session doesn't re-add it during another
freeze-from-desktop-venv pass) instead of fixing its build. Then reverted
the `Dockerfile`'s system-deps line back down to just `python3 python3-pip
libsndfile1 curl git` — everything else in `requirements-server.txt` ships
prebuilt manylinux wheels for cp310, so `build-essential`/`cmake`/dev
headers were *only* ever needed for kenlm's from-source compile. Net
result: smaller image, faster build, and the actual root problem (wasted
build time on unreachable code) is gone rather than papered over.

`pyctcdecode` was left alone — no compile step (pure Python), so no cost
to keeping it even though it's also not exercised by anything `api/main.py`
calls; not worth the same audit effort for zero build-time benefit.

Lesson for future rounds: when a build/dependency error shows up, check
whether the failing package is actually reachable from the code that runs
in that image BEFORE fixing the build — fixing the build is only the right
move if the dependency is actually needed.

**2026-08-02 (later same day, sixth round) — same audit extended to the
desktop app (Claude, chat), per Hamza's "if we're not using kenlm at all
please exempt it."**

The round-5 Docker fix only addressed the container. Traced the same
question for the desktop app and found the identical pattern, worse in
scope: `HybridViterbiPipeline` (kenlm LM load + n-gram index build over all
6235 ayahs) was being constructed **eagerly, at every model load**, in two
places:
- `app.py::load_models_once()` (both the Groq branch and the regular
  branch) — its only consumer, `transcribe()`'s `viterbi_pipeline.pipeline_from_text()`
  call, is unreachable: `app.py`'s current `gr.Blocks()` only defines the
  "🎤️ Live Recitation" tab, no file-upload tab wires `transcribe()` to any
  button.
- `realtime_streamer.py::_ensure_model_loaded()` (both the groq branch and
  the regular branch) — its only consumer, `decode_full_session()`, is
  also unreachable: `app.py`'s `stop_live_session()` has its own comment
  confirming the full-session re-decode/compare step was deliberately
  removed already, and `api/main.py` never calls it either.

This is what the startup log Hamza pasted was actually showing
(`[HybridPipeline] KenLM model loaded...` / `Inverted index built: 97342
unique bigrams+trigrams`) — real load time and memory spent on an object
nothing in the currently-wired app ever calls a method on.

Fix: made both constructions lazy instead of deleting the (currently
unreachable but not necessarily dead-forever) code that uses them:
- `realtime_streamer.py` — added `_ensure_viterbi_pipeline()`, called only
  from `decode_full_session()`. Removed the two eager
  `self._viterbi_pipeline = HybridViterbiPipeline(...)` blocks from
  `_ensure_model_loaded()`.
- `app.py` — added module-level `_ensure_viterbi_pipeline()`, called only
  from `transcribe()` right before `viterbi_pipeline.pipeline_from_text()`.
  Removed the two eager `viterbi_pipeline = HybridViterbiPipeline(...)`
  blocks from `load_models_once()`.

Result: kenlm (and the n-gram index build, which costs time regardless of
whether kenlm itself is installed) no longer loads on desktop app startup
or on any live/API session — only if `transcribe()` or
`decode_full_session()` ever actually get called, which nothing currently
does. Neither function was deleted, since both could be legitimate
features to rewire back into the UI later (a file-upload tab, a
full-session-compare feature) — this only fixes when their dependency
loads, not whether the features exist.

Not done: didn't touch `requirements.txt` (desktop venv) or suggest
uninstalling kenlm from it — harmless to leave installed now that nothing
touches it at startup, and `hybrid_pipeline.py` already degrades
gracefully via `_KENLM_AVAILABLE` if it's ever removed.
