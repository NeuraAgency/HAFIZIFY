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
