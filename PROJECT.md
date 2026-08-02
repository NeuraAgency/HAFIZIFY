# Hafizify — Project Documentation

> **Version:** 1.0.0
> **Last Updated:** 2026-08-02
> **Repository:** [github.com/NeuraAgency/HAFIZIFY](https://github.com/NeuraAgency/HAFIZIFY)

---

## 1. Overview

Hafizify is an **AI-powered Quran recitation assistant** that performs **real-time Arabic speech recognition**, corrects recitation errors against the Quran corpus, and tracks ayah-by-ayah progress — all through a **Gradio web interface**.

The system was built as a **Final Year Project (FYP)** combining a **dual-model ASR system** (Groq cloud Whisper + a locally fine-tuned Whisper-large-v3 model) with a custom-built **Quran-aware correction engine**, **BM25 + fuzzy surah detector**, **harakaat (diacritic) error detection**, and an **interactive Qari Mode** with TTS-based error correction feedback.

**The ASR engine** is the **Combined Model** — a hybrid of:

1. **Groq cloud Whisper** (`whisper-large-v3`) — provides the reliable consonant backbone.
2. **Local fine-tuned model** (`whisper-l-v3-turbo-quran-lora-dataset-mix`) — the only model that outputs Arabic **diacritics (harakaat/tashkeel)**.

The two outputs are merged word-by-word: Groq supplies the consonant skeleton, the local model supplies the vowels, producing a **diacritized transcription** that enables vowel-level (harakaat) error detection — something a single undiacritized ASR model cannot do.

**Requirements:** This mode needs **internet** (for the Groq API) and a **GPU** for real-time performance (RTX 2060 Super in development, GPU server in deployment). CUDA is auto-detected with a graceful CPU fallback.

---

## 2. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       GRADIO WEB UI (app.py)                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Tab: Live Recitation (Streaming Mic)              │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────┼─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                    Real-time Streamer (realtime_streamer.py)        │
│                    Session Manager (session_manager.py)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────┐    ┌──────────────────────────────┐  │
│  │  Combined ASR Engine       │    │  Correction Pipeline         │  │
│  │  ┌─────────────────────┐  │    │  ┌────────────────────────┐  │  │
│  │  │ Groq Whisper v3     │  │    │  │ Quran Guard            │  │  │
│  │  │ (cloud, consonants) │  │    │  │ (quran_guard.py)       │  │  │
│  │  ├─────────────────────┤  │    │  └───────────┬────────────┘  │  │
│  │  │ Local Turbo LoRA    │  │    │              │               │  │
│  │  │ (GPU, diacritics)   │  │    │              ▼               │  │
│  │  └─────────┬───────────┘  │    │  ┌────────────────────────┐  │  │
│  │            ▼              │    │  │ Harakaat Error         │  │  │
│  │  ┌─────────────────────┐  │    │  │ Detector               │  │  │
│  │  │ Hybrid Diacritic    │  │    │  │ (harakaat_error_       │  │  │
│  │  │ Pipeline (merge)    │  │    │  │  detector.py)          │  │  │
│  │  └─────────────────────┘  │    │  └────────────────────────┘  │  │
│  └───────────────────────────┘    └──────────────────────────────┘  │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────┐            │
│  │  Surah Detector      │    │  Live Display Formatter  │            │
│  │  (surah_detector.py) │    │  (live_display_formatter)│            │
│  └─────────────────────┘    └──────────────────────────┘            │
│                                                                      │
│  ┌─────────────────────┐    ┌──────────────────────────┐            │
│  │  Correction Engine   │    │  Quran Audio Provider   │            │
│  │  (correction_engine) │    │  (quran_audio_provider) │            │
│  │  (Qari Mode TTS)     │    │  (Ayah Audio Fetch)     │            │
│  └─────────────────────┘    └──────────────────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The ASR Model — Combined (Groq + Local, Harakaat-Aware)

The system uses **one ASR engine**: the **Combined Model**. It runs **two models per audio chunk** and merges their output into a single diacritized transcription.

### 3.1 Groq Cloud Whisper (`whisper-large-v3`)

| Property | Value |
|----------|-------|
| **Type** | Cloud ASR API (Groq REST API) |
| **Model** | `whisper-large-v3` |
| **Role** | Reliable consonant backbone of the merged output |
| **Internet required** | Yes |
| **API key** | Loaded from `.env` via `groq_transcriber.py` (never hardcoded) |

### 3.2 Local Fine-Tuned Model (`whisper-l-v3-turbo-quran-lora-dataset-mix`)

| Property | Value |
|----------|-------|
| **Type** | HuggingFace Transformers Whisper (fine-tuned LoRA, merged) |
| **Model** | `whisper-l-v3-turbo-quran-lora-dataset-mix` (local folder / HF: `MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix`) |
| **Role** | Supplies Arabic **diacritics (harakaat/tashkeel)** — the only model in the system that outputs vowels |
| **Compute** | GPU (CUDA auto-detected, `float16`); falls back to CPU (`float32`) if no GPU |
| **Loading** | Lazy-loaded only when Combined Mode is first used — never at app startup |

### 3.3 The Merge Algorithm

The two outputs are combined word-by-word:

1. **Word alignment** — Groq's words and the local model's words are aligned using `difflib.SequenceMatcher` over normalized (diacritic-stripped) skeletons.
2. **Vowel injection** — For each aligned word, the local model's diacritics are mapped by character position and injected onto Groq's consonant skeleton (`inject_vowels_by_character_position`).
3. **Result** — A diacritized transcription where Groq's more reliable consonants carry the local model's vowels.

This merge is implemented in `hybrid_diacritic_pipeline.py` (`run_combined_transcription`), which runs both models **concurrently** on the same audio chunk via a `ThreadPoolExecutor`.

---

## 4. Core Modules — Detailed Breakdown

### 4.1 app.py — Gradio Web UI & Orchestrator

**File:** `app.py`

The main entry point that:
- Loads the Gradio web interface on `http://127.0.0.1:7860`
- Manages the **Live Recitation** tab
- Runs a background worker thread for streaming chunk processing
- Manages session lifecycle (start → stream → stop → finalize)

**Key Functions:**
- `start_live_session()` — Creates a `RecitationSession`, starts the worker thread
- `process_streaming_audio()` — Gradio streaming callback, feeds audio deltas to the session
- `stop_live_session()` — Drains the queue, finalizes the session, runs full-session re-decode comparison
- `_process_queue_worker()` — Background thread that dispatches each ready chunk to `process_chunk_combined()` when the **ASR Engine** is set to Combined

**UI Components (Live Recitation tab):**
- **⚙️ gear button** — opens an Advanced Settings modal containing:
  - **🎯 Active Tajweed Detection** checkbox — the actual Combined Mode switch. Toggling it flips a hidden `ASR Engine` radio between `"Standard (offline, fast)"` and `"Combined (Groq + Local, harakaat-aware, needs internet)"` (default: off / Standard)
  - **🎙️ Interactive Qari Mode** checkbox
- **Surah** dropdown (default: Al-Fatiha)
- **▶️ Start Session** button — creates the session and preloads the model before the mic opens
- **Microphone** input (tap to record / tap to stop)
- **Qari Mode Status** badge, **Full Surah Progress** display — visible by default
- **📊 Chunk Details** accordion (collapsed by default) — per-chunk results table
- **📊 Session Comparison** accordion (after stop)

Several controls from earlier iterations of the UI (ASR Model dropdown, Start Ayah number, Correction Mode, Auto Surah Detection, the raw HTML displays for expected ayah/error panel, Raw vs Corrected text) are currently hidden (`visible=False`) in the Gradio layout — the functions and parameters behind them are still fully wired, just not surfaced in the sidebar today.

### 4.2 realtime_streamer.py — Streaming ASR Bridge

**File:** `realtime_streamer.py`

Bridges Gradio's streaming microphone with the Combined ASR engine and correction pipelines.

**Key Responsibilities:**
- Decodes each audio chunk via `process_chunk_combined()`
- Runs guard inference on each chunk (correction + ayah matching)
- Detects surah via BM25 + fuzzy rolling buffer (last 8 chunks)
- Manages surah lock state through `SurahLockManager`
- Supports **Qari Mode** with interactive correction via `CorrectionEngine`
- Runs full-session re-decode on stop for comparison

**`process_chunk_combined()` — the Combined Mode chunk pipeline:**
1. **Anti-hallucination gate** — skips chunks shorter than 1.5s or with RMS below 0.005
2. **Combined decode** — calls `run_combined_transcription()` (Groq + local, merged diacritized text)
3. **Surah detection** — rolling-window BM25 + fuzzy detection with lock manager
4. **Guard inference** — the existing `guard_inference()` ayah-matching/correction pipeline (unchanged)
5. **Harakaat error detection** — runs `detect_harakaat_errors()` against the matched ayah's diacritized reference; attaches `harakaat_errors` + `harakaat_error_count` to the result
6. **Qari Mode dispatch** — when enabled, hands the verdict + harakaat errors to `CorrectionEngine.process_verdict()`

**Anti-Hallucination Measures:**
```python
_MIN_CHUNK_SAMPLES = int(1.5 * 16000)   # skip chunks shorter than 1.5s
_MIN_SPEECH_RMS    = 0.005               # skip near-silent chunks
```
- Strips leading Ta'awwuz ("أعوذ بالله من الشيطان الرجيم") before matching
- Repetition loop detection (≥4 identical consecutive tokens)
- Trailing partial token detection and removal

### 4.3 session_manager.py — Recitation Session & VAD Chunking

**File:** `session_manager.py`

Manages the lifecycle of a live recitation session with **Silero VAD**-based intelligent segmentation.

**Key Classes:**

#### RecitationSession
- **VAD segmentation:** Silero VAD splits audio on ayah-boundary pauses
- **Dual WAV Recording:** Saves individual chunk WAVs + full session WAV
- **Ayah Progression:** Tracks `current_ayah` across matched chunks
- **Context Window:** Each emitted chunk includes 1.0s pre-context + 0.5s post-context to prevent boundary word clipping
- **Segment ID Deduplication:** Uses `f"{abs_start}_{abs_end}"` as segment key to prevent duplicate emission across VAD rescans
- **Normalization:** Only boosts genuinely quiet audio (peak < 0.3) to 0.6

#### ChunkResult (Dataclass)
| Field | Type | Description |
|-------|------|-------------|
| `chunk_index` | int | Sequential index |
| `start_sample` / `end_sample` | int | Sample positions in session |
| `start_time_s` / `end_time_s` / `duration_s` | float | Time domain |
| `raw_asr` | str | Raw ASR output text |
| `corrected_text` | str | Guard-corrected text |
| `matched_ayah` | int | Matched ayah number |
| `matched_surah_ayah_id` | str | E.g. "1:1" |
| `cer` / `wer` / `coverage` | float | Error metrics |
| `confidence` | float | Match confidence (0–1) |
| `confidence_level` | str | "high", "medium", "low" |
| `verdict` | str | "ok", "minor", "error", "skipped" |
| `chunk_wav_path` | str | Path to saved WAV |
| `matched_ayah_text` | str | Reference ayah text |
| `errors` | list | Word-level error annotations |
| `surah_lock_state` | dict | SurahLockManager state snapshot |
| `harakaat_errors` | list | Word-level vowel-error annotations (Combined Mode only) |
| `harakaat_error_count` | int | Count of vowel-only mistakes (Combined Mode only) |

### 4.4 hybrid_diacritic_pipeline.py — Combined ASR Engine

**File:** `hybrid_diacritic_pipeline.py`

Runs the two models on one audio chunk and merges their output into a single diacritized transcription.

**Key Functions:**
- `transcribe_groq()` — Undiacritized consonant-backbone transcription via the existing Groq client
- `transcribe_local()` — Diacritized transcription via the local turbo LoRA model (lazy-loaded, CUDA-aware)
- `smart_normalize_word()` — Diacritic-stripping + alef-variant normalization for alignment
- `inject_vowels_by_character_position()` — Rebuilds Groq's consonant skeleton with the local model's diacritics, matched by character position
- `run_hybrid_combination_logic()` — Word-aligns Groq vs local output, keeps Groq's consonants, injects local's diacritics
- `run_combined_transcription()` — **Public entry point** — runs both models concurrently and returns `{groq_text, local_text, combined_text}`

**Design note:** This module does **not** decide whether the recitation was correct — it only reconstructs "what was actually said, with vowels". Error detection against the reference ayah is a separate concern handled by `harakaat_error_detector.py`.

### 4.5 harakaat_error_detector.py — Vowel-Level Error Detection

**File:** `harakaat_error_detector.py`

Compares the diacritized ASR transcription against the expected ayah's diacritized reference text and classifies each word as:

| Status | Meaning |
|--------|---------|
| `ok` | Consonants and diacritics both match |
| `harakaat_error` | Consonant skeleton matches, diacritics differ — a vowel/tajweed-mark mistake (previously undetectable, since the standard guard pipeline strips all diacritics before comparing) |
| `makhraj_error` | Consonant skeleton itself differs — wrong word/letter |
| `missing` / `extra` | Word-count drift relative to the reference |

**Reference source:** `fyp_model/all_ayat.json` — 6,235 ayahs, 6,215 of them (99.7%) carry full Uthmani tashkeel. Edge cases handled gracefully:
- `1:1` (the Basmala) has no key → `skipped` with `skip_reason="no_reference"`
- ~20 ayahs have no diacritics → `skipped` with `skip_reason="reference_not_diacritized"`

**Key Functions:**
- `load_diacritized_ayat_map()` — Loads `all_ayat.json` keeping full tashkeel (deliberately NOT run through `normalize_arabic()`, which strips diacritics)
- `detect_harakaat_errors()` — Main entry point; word-aligns predicted vs reference using `difflib.SequenceMatcher` over consonant-only skeletons, then classifies each word

### 4.6 correction_engine.py — Interactive Qari Mode

**File:** `correction_engine.py`

State machine that implements **Interactive Qari Mode** — providing real-time audio feedback when the reciter makes an error.

**State Machine:**
```
LISTENING → CORRECTING → VERIFYING → CONFIRMED/SKIPPED
```

**Triggering Rules:**
```python
_consecutive_errors = 0         # tracks consecutive error chunks
_min_trigger_confidence = 0.30  # minimum confidence to treat as a real error
```

The engine only transitions to CORRECTING when:
1. **2 consecutive error chunks** (a single isolated bad chunk is ignored — could be a mid-ayah split)
2. **confidence > 0.30** (very low confidence = garbled/silent audio, not a recitation mistake)
3. Counter resets to 0 on any `ok` or `minor` verdict

**Harakaat Hint (Combined Mode only):**
When a chunk has `harakaat_errors` (vowel-only mistakes) but the verdict is `ok`/`minor`, the engine plays a short, distinct audio cue — **"انتبه للتشكيل"** — and immediately returns to `LISTENING`. This is a lightweight review note, **not** a stop-and-retry event: it does **not** use the strict consecutive-error gate, does not enter `CORRECTING`/`VERIFYING`, and never interrupts the reciter.

**TTS Feedback:**
- Primary: `QuranAudioProvider` — fetches exact ayah audio from Quran.com API with word-level timestamps
- Fallback: `edge_tts` with `ar-SA-HamedNeural` voice (Microsoft Edge TTS)

**`process_verdict()` signature:**
```python
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
    harakaat_errors: list | None = None,   # Combined Mode only
) -> dict:
```

### 4.7 live_display_formatter.py — Live Color-Coded Display

**File:** `live_display_formatter.py`

Generates HTML with word-level color coding for the live merged transcript.

**Colors:**
| Color | Meaning |
|-------|---------|
| 🟢 Green (`#10b981`) | Correct word (exact match after normalization) |
| 🟠 Amber (`#f59e0b`) | Minor error (>70% character similarity) |
| 🔴 Red (`#ef4444`) | Major error (<70% similarity) |
| ⚪ Gray (`#9ca3af`) | Low confidence (<0.4) |
| 🟡 Gold (`#eab308`) | **Harakaat (vowel-only) mistake — Combined Mode only** (consonants correct, diacritics wrong) |

The gold harakaat highlight only fires when `harakaat_errors` is non-empty on a chunk result — standard mode chunks never populate that field.

### 4.8 surah_detector.py — High-Precision Surah Detection

**File:** `surah_detector.py`

Two-stage information retrieval pipeline for auto-detecting which surah is being recited.

**Stage 1: BM25 Retrieval**
- Word-level BM25 scoring at the ayah level (6,235 documents)
- k1=1.5, b=0.75
- Top-30 ayahs forwarded to Stage 2

**Stage 2: Fuzzy Substring Reranking**
- `fuzz.partial_ratio` handles fragments, ASR typos, split-word spacing
- Combined score: 40% BM25 + 60% fuzzy

**Key Design Notes:**
- Basmala (1:1) and Al-Fatiha 1:1 excluded from index (shared by 113 surahs — useless for identification)
- Ta'awwuz stripped before scoring; Basmala is NOT stripped (it identifies surahs starting with it)

#### SurahLockManager
```python
min_score = 0.35
avg_score_threshold = 0.45
margin_threshold = 0.15
history_size = 5
lock_votes = 3
unlock_score = 0.20
unlock_votes = 3
```

### 4.9 fyp_model/quran_guard.py — Quran-Aware Correction Engine

**File:** `fyp_model/quran_guard.py`

The **central correction pipeline** that maps raw ASR output to the correct Quranic ayah.

**Key Functions:**

#### `normalize_arabic(text) -> str`
Standard normalization for all Arabic text in the system:
- Removes diacritics (tashkeel): Fatha, Kasra, Damma, Shadda, etc.
- Removes tatweel (kashida) and special characters
- Keeps only Arabic letters (U+0600–U+06FF), digits, and spaces

#### `correct_text_rules()` — Rule-Based Pre-Correction
Applied before ayah matching. Two rule tables:
- **`_PHRASE_FIXES`** — Multi-word phrase corrections (e.g. `"الر حيم"` → `"الرحيم"`, `"اهتنا"` → `"اهدنا"`, `"السراط"` → `"الصراط"`)
- **`_TOKEN_FIXES`** — Single-token corrections (e.g. `"المغبوب"` → `"المغضوب"`, `"مالكيع"` → `"مالك"`)

#### `guard_inference()` — Public Entry Point
Parameters:
- `raw_text` — Raw ASR output
- `ayah_map` — Full Quran ayah database
- `surah` / `expected_ayah` — Current position context
- `lookahead` / `window_back` — Search window
- `correction_mode` — "safe", "balanced", or "aggressive"
- `preserve_reciter` — If True, avoid full ayah replacement
- `use_sequence_match` — Enable multi-ayah sequence matching
- `lock_surah` — Restrict matching to a specific surah

#### Confidence Computation
```python
# Single ayah
confidence = 0.65 * (1.0 - CER) + 0.35 * coverage

# Sequence match
confidence = 0.45 * (1.0 - CER) + 0.20 * (1.0 - WER) + 0.30 * coverage + span_bonus
```

---

## 5. Live Recitation Data Flow

```
User clicks "Start"
  │
  ▼
start_live_session()
  ├── Creates RecitationSession (VAD-based)
  ├── Starts background worker thread
  └── Returns placeholder UI
  │
  ▼ (repeated via Gradio streaming)
process_streaming_audio()
  ├── Resamples audio to 16kHz
  ├── Detects incremental vs cumulative streaming mode
  ├── Feeds audio DELTA into session buffer
  └── Session extracts ready VAD chunks
  │
  ▼ (background worker)
_process_queue_worker()
  └── Calls rt_streamer.process_chunk_combined()
        │
        ▼
    process_chunk_combined()
      ├── 1. Anti-hallucination gate (RMS & duration check)
      ├── 2. Combined decode (Groq + local, merged diacritized text)
      ├── 3. Strip leading Ta'awwuz
      ├── 4. Update BM25 detection buffer (last 8 chunks)
      ├── 5. Surah detection via SurahDetector
      ├── 6. Guard inference (rule fix + ayah matching + correction)
      ├── 7. Guard confidence vote → SurahLockManager
      ├── 8. Harakaat error detection vs diacritized reference
      ├── 9. Qari Mode word-level annotations + harakaat hint (if enabled)
      │     └── CorrectionEngine.process_verdict(harakaat_errors=...)
      └── 10. Register ChunkResult in session
  │
  ▼ (UI updates)
Gradio returns HTML:
  ├── Expected Ayah Display (color-coded words, gold for harakaat slips)
  ├── Surah Progress Bar
  ├── Error Statistics Panel
  ├── Raw vs Corrected Text
  └── Chunk Details Table
  │
  ▼ (User clicks "Stop")
stop_live_session()
  ├── Flushes remaining audio
  ├── Drains queue
  ├── Finalizes session (saves WAV + JSON)
  └── Runs full-session re-decode
      ├── Decodes entire audio as one piece
      ├── Guard correction (aggressive mode, sequence_max_ayahs=30)
      └── Viterbi alignment on full text
```

---

## 6. Data Assets

### 6.1 Quran Reference Database
**File:** `fyp_model/all_ayat.json`
- 6,235 ayahs of the Quran
- Key format: `"<surah>_<ayah>"` (e.g. `"1_1"`, `"2_255"`)
- 99.7% of ayahs carry full Uthmani tashkeel in `text` — used as the diacritized reference for harakaat error detection
- Loaded in two forms:
  - **Normalized** (diacritics stripped) by `quran_guard.py` for ayah matching/correction
  - **Raw diacritized** by `harakaat_error_detector.py` for vowel-level comparison

### 6.2 Language Model
**File:** `quran_5gram.arpa`
- 5-gram ARPA-format KenLM trained on Quranic text
- Used by the Viterbi alignment pipeline on session finalization

### 6.3 Recording Outputs
```
recordings/
└── <timestamp>_<session_id>/
    ├── session_full.wav          # Complete session recording (16kHz mono)
    ├── results.json              # Full session metadata + all chunk results
    ├── chunks/
    │   ├── chunk_0000.wav
    │   └── ...
    └── vad_segments/
        ├── vad_0000_<start>_<end>.wav   # True speech boundaries (without context)
        └── ...
```

---

## 7. Dependencies

| Package | Purpose |
|---------|---------|
| `torch` ≥2.2.0 | Deep learning framework (GPU inference for the local model) |
| `torchaudio` | Audio processing, resampling |
| `transformers` | HuggingFace model loading (local turbo LoRA model) |
| `faster-whisper` | CTranslate2-optimized Whisper inference |
| `gradio` | Web UI framework (streaming audio support) |
| `silero-vad` ≥5.0 | Voice Activity Detection for segmentation |
| `soundfile` | WAV file I/O |
| `librosa` | Audio loading |
| `rapidfuzz` | Fast string matching (Levenshtein, partial_ratio) |
| `edge-tts` | Microsoft Edge TTS for Qari Mode audio feedback |
| `pygame` | Audio playback for TTS |
| `numpy` <2.3.0 | Numerical operations |
| `groq` | Groq cloud ASR API (Combined Mode) |
| `kenlm` | KenLM LM scoring (Viterbi pipeline) |

---

## 8. Running the System

### Prerequisites
```bash
# Python 3.10+
# FFmpeg (for audio decoding)
# Windows: choco install ffmpeg
# GPU recommended for Combined Mode (RTX 2060 Super / GPU server)
# Internet connection required (Groq API)
```

### Setup
```bash
git clone https://github.com/NeuraAgency/HAFIZIFY.git
cd HAFIZIFY
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Configure the Groq API Key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Run
```bash
python app.py
# Opens at http://127.0.0.1:7860
```

### Using Combined Mode
1. Open the **Live Recitation** tab.
2. Click the **⚙️ gear button** to open Advanced Settings.
3. Enable **🎯 Active Tajweed Detection** — this switches the (hidden) ASR Engine to `"Combined (Groq + Local, harakaat-aware, needs internet)"`.
4. Optionally enable **🎙️ Interactive Qari Mode** for TTS-based correction feedback — including the gentle "انتبه للتشكيل" hint on vowel mistakes.
5. Close the modal, select the surah, and click **▶️ Start Session**.
6. Tap the microphone to start recording and begin reciting.
7. The local turbo model loads lazily on the first Combined Mode chunk (GPU-accelerated if CUDA is available).
8. Word-level results — including gold harakaat-slip highlighting — are computed every chunk and available via the Chunk Details table and error panel.
9. Tap the microphone again to stop, which finalizes the session and runs the full-session comparison (visible in the Session Comparison accordion).

---

## 9. File Tree (Live Recitation)

```
hafizify/
├── app.py                        # Main Gradio app — entry point
├── realtime_streamer.py          # Streaming ASR pipeline (process_chunk_combined)
├── session_manager.py            # Per-session recitation state + VAD segmentation
├── hybrid_diacritic_pipeline.py  # Combined ASR engine (Groq + local merge)
├── harakaat_error_detector.py    # Vowel-level (harakaat) error detection
├── correction_engine.py          # Qari Mode state machine + harakaat hint
├── live_display_formatter.py     # HTML formatting for Gradio UI
├── surah_detector.py             # BM25 + fuzzy surah auto-detection
├── quran_audio_provider.py       # Quran.com audio fetch + word segments
├── groq_transcriber.py           # Groq cloud API client (env-based key)
├── hybrid_pipeline.py            # Viterbi + LM alignment pipeline
├── quran_5gram.arpa              # Pre-built Quran 5-gram ARPA LM
├── requirements.txt              # Python dependencies
├── PROJECT.md                    # THIS FILE
│
├── fyp_model/                    # Core model directory
│   ├── __init__.py
│   ├── all_ayat.json             # Full Quran ayah database (6,235 ayahs, diacritized)
│   ├── quran_guard.py            # Correction + ayah matching engine
│   └── run.py                    # CLI inference tool
│
├── whisper-l-v3-turbo-quran-lora-dataset-mix/   # Local fine-tuned model (diacritics)
│
├── data/
│   ├── number_of_verses.txt
│   └── readerlist.tsv
│
├── audio_cache/                  # Quran.com audio clips cache (runtime)
└── recordings/                   # Session output (runtime generated)