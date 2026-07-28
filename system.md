# Hafizify — Complete System Architecture Documentation

> **Version:** 1.2.0
> **Last Updated:** 2026-06-29
> **Repository:** [github.com/NeuraAgency/HAFIZIFY](https://github.com/NeuraAgency/HAFIZIFY)

---

## 1. Overview

Hafizify is an **AI-powered Quran recitation assistant** that performs **real-time Arabic speech recognition**, corrects recitation errors against the Quran corpus, and tracks ayah-by-ayah progress — all running locally via a **Gradio web interface**.

The system was built as a **Final Year Project (FYP)** combining Speech Recognition (faster-whisper CTranslate2) with a custom-built **Quran-aware correction engine**, **Viterbi alignment pipeline**, **BM25 + fuzzy surah detector**, and an **interactive Qari Mode** with TTS-based error correction feedback.

**Design constraint:** The system targets **CPU-only inference** (IoT/laptop deployment) with no GPU requirement. The single offline model (`whisper-base-quran-lora-ct2`) is the only model exposed in the UI.

---

## 2. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       GRADIO WEB UI (app.py)                        │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Tab 1: Live Recitation  │  │  Tab 2: Upload Audio            │  │
│  │  (Streaming Mic)         │  │  (File-based Batch)             │  │
│  └───────────┬─────────────┘  └───────────────┬──────────────────┘  │
└──────────────┼────────────────────────────────┼─────────────────────┘
               │                                │
┌──────────────▼────────────────────────────────▼─────────────────────┐
│                    Real-time Streamer (realtime_streamer.py)         │
│                    Session Manager (session_manager.py)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────┐    ┌──────────────────────────┐          │
│  │  ASR Model (single)    │    │  Correction Pipeline     │          │
│  │  ┌─────────────────┐  │    │  ┌────────────────────┐  │          │
│  │  │ faster-whisper  │  │    │  │ Quran Guard        │  │          │
│  │  │ (CT2, int8 CPU) │  │    │  │ (quran_guard.py)   │  │          │
│  │  └─────────────────┘  │    │  └─────────┬──────────┘  │          │
│  └───────────────────────┘    │            │             │          │
│                               │            ▼             │          │
│                               │  ┌────────────────────┐  │          │
│                               │  │ Hybrid Viterbi     │  │          │
│                               │  │ (hybrid_pipeline)  │  │          │
│                               │  └────────────────────┘  │          │
│                               └──────────────────────────┘          │
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

## 3. ASR Model

The system uses **one offline model only** exposed in the UI. All other model variants were evaluated and removed from the UI for clarity and IoT suitability.

### 3.1 whisper-base-quran-lora-ct2 (Offline, CPU-Optimized — Default & Only)

| Property | Value |
|----------|-------|
| **Local path** | `whisper-base-quran-lora-ct2/` |
| **Registry key** | `"whisper-base-quran-lora"` |
| **Type** | CTranslate2 int8 quantized Whisper, loaded via `faster_whisper` |
| **Origin** | CTranslate2 conversion of the LoRA-merged Whisper-base model |
| **Compute** | CPU with `int8` quantization, uses all available threads |
| **Decode time** | ~1.7–2.5s per 3–6s audio chunk on modern CPU |
| **GPU required** | No |
| **Internet required** | No (fully offline) |

**Why this model:**
- fastest-whisper's CTranslate2 backend gives 3–4x speedup over HuggingFace Transformers on CPU
- int8 quantization reduces memory footprint significantly
- Already on disk — no HuggingFace download needed at runtime
- No `forced_decoder_ids` / logits processor conflicts (faster-whisper handles this internally)

### 3.2 Registry (Internal — Not Exposed in UI)

These entries exist in `_MODEL_REGISTRY` in `realtime_streamer.py` for developer use but are not shown in the Gradio dropdown:

| Key | Description |
|-----|-------------|
| `faster-whisper-base-ar-quran` | Alias pointing to same `whisper-base-quran-lora-ct2` folder |
| `whisper-base-quran-lora` | HF Transformers version (slower on CPU, kept as fallback) |
| `groq-whisper-large-v3-turbo` | Cloud API — best quality, requires internet + Groq API key |

---

## 4. Core Modules — Detailed Breakdown

### 4.1 app.py — Gradio Web UI & Orchestrator

**File:** `app.py`

The main entry point that:
- Loads the Gradio web interface on `http://127.0.0.1:7860`
- Manages **single model** initialization at startup (no model switching in UI)
- Handles both **Live Recitation** and **Upload Audio** tabs
- Runs a background worker thread for streaming chunk processing
- Manages session lifecycle (start → stream → stop → finalize → compare)

**Model Init (single):**
```python
MODEL_CHOICES = [
    "whisper-base-quran-lora",   # → resolves to whisper-base-quran-lora-ct2 (offline)
]
```

**Key Functions:**
- `load_models_once()` — Loads the CT2 model + ayah_map + viterbi_pipeline + surah_detector at first use
- `transcribe()` — File upload transcription with guard correction + Viterbi alignment
- `start_live_session()` — Creates RecitationSession, starts worker thread
- `process_streaming_audio()` — Gradio streaming callback, feeds audio delta to session
- `stop_live_session()` — Drains queue, finalizes session, runs full re-decode comparison

**UI Components:**
- **Tab 1 (Live Recitation):** Surah dropdown, ayah number, correction mode, chunk settings, VAD toggle, Auto Surah Detection, Qari Mode, mic input, live HTML display (expected ayah, surah progress, error panel), session comparison accordion
- **Tab 2 (Upload Audio):** File upload, beam search toggle, correction mode, sequence guard, reference replacement toggle

---

### 4.2 realtime_streamer.py — Streaming ASR Bridge

**File:** `realtime_streamer.py`

Bridges Gradio's streaming microphone with the faster-whisper model and correction pipelines.

**Key Responsibilities:**
- Loads `whisper-base-quran-lora-ct2` via `faster_whisper.WhisperModel` on first use
- Decodes each audio chunk with anti-hallucination gates (min 1.5s, min RMS 0.005)
- Runs guard inference on each chunk (correction + ayah matching)
- Detects surah via BM25 + fuzzy rolling buffer (last 8 chunks)
- Manages surah lock state through `SurahLockManager`
- Supports **Qari Mode** with interactive correction via `CorrectionEngine`
- Runs full-session re-decode on stop for comparison

**Model Loading:**
```python
_MODEL_REGISTRY = {
    "whisper-base-quran-lora": {
        "type": "whisper",
        "local": "whisper-base-quran-lora-ct2",   # CT2 folder on disk
        "hf": "KheemP/whisper-base-quran-lora",   # HF fallback if local missing
    },
    # ... developer-only entries not shown in UI
}
```

The `_resolve_model_path()` function checks for the local CT2 folder first; if present it loads it as faster-whisper (detected by `_ct2` suffix). If missing, falls back to HF download.

**Anti-Hallucination Measures:**
```python
_MIN_CHUNK_SAMPLES = int(1.5 * 16000)   # skip chunks shorter than 1.5s
_MIN_SPEECH_RMS    = 0.005               # skip near-silent chunks
```
- Strips leading Ta'awwuz ("أعوذ بالله من الشيطان الرجيم") before matching
- Repetition loop detection (≥4 identical consecutive tokens)
- Trailing partial token detection and removal

**Key Functions:**
- `_ensure_model_loaded()` — Loads model based on `_model_choice` from registry
- `_decode_chunk()` — Decodes a single audio chunk via faster-whisper
- `_decode_raw()` — Handles audio >30s by splitting into 25s chunks with 2s overlap
- `process_chunk()` — Core chunk pipeline (see Section 5.1)
- `create_session()` — Creates a `RecitationSession` with SurahLockManager
- `decode_full_session()` — Re-decodes entire session audio for comparison

---

### 4.3 session_manager.py — Recitation Session & VAD Chunking

**File:** `session_manager.py`

Manages the lifecycle of a live recitation session with **Silero VAD**-based intelligent chunking.

**Key Classes:**

#### RecitationSession
- **VAD Chunking:** Silero VAD splits audio on ayah-boundary pauses
- **Fixed-time fallback:** If VAD unavailable, chunks by fixed duration with configurable overlap
- **Dual WAV Recording:** Saves individual chunk WAVs + full session WAV
- **Ayah Progression:** Tracks `current_ayah` across matched chunks
- **Context Window:** Each emitted chunk includes 1.0s pre-context + 0.5s post-context to prevent boundary word clipping
- **Segment ID Deduplication:** Uses `f"{abs_start}_{abs_end}"` as segment key (not a fragile counter) to prevent duplicate emission across VAD rescans
- **Normalization:** Only boosts genuinely quiet audio (peak < 0.3) to 0.6 — does not over-compress normal speech

**VAD Tuning (Quran-optimized):**
```python
_VAD_MIN_SILENCE_MS      = 400     # Split on ayah-boundary pauses
_VAD_MIN_SPEECH_MS       = 300     # Ignore short noise bursts
_VAD_THRESHOLD           = 0.30    # Less sensitive — reduces false triggers
_VAD_END_PAD_MS          = 80      # Brief padding at segment end
_VAD_MIN_CHUNK_DURATION  = 2.0     # Minimum 2s — Whisper needs context
_VAD_MAX_CHUNK_DURATION  = 8.0     # Safety split for long segments
_VAD_RESCAN_INTERVAL     = 0.15    # Re-run VAD every 150ms
_VAD_TAIL_CONFIRM        = 0.20    # Wait for silence before emitting
```

**Context Window (anti-boundary-clipping):**
```python
_CTX_PRE_S  = 1.0   # 1s of audio before segment start
_CTX_POST_S = 0.5   # 0.5s of audio after segment end
```
Whisper receives `[pre_context | SPEECH | post_context]` but `abs_start/abs_end` are returned unchanged so `_vad_committed_up_to` advances correctly.

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

---

### 4.4 fyp_model/quran_guard.py — Quran-Aware Correction Engine

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

**`_PHRASE_FIXES`** — Multi-word phrase corrections (applied in order):
- Whisper split-word artifacts: `"الر حيم"` → `"الرحيم"`, `"الرحم ن"` → `"الرحمن"`, etc.
- Model-specific failures for `whisper-base-quran-lora-ct2`:

| Raw ASR Output | Corrected To |
|----------------|-------------|
| `أن عمتا ليهم` / `أنا أمتى ليهم` | `أنعمت عليهم` |
| `أويد المغضوب` / `أيد المغضوب` | `غير المغضوب` |
| `مالكيع من الدين` / `مالك من الدين` | `مالك يوم الدين` |
| `إيا كان` / `إيا كنا` | `إياك` |
| `وأياهاك` / `الوأيى` | `وإياك` |
| `السراط` / `الصيراط` / `سراط` | `الصراط` |
| `اهتنا` / `إهتنا` / `إهدنا` | `اهدنا` |
| `الضالن` / `الضالي` | `الضالين` |

**`_TOKEN_FIXES`** — Single-token corrections:

| Token | Corrected To |
|-------|-------------|
| `المغبوب` / `المغووب` | `المغضوب` |
| `مالكيع` | `مالك` |
| `اهتنا` / `إهتنا` | `اهدنا` |
| `السراط` | `الصراط` |
| `الصيرات` | `الصراط` |
| `الضالن` | `الضالين` |
| `نعبدو` | `نعبد` |
| `وأياهاك` | `وإياك` |
| `ويللمغموب` / `ويللمغووب` | `المغضوب` |

#### `guard_inference()` — Public Entry Point
Parameters:
- `raw_text` — Raw ASR output
- `ayah_map` — Full Quran ayah database
- `surah` / `expected_ayah` — Current position context
- `lookahead=5` / `window_back=2` — Search window (wider than default for better coverage)
- `correction_mode` — "safe", "balanced", or "aggressive"
- `preserve_reciter` — If True, avoid full ayah replacement
- `use_sequence_match` — Enable multi-ayah sequence matching
- `lock_surah` — Restrict matching to a specific surah

#### `apply_correction_pipeline()` — Core Logic
1. Rule-based fixes (`correct_text_rules`)
2. Single ayah matching (`match_ayah`) using CER, WER, token coverage
3. Optional sequence matching (`match_ayah_sequence`) for multi-ayah spans
4. Correction application by mode (safe / balanced / aggressive)
5. Adaptive confidence thresholds by token count

#### Confidence Computation
```python
# Single ayah
confidence = 0.65 * (1.0 - CER) + 0.35 * coverage

# Sequence match
confidence = 0.45 * (1.0 - CER) + 0.20 * (1.0 - WER) + 0.30 * coverage + span_bonus
```

**Adaptive thresholds by token count:**
| Tokens | High Threshold | Medium Threshold |
|--------|---------------|------------------|
| ≤3     | 0.40          | 0.25             |
| ≤6     | 0.60          | 0.40             |
| ≤10    | 0.72          | 0.55             |
| >10    | 0.82          | 0.65             |

---

### 4.5 hybrid_pipeline.py — Hybrid Viterbi Alignment

**File:** `hybrid_pipeline.py`

Production-grade ASR alignment pipeline combining multiple scoring methods. Runs on session finalization only (not per-chunk).

**Scoring Components:**
1. **KenLM 5-gram Score (50% weight)** — Scores text against `quran_5gram.arpa`
2. **N-gram Jaccard Overlap (30% weight)** — Unigram + Bigram + Trigram overlap
3. **Levenshtein Edit Distance (20% weight)** — RapidFuzz partial ratio score

**Viterbi DP:**
- Sliding windows (chunk_size=5, step=3 words) over full session text
- Inverted n-gram index for fast candidate pre-selection (top-30)
- Transition penalties: forward (+1.0), same (+0.5), jump-2 (−0.5), jump-3 (−1.0), backward (−5.0), cross-surah-invalid (−10.0)
- Beam pruning: top-30 states per timestep

---

### 4.6 surah_detector.py — High-Precision Surah Detection

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
- Correct matches score near 1.0; wrong matches score near 0.1

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

---

### 4.7 correction_engine.py — Interactive Qari Mode

**File:** `correction_engine.py`

State machine that implements **Interactive Qari Mode** — providing real-time audio feedback when the reciter makes an error.

**State Machine:**
```
LISTENING → CORRECTING → VERIFYING → CONFIRMED/SKIPPED
```

**Triggering Rules (updated):**
```python
_consecutive_errors = 0         # tracks consecutive error chunks
_min_trigger_confidence = 0.30  # minimum confidence to treat as a real error
```

The engine only transitions to CORRECTING when:
1. **2 consecutive error chunks** (single isolated bad chunk is ignored — could be mid-ayah split)
2. **confidence > 0.30** (very low confidence = garbled/silent audio, not a recitation mistake)
3. Counter resets to 0 on any `ok` or `minor` verdict

This prevents the previous behavior where a single chunk with `verdict=error` immediately paused the session mid-ayah (e.g., `إياك نعبد وإياك نستعين` split across 4 chunks each triggering interruptions).

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
    confidence: float = 1.0,     # ← now passed from guard result
) -> dict:
```

---

## 5. Data Flow

### 5.1 Live Recitation Flow

```
User clicks "Start"
  │
  ▼
start_live_session()
  ├── Creates RecitationSession (VAD-based)
  ├── Starts background worker thread
  └── Returns placeholder UI
  │
  ▼ (repeated every ~150ms via Gradio streaming)
process_streaming_audio()
  ├── Resamples audio to 16kHz
  ├── Detects incremental vs cumulative streaming mode
  ├── Feeds audio DELTA into session buffer
  └── Session extracts ready VAD chunks
  │
  ▼ (background worker)
_process_queue_worker()
  └── Calls rt_streamer.process_chunk()
        │
        ▼
    process_chunk()
      ├── 1. Anti-hallucination gate (RMS & duration check)
      ├── 2. Raw ASR decode via faster-whisper
      ├── 3. Strip leading Ta'awwuz
      ├── 4. Update BM25 detection buffer (last 8 chunks)
      ├── 5. Surah detection via SurahDetector
      ├── 6. Guard inference (rule fix + ayah matching + correction)
      ├── 7. Guard confidence vote → SurahLockManager
      ├── 8. Qari Mode word-level annotations (if enabled)
      │     └── CorrectionEngine.process_verdict(confidence=...)
      └── 9. Register ChunkResult in session
  │
  ▼ (UI updates)
Gradio returns HTML:
  ├── Expected Ayah Display (color-coded words)
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

### 5.2 Upload Audio Flow

```
User uploads file + clicks "Transcribe"
  │
  ▼
transcribe()
  ├── Loads audio to 16kHz
  ├── ASR decode via faster-whisper (greedy or beam search)
  ├── Guard inference (ayah matching + correction)
  ├── Viterbi alignment on guard-corrected text
  ├── Surah detection on raw text
  └── Returns: raw_text, corrected_text, confidence, verdict,
               decode_method, detected_surah, eval_report
```

---

## 6. Data Assets

### 6.1 Quran Reference Database
**File:** `fyp_model/all_ayat.json`
- 6,235 ayahs of the Quran
- Key format: `"<surah>_<ayah>"` (e.g. `"1_1"`, `"2_255"`)
- Loaded and normalized at startup by every module

### 6.2 Language Model
**File:** `quran_5gram.arpa`
- 5-gram ARPA-format KenLM trained on Quranic text
- Used by `HybridViterbiPipeline` (50% weight in scoring)
- Training: `train_kenlm_python.py` or `train_kenlm.ps1`

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

## 7. Configuration & Settings

### 7.1 Correction Modes
| Mode | Behavior |
|------|----------|
| `safe` | Rule-based fixes only, no ayah replacement |
| `balanced` | Partial word correction at medium confidence, full replacement at high |
| `aggressive` | Full ayah replacement at medium+ confidence |

Default for live: `aggressive` | Default for upload: `aggressive`

### 7.2 Chunk Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `use_vad` | True | Use Silero VAD for intelligent chunking |
| `chunk_duration` | 5.0s | Fallback chunk duration if VAD off |
| `overlap_duration` | 1.5s | Overlap between consecutive chunks |
| Detection buffer | 8 chunks | Rolling window for surah detection |

### 7.3 Model in UI (single)
```python
MODEL_CHOICES = [
    "whisper-base-quran-lora",   # → whisper-base-quran-lora-ct2/ (offline, CPU, int8)
]
```

---

## 8. Dependencies

| Package | Purpose |
|---------|---------|
| `torch` ≥2.2.0 | Deep learning framework |
| `torchaudio` | Audio processing, resampling |
| `faster-whisper` | CTranslate2-optimized Whisper inference (primary ASR) |
| `transformers` | HuggingFace model loading (fallback path only) |
| `peft` | LoRA adapter loading (fallback path only) |
| `gradio` | Web UI framework (streaming audio support) |
| `silero-vad` ≥5.0 | Voice Activity Detection for chunking |
| `soundfile` | WAV file I/O |
| `librosa` | Audio loading |
| `rapidfuzz` | Fast string matching (Levenshtein, partial_ratio) |
| `edge-tts` | Microsoft Edge TTS for Qari Mode audio feedback |
| `pygame` | Audio playback for TTS |
| `numpy` <2.3.0 | Numerical operations |
| `pyctcdecode` | CTC beam search decoder (optional, Wav2Vec2 only) |
| `kenlm` | KenLM LM scoring (optional, Viterbi pipeline) |
| `groq` | Groq cloud ASR API (optional, developer use only) |

---

## 9. Scripts & Tooling

### 9.1 LM Training
| Script | Purpose |
|--------|---------|
| `prepare_quran_lm.py` | Cleans Quran text → `quran_lm.txt` |
| `train_kenlm_python.py` | Builds 5-gram ARPA model (CLI or Python fallback) |
| `train_kenlm.ps1` | PowerShell helper: preparation + LM training |

### 9.2 Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `scripts/eval_wav2vec2_quran.py` | WER/CER evaluation on manifest CSV |
| `scripts/evaluate_whole_quran_sequence.py` | Sequence-aware evaluation |
| `scripts/run_dual_test.py` | Compares two models on same audio |
| `scripts/run_all_reciters.py` | Batch evaluation across reciters |

### 9.3 One-time Patch Scripts (delete after use)
| Script | Purpose |
|--------|---------|
| `patch_quran_guard.py` | Adds CT2 model correction rules to `fyp_model/quran_guard.py` |
| `apply_realtime_patch.py` | Adds `confidence=` param to `process_verdict` call |

### 9.4 CLI Tool
**File:** `fyp_model/run.py` — Single-file inference with progressive chunk decoding.

---

## 10. Qari Mode — Interactive Error Correction

### State Machine
```
LISTENING → CORRECTING → VERIFYING → CONFIRMED/SKIPPED
```

### Trigger Logic
```
On each chunk verdict:
  ok / minor  → reset consecutive_errors counter → stay LISTENING
  error       → increment consecutive_errors
                if consecutive_errors < 2: stay LISTENING (warn only)
                if confidence < 0.30:      stay LISTENING (garbled audio)
                else: reset counter → CORRECTING
```

### Flow
1. **LISTENING:** Processes chunks, shows live display
2. **Error detected** (2nd consecutive error, confidence > 0.30):
   - State → CORRECTING
   - Plays wrong word(s) via Quran audio clip or Edge TTS fallback
3. **VERIFYING:** Listens for corrected word
   - Correct (verdict="ok"): Play "أحسنت" → LISTENING
   - Still wrong: Increment attempts, retry (max 3)
   - Max attempts exceeded: Skip, speak correct word, advance to next ayah

### Audio Feedback Sources
1. **Quran Audio Provider** — Quran.com API with word-level timestamps (segment playback)
2. **Edge TTS Fallback** — `ar-SA-HamedNeural` voice when Quran.com unavailable (HTTP 403 handled gracefully)

---

## 11. Performance & Anti-Hallucination

### Anti-Hallucination Gates
| Gate | Threshold | Purpose |
|------|-----------|---------|
| Min chunk duration | 1.5s | Skip empty/silent chunks |
| Min RMS energy | 0.005 | Skip near-silent chunks |
| Repetition detection | ≥4 identical tokens | Prevent hallucination loops |
| Trailing partial token | <2 chars | Remove incomplete final tokens |
| Leading invocation strip | Ta'awwuz only | Clean input for matching |
| Adaptive confidence | By token count | Prevent false matches on short input |
| Qari Mode consecutive gate | 2 errors required | Prevent mid-ayah interruptions |

### Surah Detection Safeguards
- Basmala (1:1) excluded from detection index
- Guard confidence < 0.70 at ayah=1: synthetic candidate not injected
- Rolling buffer: minimum 8 chunks before detection starts
- Lock: 3+ consistent chunks with clear margin

### Decode Speed (CPU, whisper-base-quran-lora-ct2)
| Chunk Duration | Decode Time | Real-time Ratio |
|---------------|-------------|-----------------|
| 2–3s          | ~1.7–1.9s   | ~0.7x (faster than real-time) |
| 3–6s          | ~1.9–2.5s   | ~0.4x–0.6x |
| 6–8s          | ~2.5–3.5s   | ~0.3x–0.5x |

---

## 12. Model Files & Weights

### On Disk (committed / pre-existing)
| Path | Description |
|------|-------------|
| `whisper-base-quran-lora-ct2/` | **Primary model** — CTranslate2 int8 quantized, offline |
| `fyp_model/weights/` | Wav2Vec2 large-xlsr-53 fine-tuned (developer fallback) |
| `fyp_model/all_ayat.json` | Full Quran ayah database (6,235 ayahs) |
| `quran_5gram.arpa` | Pre-built 5-gram KenLM language model |
| `quran_lm.txt` | Cleaned Quran text for LM training |

### Available via HuggingFace (not auto-downloaded at runtime)
| Model | HF ID | Use |
|-------|-------|-----|
| HF Whisper LoRA | `KheemP/whisper-base-quran-lora` | Fallback if CT2 missing |
| Tarteel base | `tarteel-ai/whisper-base-ar-quran` | Developer reference |

---

## 13. Key Technical Decisions

### Why Single Offline Model?
The system targets IoT/laptop deployment with no GPU. Exposing multiple models in the UI caused confusion (mis-selection, unexpected behavior, HF downloads on startup). The CT2 model is pre-quantized, fully offline, and fast enough for real-time CPU use.

### Why faster-whisper over HuggingFace Transformers?
- 3–4x faster on CPU via CTranslate2 int8 quantization
- No `forced_decoder_ids` / logits processor conflicts
- No 30s padding hallucination issues at chunk level
- Uses all available CPU cores efficiently

### Why VAD + Context Window?
- VAD provides intelligent chunking on natural ayah-boundary pauses
- Context window (1.0s pre + 0.5s post) prevents first/last word clipping without causing hallucination
- Fixed-time fallback ensures operation when VAD library unavailable

### Why Separate Guard + Viterbi Pipelines?
- **Guard Pipeline:** Fast per-chunk correction (~50ms), runs on every chunk
- **Viterbi Pipeline:** Full global alignment, runs on session finalization only
- Guard provides real-time feedback; Viterbi provides comprehensive session review

### Why Consecutive Error Gate in Qari Mode?
Whisper chunks often split a long ayah mid-word, causing isolated bad chunks that look like errors but are just boundary artifacts. Requiring 2 consecutive errors prevents interrupting the reciter mid-ayah.

---

## 14. Bug Fixes Log (v1.1.0 → v1.2.0)

| Bug | File | Fix |
|-----|------|-----|
| VAD segment deduplication using fragile counter — same segment emitted twice with wrong offset | `session_manager.py` | Use `f"{abs_start}_{abs_end}"` as segment ID |
| Over-normalization of audio (peak < 0.9 boosted to 0.9) — amplifies noise, causes hallucination | `session_manager.py` | Only boost genuinely quiet audio (peak < 0.3) to 0.6 |
| Context window missing — first word of each ayah clipped (هيحمد instead of الحمد) | `session_manager.py` | Added 1.0s pre-context + 0.5s post-context per emitted segment |
| Qari Mode triggers CORRECTING on every single error chunk — interrupts mid-ayah | `correction_engine.py` | Require 2 consecutive errors AND confidence > 0.30 before triggering |
| `confidence` not passed to `process_verdict` — gate always received default 1.0 | `realtime_streamer.py` | Pass `confidence=float(guard_result.get("confidence") or 0.0)` |
| `faster-whisper-base-ar-quran` not in model registry — fell back to default silently | `realtime_streamer.py` | Added alias entry pointing to `whisper-base-quran-lora-ct2` |
| Missing correction rules for CT2 model's consistent failures on specific words | `fyp_model/quran_guard.py` | Added 20+ `_PHRASE_FIXES` + `_TOKEN_FIXES` entries |
| UI showed multiple model choices causing confusion and unexpected HF downloads | `app.py` | Reduced `MODEL_CHOICES` to single offline model |

---

## 15. File Tree

```
hafizify/
├── app.py                        # Main Gradio app — entry point
├── realtime_streamer.py          # Streaming ASR pipeline (faster-whisper)
├── session_manager.py            # Per-session recitation state + VAD chunking
├── correction_engine.py          # Qari Mode state machine (consecutive error gate)
├── hybrid_pipeline.py            # Viterbi + LM alignment pipeline
├── live_display_formatter.py     # HTML formatting for Gradio UI
├── surah_detector.py             # BM25 + fuzzy surah auto-detection
├── error_analysis.py             # ASR error analysis utilities
├── quran_audio_provider.py       # Quran.com audio fetch + word segments
├── groq_transcriber.py           # Groq cloud API client (developer only)
├── prepare_quran_lm.py           # Language model preparation script
├── train_kenlm.ps1               # KenLM 5-gram training (PowerShell)
├── train_kenlm_python.py         # KenLM 5-gram training (Python)
├── quran_5gram.arpa              # Pre-built Quran 5-gram ARPA LM
├── quran_lm.txt                  # Quran corpus for LM training
├── requirements.txt              # Python dependencies
├── system.md                     # THIS FILE — system documentation
├── Hafizify.md                   # Original developer handbook
├── README.md                     # Project README
│
├── fyp_model/                    # Core model directory
│   ├── __init__.py
│   ├── all_ayat.json             # Full Quran ayah database (6,235 ayahs)
│   ├── quran_guard.py            # Correction + ayah matching engine
│   ├── beam_decoder.py           # Beam search decoder (Wav2Vec2, optional)
│   ├── run.py                    # CLI inference tool
│   └── weights/                  # Wav2Vec2 large-xlsr-53 model (developer fallback)
│
├── whisper-base-quran-lora-ct2/  # PRIMARY MODEL — CTranslate2 int8 (offline)
├── whisper-base-quran-lora/      # LoRA adapter weights (HF fallback path)
├── whisper-base-ar-quran/        # Tarteel-ai base (developer reference)
│
├── data/
│   ├── number_of_verses.txt
│   └── readerlist.tsv
│
├── scripts/                      # Evaluation scripts
│   ├── eval_wav2vec2_quran.py
│   ├── evaluate_whole_quran_sequence.py
│   ├── run_all_reciters.py
│   └── run_dual_test.py
│
├── audio_cache/                  # Quran.com audio clips cache (runtime)
└── recordings/                   # Session output (runtime generated)
```

---

## 16. Running the System

### Prerequisites
```bash
# Python 3.10+
# FFmpeg (for audio decoding)
# Windows: choco install ffmpeg
# 4GB RAM minimum, 8GB recommended
# No GPU required
```

### Setup
```bash
git clone https://github.com/NeuraAgency/HAFIZIFY.git
cd HAFIZIFY
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### First Run
```bash
python app.py
# Opens at http://127.0.0.1:7860
# Model loads from whisper-base-quran-lora-ct2/ on first session start
# No internet connection required
```

### One-time Setup (if patch scripts present)
```bash
python patch_quran_guard.py      # Adds CT2 correction rules to quran_guard.py
python apply_realtime_patch.py   # Patches confidence param in realtime_streamer.py
# Delete both scripts after running
```
