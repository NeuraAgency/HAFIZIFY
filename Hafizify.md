# Hafizify Program Handbook

This document is a developer handbook for the Hafizify Quran ASR program. It covers architecture, runtime behavior, data/model assets, scripts, and operational details across the whole repository.

## 1) High-level overview
Hafizify is a Quran ASR system with two user-facing modes:
- Live recitation: real-time microphone streaming, chunked decoding, and session recording.
- Upload audio: batch transcription with optional beam search and Quran-aware correction.

Core ideas:
- Decode audio with Wav2Vec2 or Whisper (greedy or beam search depending on model).
- Normalize Arabic text and strip leading invocations (Ta'awwuz/Basmala) at the start for surah classification.
- Run Quran-aware correction and ayah matching (guard pipeline) on all decodes.
- Run Hybrid Viterbi alignment on guard-corrected text with surah locking and an optional start-surah prior.
- Track recitation session progress and save all artifacts.

## 2) Primary entry points
### Gradio app
- [app.py](app.py) launches the Gradio UI and manages real-time and upload flows.
- The UI exposes two tabs:
  - Live Recitation (streaming mic, overlapping chunks, session comparison)
  - Upload Audio (file-based decode, guard correction, optional beam search)

### CLI inference (packaged model)
- [fyp_model/run.py](fyp_model/run.py) is a CLI tool for single-file inference and progressive chunk decoding.
- It supports standard and progressive (chunked) recitation workflows.

## 3) Runtime architecture and data flow
### Upload audio flow (batch)
1. Audio file is loaded and resampled to 16 kHz.
2. ASR decode happens using either:
  - Wav2Vec2 CTC argmax (greedy) or beam search via pyctcdecode + KenLM.
  - Whisper generate() (greedy) or beam search using `num_beams` from the UI.
3. Raw text is normalized (Arabic normalization).
4. Guard pipeline runs Quran-aware correction and ayah matching; corrected text becomes the primary output.
5. Hybrid Viterbi alignment runs on the guard-corrected text and strips leading invocations at the start.
6. Results are shown in Gradio and a JSON report is returned (guard + Viterbi details).

### Live recitation flow (streaming)
1. Microphone audio streams into Gradio in small buffers.
2. Audio is resampled to 16 kHz and appended to a session ring buffer.
3. Overlapping chunks are extracted based on chunk duration and overlap.
4. A background worker thread processes queued chunks:
  - Decode raw ASR (Whisper can use beam search if enabled)
  - Guard correction and ayah matching
  - Register chunk results and advance ayah progression
5. On Stop, a full-session re-decode runs; guard inference and Viterbi alignment run for comparison.
6. All WAVs and JSON results are saved under recordings/.

## 4) Core modules and responsibilities
### 4.1 Quran guard (correction + matching)
- [fyp_model/quran_guard.py](fyp_model/quran_guard.py)
- Key functions:
  - `normalize_arabic()`: removes diacritics and normalizes variants.
  - `compute_cer()`, `compute_wer()`, `token_coverage()` for matching.
  - `match_ayah()` and `match_ayah_sequence()` for ayah and multi-ayah matching.
  - `apply_correction_pipeline()` orchestrates rules + matching + correction policy.
  - `guard_inference()` is the public entry point for correction + match results.

Correction modes:
- `safe`: minimal changes, display only.
- `balanced`: partial corrections; full ayah replacement only at high confidence.
- `aggressive`: full replacement at medium/high confidence.

Match scoring details:
- Adaptive thresholds vary by token count.
- Confidence is computed from CER and token coverage.
- Sequence matching can join multiple ayahs and optionally override single-ayah match.
- Applied to both Wav2Vec2 and Whisper outputs in live and upload flows.

### 4.2 Hybrid Viterbi alignment
- [hybrid_pipeline.py](hybrid_pipeline.py)
- Combines:
  - N-gram Jaccard overlap
  - RapidFuzz partial ratio edit score
  - Optional KenLM scoring
- Uses a Viterbi DP with penalties for unlikely jumps and surah boundary logic.
- Surah locking:
  - Locks after consistent high-confidence surah predictions.
  - Penalizes cross-surah jumps while locked.
- Supports a start-surah prior to reduce early mis-locking.
- Strips leading Ta'awwuz/Basmala from the start using fuzzy matching (configurable via `strip_invocations`).
- Output includes aligned ayahs with confidence and path metadata.

### 4.3 Beam search decoder
- [fyp_model/beam_decoder.py](fyp_model/beam_decoder.py)
- Uses pyctcdecode with Wav2Vec2 vocabulary alignment.
- Supports KenLM .arpa or .bin language models.
- Provides hotwords for Quranic terms.
- Note: Whisper beam search uses `generate(num_beams=...)`, not pyctcdecode.
- Functions:
  - `load_beam_decoder()` builds a decoder.
  - `decode_beam()` runs beam search decoding.
  - `decode_beam_with_alternatives()` returns top-k hypotheses.

### 4.4 Real-time streamer and session manager
- [realtime_streamer.py](realtime_streamer.py)
  - Loads Wav2Vec2 or Whisper models on demand.
  - Decodes each chunk and applies guard correction.
  - Performs full-session decode with optional Viterbi alignment and start-surah hint.
- [session_manager.py](session_manager.py)
  - Maintains the audio buffer and overlapping chunk extraction.
  - Saves chunk WAVs and the full session WAV.
  - Tracks ayah progression and merges chunk transcripts.
  - Produces a `results.json` summary.

### 4.5 Error analysis (legacy)
- [error_analysis.py](error_analysis.py)
- Loads all ayahs and supports detailed word-level diffs, but the global evaluator is disabled in favor of Hybrid Viterbi.

## 5) Models and assets
### Model folders
- [fyp_model/weights](fyp_model/weights)
  - Fine-tuned Wav2Vec2 CTC model used by the app and realtime streamer.
- [combined_model/weights](combined_model/weights)
  - Alternate Wav2Vec2 model weights (used in some scripts).
- [whisper-base-ar-quran](whisper-base-ar-quran)
  - Fine-tuned Whisper model with its own config and tokenizer.

### Quran reference data
- [fyp_model/all_ayat.json](fyp_model/all_ayat.json)
  - Full Quran ayah text used by guard and alignment pipelines.

### Language model assets
- [quran_lm.txt](quran_lm.txt)
  - Cleaned Quran text for KenLM training.
- [quran_5gram.arpa](quran_5gram.arpa)
  - 5-gram ARPA LM (used by Hybrid Viterbi and beam search).
- Optional: quran_5gram.bin (can be produced for faster load).

## 6) Data inputs
- [data/transcripts.tsv](data/transcripts.tsv)
  - TSV with columns: PATH, DURATION, TRANSCRIPT.
  - PATH uses ${DATASET_PATH} placeholder for dataset roots.
- [data/readerlist.tsv](data/readerlist.tsv)
  - List of reciter IDs and optional notes.
- [data/number_of_verses.txt](data/number_of_verses.txt)
  - A list of verse counts per surah (114 items expected).

## 7) Recordings and outputs
- [recordings](recordings)
  - Each session has a folder: <timestamp>_<id>/
  - Subfolders:
    - chunks/ chunk_####.wav
  - Files:
    - session_full.wav
    - results.json

## 8) Scripts and tooling
### Language model preparation
- [prepare_quran_lm.py](prepare_quran_lm.py)
  - Cleans quran-simple.txt and outputs quran_lm.txt.
- [train_kenlm_python.py](train_kenlm_python.py)
  - Builds a 5-gram ARPA model with KenLM CLI if available, otherwise a pure-Python fallback.
- [train_kenlm.ps1](train_kenlm.ps1)
  - PowerShell helper: runs preparation + LM training.

### Evaluation and batch tools
- [scripts/eval_wav2vec2_quran.py](scripts/eval_wav2vec2_quran.py)
  - WER/CER evaluation on a manifest CSV.
  - Inputs: model_dir, manifest, max_eval, print_samples.
- [scripts/evaluate_whole_quran_sequence.py](scripts/evaluate_whole_quran_sequence.py)
  - Runs guarded sequence-aware evaluation across many ayahs.
  - Produces text and JSON reports, with per-reciter metrics.
- [scripts/run_dual_test.py](scripts/run_dual_test.py)
  - Compares two models on the same audio file.
  - Uses optional ayah-aware evaluation.
- [scripts/run_all_reciters.py](scripts/run_all_reciters.py)
  - Batch training loop across reciters.
  - Note: contains hard-coded absolute paths that must be updated to your environment.

## 9) CLI reference: fyp_model/run.py
Key options:
- `--audio`: audio file path or name
- `--device`: cpu or cuda (auto by default)
- `--ayah_json`: path to all_ayat.json
- `--surah`, `--expected_ayah`: fixed ayah settings
- `--lookahead`, `--window_back`: ayah search window
- `--correction_mode`: safe | balanced | aggressive
- `--progress_ayah`: enable chunked progressive mode
- `--chunk_seconds`, `--chunk_overlap_seconds`: chunking parameters
- `--progress_max_cer`, `--progress_min_coverage`: progression thresholds
- `--max_progress_jump`: maximum ayah jump per chunk
- `--allow_auto_correct`: allow high-confidence auto-correct
- `--sequence_guard`, `--sequence_max_ayahs`: multi-ayah matching
- `--allow_reference_replacement`: allow full ayah replacement
- `--disable_auto_fallback`: disable fallback to full decode in progressive mode

Behavior notes:
- Progressive mode tracks ayah progression chunk-by-chunk and falls back to full decode if confidence is low.
- Non-progressive mode runs a single decode and guard correction.

## 10) Gradio UI behavior (app.py)
Live Recitation tab:
- Start button creates a session and starts background worker.
- Streaming mic sends audio chunks to `process_streaming_audio()`.
- Stop button finalizes session, runs full-session decode, and shows comparison.

Upload Audio tab:
- Optional beam search with adjustable beam width (Wav2Vec2 beam search or Whisper `num_beams`).
- Optional sequence guard and aggressive replacement toggles.
- Returns raw ASR, guard-corrected output, confidence, and a JSON report (guard + Viterbi).

## 11) Dependencies and environment
- Dependencies are listed in [requirements.txt](requirements.txt).
- Important optional packages:
  - kenlm (for LM scoring)
  - pyctcdecode (for beam search)
  - rapidfuzz (required by Hybrid Viterbi and error analysis)
- GPU: PyTorch with CUDA improves performance.

## 12) Running the program
### Install
1. Create and activate a venv.
2. Install dependencies from [requirements.txt](requirements.txt).
3. Install PyTorch separately if needed for your CUDA version.

### Run Gradio app
- Run: python app.py
- The app opens in a browser at 127.0.0.1:7860 by default.

### Enable beam search
- Ensure pyctcdecode and kenlm are installed.
- Build a LM using [train_kenlm.ps1](train_kenlm.ps1) or [train_kenlm_python.py](train_kenlm_python.py).
- Ensure quran_5gram.arpa or quran_5gram.bin exists in repo root.

## 13) Known constraints and implementation notes
- [fyp_model/run.py](fyp_model/run.py) expects a model folder named fyp_model/model, while the app uses fyp_model/weights. Ensure the path exists or update the script if needed.
- [realtime_streamer.py](realtime_streamer.py) populates eval_result as an empty dict; error_analysis is not wired into live sessions.
- [error_analysis.py](error_analysis.py) disables the global evaluator in favor of Hybrid Viterbi.
- Hybrid Viterbi strips leading Ta'awwuz/Basmala from the start using fuzzy matching; disable via `strip_invocations` if needed.
- Quran LM training expects a quran-simple.txt input file which is not tracked here; provide it or pass --input.
- [scripts/run_all_reciters.py](scripts/run_all_reciters.py) uses environment-specific absolute paths.

## 14) File-by-file reference
Top-level:
- [app.py](app.py): Gradio UI, live + upload flows, model loading, beam search toggle.
- [hybrid_pipeline.py](hybrid_pipeline.py): Hybrid Viterbi alignment pipeline.
- [realtime_streamer.py](realtime_streamer.py): Streaming model decode + guard inference.
- [session_manager.py](session_manager.py): Session buffer, chunk extraction, storage, results.
- [error_analysis.py](error_analysis.py): Legacy ASR error analysis (global matching disabled).
- [prepare_quran_lm.py](prepare_quran_lm.py): Build LM training text from quran-simple.txt.
- [train_kenlm_python.py](train_kenlm_python.py): Build 5-gram LM with CLI or Python fallback.
- [train_kenlm.ps1](train_kenlm.ps1): PowerShell helper for LM training.
- [quran_lm.txt](quran_lm.txt): Clean Quran LM corpus.
- [quran_5gram.arpa](quran_5gram.arpa): Trained 5-gram ARPA LM.
- [requirements.txt](requirements.txt): Python dependencies.

Model folders:
- [fyp_model](fyp_model): guard, CLI run script, and Wav2Vec2 weights.
- [combined_model](combined_model): alternate Wav2Vec2 weights.
- [whisper-base-ar-quran](whisper-base-ar-quran): fine-tuned Whisper model card + weights.

Data:
- [data](data): transcripts.tsv, readerlist.tsv, number_of_verses.txt.

Scripts:
- [scripts](scripts): evaluation scripts and batch training helpers.

Recordings:
- [recordings](recordings): session outputs and chunk WAVs.
