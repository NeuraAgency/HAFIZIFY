# Hafizify — AI Quran Recitation Assistant

Hafizify is an AI-powered Quran recitation assistant that performs **real-time Arabic speech recognition**, corrects recitation errors against the Quran corpus, and tracks ayah-by-ayah progress — all running locally in the browser via a Gradio interface.

---

## Features

- 🎤 **Real-time streaming ASR** — records mic in chunks, transcribes on-the-fly
- 🕌 **Quran-aware correction engine** — maps raw ASR output to the correct ayah using sequence alignment and a 5-gram language model
- 📖 **Surah/Ayah tracker** — auto-detects which surah and ayah is being recited
- 🔁 **Qari Mode** — strict recitation mode that pauses until an error is corrected
- 📂 **File upload tab** — transcribe and evaluate a pre-recorded audio file
- 🧠 **Two model options** — lightweight LoRA fine-tune or faster-whisper CTranslate2 model

---

## Models

| Model | Source | Notes |
|---|---|---|
| `whisper-base-quran-lora` | Local (`whisper-base-quran-lora/`) + HF fallback [`KheemP/whisper-base-quran-lora`](https://huggingface.co/KheemP/whisper-base-quran-lora) | Default model. Fine-tuned Whisper-base with LoRA adapter on Quran audio. |
| `faster-whisper-base-ar-quran` | Auto-downloaded from HF [`OdyAsh/faster-whisper-base-ar-quran`](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran) | CTranslate2 quantized — faster CPU inference. |

> The LoRA adapter weights (`whisper-base-quran-lora/adapter_model.safetensors`, ~2.4 MB) are committed to this repo.  
> The base Whisper model weights are **not** committed — they are downloaded automatically from HuggingFace on first run.  
> The `faster-whisper-base-ar-quran` model is **always** downloaded automatically from HuggingFace.

---

## Project Structure

```
hafizify/
├── app.py                     # Main Gradio app — entry point
├── realtime_streamer.py       # Streaming ASR pipeline
├── session_manager.py         # Per-session recitation state
├── correction_engine.py       # Quran error correction logic
├── hybrid_pipeline.py         # Viterbi + LM alignment pipeline
├── live_display_formatter.py  # HTML formatting for Gradio UI
├── surah_detector.py          # Surah auto-detection
├── error_analysis.py          # ASR error analysis utilities
├── quran_audio_provider.py    # Audio reference fetching
├── prepare_quran_lm.py        # Language model preparation script
├── train_kenlm.ps1            # KenLM 5-gram training (PowerShell)
├── train_kenlm_python.py      # KenLM 5-gram training (Python)
├── quran_5gram.arpa           # Pre-built Quran 5-gram ARPA LM
├── quran_lm.txt               # Quran corpus for LM training
├── requirements.txt           # Python dependencies
├── whisper-base-quran-lora/   # LoRA adapter weights (committed, ~2.4 MB)
├── fyp_model/                 # Quran guard + ayah data
│   ├── all_ayat.json          # Full Quran ayah database
│   ├── quran_guard.py         # Sequence guard inference
│   └── beam_decoder.py        # Beam search decoder
├── data/                      # Reference data
│   ├── number_of_verses.txt
│   └── readerlist.tsv
└── scripts/                   # Evaluation scripts
    ├── eval_wav2vec2_quran.py
    ├── evaluate_whole_quran_sequence.py
    └── run_dual_test.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/NeuraAgency/HAFIZIFY.git
cd HAFIZIFY
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Install the appropriate PyTorch CUDA build first — see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 4. Run the app

```bash
python app.py
```

Gradio will start a local server (default: `http://127.0.0.1:7860`).  
On first run, HuggingFace model weights are downloaded automatically (~300 MB for whisper-base-quran-lora base, ~75 MB for faster-whisper).

---

## Usage

### Live Recitation Tab
1. Select a **model**, **surah**, and **starting ayah**
2. Click **Start Session**
3. Recite into your microphone — transcription and correction appear in real time
4. Click **Stop** when done

### File Upload Tab
1. Upload a `.wav` or `.mp3` audio file
2. Select a model and correction options
3. Click **Transcribe** — results show raw ASR, corrected text, and matched ayah

---

## Requirements

- Python 3.9+
- FFmpeg (for audio decoding) — install via `choco install ffmpeg` (Windows) or `brew install ffmpeg` (macOS)
- 4 GB RAM minimum; 8 GB recommended for smooth real-time performance
- GPU optional but speeds up Whisper inference significantly

---

## Team Handover Notes

- **Two active models only**: `whisper-base-quran-lora` (default) and `faster-whisper-base-ar-quran`
- **No model binaries in git** (except the 2.4 MB LoRA adapter) — everything else auto-downloads from HuggingFace
- The `fyp_model/all_ayat.json` contains the full Quran ayah database used by the guard and formatter — do not delete
- `quran_5gram.arpa` is the pre-built KenLM language model — do not delete
- Session recordings and audio cache are excluded from git (`.gitignore`) and generated at runtime
