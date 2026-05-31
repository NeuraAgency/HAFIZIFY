# ============================================================================
# Train a 5-gram KenLM language model from Quran text
# ============================================================================
#
# PREREQUISITES:
#   pip install pyctcdecode kenlm
#
# OPTION A — If you have KenLM command-line tools (Linux / WSL / conda):
#   1. Install KenLM:
#       conda install -c conda-forge kenlm   (easiest)
#       OR build from https://github.com/kpu/kenlm
#   2. Run this script or manually:
#       lmplz -o 5 < quran_lm.txt > quran_5gram.arpa
#       build_binary quran_5gram.arpa quran_5gram.bin
#
# OPTION B — Pure Python (Windows-friendly, no compilation needed):
#   Just run: python train_kenlm_python.py
#   (This script is created below alongside this file)
#
# ============================================================================

Write-Host "=== Step 1: Preparing clean Quran text ===" -ForegroundColor Cyan
python prepare_quran_lm.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Step 2: Training KenLM (Python fallback) ===" -ForegroundColor Cyan
python train_kenlm_python.py
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Done! ===" -ForegroundColor Green
Write-Host "LM binary: quran_5gram.bin"
Write-Host "You can now start the Gradio app with beam search enabled."
