# -*- coding: utf-8 -*-
"""
Evaluate a trained Wav2Vec2 CTC model on the manifest built during training.
Computes WER and CER with Arabic normalization.

Usage:
  python eval_wav2vec2_quran.py --model_dir runs/wav2vec2-quran/final --manifest runs/wav2vec2-quran/manifest.csv --max_eval 500
"""
import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import torchaudio
    import torchaudio.functional as TAF
except Exception:
    torchaudio = None
    TAF = None

import soundfile as sf
import evaluate
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Arabic normalization (same as training/transcription)
_AR_DIACRITICS = (
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061A"
    "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F\u0670\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4\u06E7\u06E8\u06EA\u06EB\u06EC\u06ED"
)
_AR_TATWEEL = "\u0640"
_ALLOWED_CHARS_PATTERN = re.compile(r"[^\u0600-\u06FFA\u0660-\u0669\u06F0-\u06F9 ]+")
_DIACRITICS_PATTERN = re.compile("[" + _AR_DIACRITICS + "]")

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = _DIACRITICS_PATTERN.sub("", text)
    text = text.replace(_AR_TATWEEL, "")
    text = _ALLOWED_CHARS_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Audio helpers

def _resample_to_16k(waveform: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return waveform.astype(np.float32)
    if TAF is not None and torch.is_available():
        tw = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
        try:
            tw = TAF.resample(tw, orig_freq=sr, new_freq=16000)
            return tw.squeeze(0).numpy()
        except Exception:
            pass
    old_len = waveform.shape[0]
    new_len = int(old_len * 16000 / sr)
    return np.interp(np.linspace(0, old_len, new_len), np.arange(old_len), waveform).astype(np.float32)


def load_audio_16k_mono(path: str) -> np.ndarray:
    try:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _resample_to_16k(wav, sr)
    except Exception:
        if torchaudio is not None:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0).numpy().astype(np.float32)
            return _resample_to_16k(wav, sr)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="runs/wav2vec2-quran/final")
    ap.add_argument("--manifest", type=str, default="runs/wav2vec2-quran/manifest.csv")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_eval", type=int, default=None, help="Limit number of rows to evaluate")
    ap.add_argument("--print_samples", type=int, default=5, help="Number of sample predictions to print")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_dir} on device={device}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)
    model.eval()

    df = pd.read_csv(args.manifest)
    if args.max_eval is not None:
        df = df.iloc[: args.max_eval].copy()
    print(f"Evaluating {len(df)} samples from manifest: {args.manifest}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    preds = []
    refs = []

    samples_to_show = []

    for i, row in df.iterrows():
        p = row["path"]
        ref = normalize_arabic(str(row["sentence"]))
        try:
            audio = load_audio_16k_mono(p)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
            pred_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(pred_ids)[0]
            text_norm = normalize_arabic(text)
        except Exception as e:
            text_norm = ""
            samples_to_show.append({"path": p, "error": str(e)})
        preds.append(text_norm)
        refs.append(ref)
        if len(samples_to_show) < args.print_samples and text_norm:
            samples_to_show.append({"path": p, "pred": text_norm, "ref": ref})

    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    print(f"WER: {wer:.4f}\nCER: {cer:.4f}")

    print("\nSample predictions:")
    for s in samples_to_show:
        if "error" in s:
            print(f"- {s['path']}\n  Error: {s['error']}")
        else:
            print(f"- {s['path']}\n  Pred: {s['pred']}\n  Ref : {s['ref']}")


if __name__ == "__main__":
    main()
