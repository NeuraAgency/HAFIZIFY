import io
import os
import sys
import re
import argparse
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from quran_guard import (
    correct_text_rules,
    guard_inference,
    load_all_ayat_json,
    parse_surah_ayah_from_filename,
    safe_display_text,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32)


def postprocess_for_display(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def transcribe(model_dir: str, audio_np: np.ndarray, device: str) -> str:
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def _resolve_optional_ayah_map(root: str, explicit_path: str | None):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(
        [
            os.path.join(root, "all_ayat.json"),
            os.path.join(root, "fyp model", "all_ayat.json"),
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return load_all_ayat_json(path), os.path.abspath(path)
    return None, None


def _run_guard(
    raw_text: str,
    ayah_map,
    surah: int | None,
    expected_ayah: int | None,
    correction_mode: str,
    allow_auto_correct: bool,
):
    if ayah_map is not None and surah is not None and expected_ayah is not None:
        return guard_inference(
            raw_text=raw_text,
            ayah_map=ayah_map,
            surah=surah,
            expected_ayah=expected_ayah,
            correction_mode=correction_mode,
            allow_auto_correct=allow_auto_correct,
        )

    return {
        "raw_asr": raw_text,
        "corrected_text": correct_text_rules(raw_text, mode=correction_mode),
        "matched_ayah": None,
        "cer": None,
        "wer": None,
        "coverage": None,
        "confidence": None,
        "alignment_score": None,
        "confidence_level": "n/a",
        "verdict": "rule_only",
        "correction_applied": correction_mode != "safe",
    }


def main():
    parser = argparse.ArgumentParser(description="Run dual-model ASR test with optional Quran-aware correction")
    parser.add_argument("--audio", default="1.mp3", help="Audio path (default: 1.mp3)")
    parser.add_argument("--out", default="text-fym.txt", help="Output report file")
    parser.add_argument(
        "--correction_mode",
        default="balanced",
        choices=["safe", "balanced", "aggressive"],
        help="Correction policy applied to display/corrected output",
    )
    parser.add_argument("--allow_auto_correct", action="store_true", help="Enable auto-correct behavior in guard")
    parser.add_argument("--ayah_json", default=None, help="Path to all_ayat.json")
    parser.add_argument("--surah", type=int, default=None, help="Surah number for ayah-aware evaluation")
    parser.add_argument("--expected_ayah", type=int, default=None, help="Expected ayah number for ayah-aware evaluation")
    args = parser.parse_args()

    root = os.path.dirname(__file__)
    audio_path = args.audio if os.path.isabs(args.audio) else os.path.join(root, args.audio)
    combined_model = os.path.join(root, "outputs", "quran_asr_restart", "merged")
    fyp_model = os.path.join(root, "fyp model", "model")
    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)

    if not os.path.isfile(audio_path):
        raise SystemExit(f"Missing audio file: {audio_path}")
    if not os.path.isdir(combined_model):
        raise SystemExit(f"Missing combined model: {combined_model}")
    if not os.path.isdir(fyp_model):
        raise SystemExit(f"Missing fyp model: {fyp_model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_np = load_audio_16k(audio_path)

    inferred_surah, inferred_ayah = parse_surah_ayah_from_filename(audio_path)
    use_surah = args.surah if args.surah is not None else inferred_surah
    use_expected_ayah = args.expected_ayah if args.expected_ayah is not None else inferred_ayah

    ayah_map, ayah_json_used = _resolve_optional_ayah_map(root, args.ayah_json)
    ayah_eval_enabled = ayah_map is not None and use_surah is not None and use_expected_ayah is not None

    raw_combined = transcribe(combined_model, audio_np, device)
    raw_fyp = transcribe(fyp_model, audio_np, device)

    guard_combined = _run_guard(
        raw_text=raw_combined,
        ayah_map=ayah_map if ayah_eval_enabled else None,
        surah=use_surah,
        expected_ayah=use_expected_ayah,
        correction_mode=args.correction_mode,
        allow_auto_correct=args.allow_auto_correct,
    )
    guard_fyp = _run_guard(
        raw_text=raw_fyp,
        ayah_map=ayah_map if ayah_eval_enabled else None,
        surah=use_surah,
        expected_ayah=use_expected_ayah,
        correction_mode=args.correction_mode,
        allow_auto_correct=args.allow_auto_correct,
    )

    display_combined = safe_display_text(guard_combined["corrected_text"])
    display_fyp = safe_display_text(guard_fyp["corrected_text"])

    lines = []
    lines.append("=" * 88)
    lines.append("Dual Model Test Results")
    lines.append("=" * 88)
    lines.append(f"Audio: {audio_path}")
    lines.append(f"Device: {device}")
    lines.append(f"Correction mode: {args.correction_mode}")
    lines.append(f"Ayah-aware eval enabled: {ayah_eval_enabled}")
    if ayah_json_used:
        lines.append(f"Ayah JSON: {ayah_json_used}")
    lines.append(f"Surah: {use_surah}")
    lines.append(f"Expected ayah: {use_expected_ayah}")
    lines.append("")
    lines.append("[TEST 1] Combined model")
    lines.append(f"Model path: {combined_model}")
    lines.append(f"Raw: {raw_combined}")
    lines.append(f"Corrected: {guard_combined['corrected_text']}")
    lines.append(f"Display: {display_combined}")
    lines.append(
        "Guard metrics: "
        f"best_ayah={guard_combined.get('matched_ayah')} "
        f"cer={guard_combined.get('cer')} wer={guard_combined.get('wer')} "
        f"coverage={guard_combined.get('coverage')} conf={guard_combined.get('confidence')} "
        f"align={guard_combined.get('alignment_score')} "
        f"level={guard_combined.get('confidence_level')} verdict={guard_combined.get('verdict')}"
    )
    lines.append("")
    lines.append("[TEST 2] FYP model folder")
    lines.append(f"Model path: {fyp_model}")
    lines.append(f"Raw: {raw_fyp}")
    lines.append(f"Corrected: {guard_fyp['corrected_text']}")
    lines.append(f"Display: {display_fyp}")
    lines.append(
        "Guard metrics: "
        f"best_ayah={guard_fyp.get('matched_ayah')} "
        f"cer={guard_fyp.get('cer')} wer={guard_fyp.get('wer')} "
        f"coverage={guard_fyp.get('coverage')} conf={guard_fyp.get('confidence')} "
        f"align={guard_fyp.get('alignment_score')} "
        f"level={guard_fyp.get('confidence_level')} verdict={guard_fyp.get('verdict')}"
    )
    lines.append("")
    lines.append("Match check")
    lines.append(f"Raw equal: {raw_combined == raw_fyp}")
    lines.append(f"Corrected equal: {guard_combined['corrected_text'] == guard_fyp['corrected_text']}")
    lines.append(f"Display equal: {display_combined == display_fyp}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
