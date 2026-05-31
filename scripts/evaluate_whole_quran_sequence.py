import argparse
import csv
import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


@dataclass
class EvalRow:
    reciter: str
    audio_path: str
    surah: Optional[int]
    ayah: Optional[int]
    ref: str
    raw: str
    corrected: str
    raw_cer: float
    raw_wer: float
    corrected_cer: float
    corrected_wer: float
    improvement_wer: float
    verdict: str
    confidence_level: Optional[str]


def _resolve_audio_path(audio_root: str, tsv_path: str) -> Tuple[str, str, Optional[int], Optional[int]]:
    rel = tsv_path.strip().replace("\\", "/")
    marker = "/audio_data/"
    idx = rel.find(marker)
    if idx >= 0:
        rel = rel[idx + len(marker) :]
    rel = rel.lstrip("/")

    parts = rel.split("/")
    reciter = parts[0] if parts else "unknown"
    file_name = parts[-1] if parts else rel

    local_path = os.path.join(audio_root, *parts)
    surah = None
    ayah = None
    stem = os.path.splitext(file_name)[0]
    if len(stem) >= 6 and stem[:6].isdigit():
        surah = int(stem[:3])
        ayah = int(stem[3:6])
    return local_path, reciter, surah, ayah


def _load_audio_16k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).numpy().astype(np.float32)


def _decode_raw_text(model, processor, audio_np: np.ndarray, device: str) -> str:
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def _find_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Quran ASR on many ayahs with sequence-aware guarded correction")
    parser.add_argument("--model_dir", default="fyp model/model")
    parser.add_argument("--audio_root", default="Quran/audio_data")
    parser.add_argument("--transcripts_tsv", default="Quran/transcripts.tsv")
    parser.add_argument("--all_ayat_json", default="", help="Optional explicit path to all_ayat.json")
    parser.add_argument("--out_txt", default="quran_eval_report.txt")
    parser.add_argument("--out_json", default="quran_eval_report.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--reciter", default="", help="Optional reciter folder filter (contains match)")
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--window_back", type=int, default=1)
    parser.add_argument("--sequence_max_ayahs", type=int, default=10)
    parser.add_argument(
        "--correction_mode",
        default="balanced",
        choices=["safe", "balanced", "aggressive"],
    )
    parser.add_argument(
        "--allow_reference_replacement",
        action="store_true",
        help="Allow full ayah replacement (disabled by default to preserve reciter wording).",
    )
    args = parser.parse_args()

    root = os.path.dirname(__file__)
    model_dir = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(root, args.model_dir)
    audio_root = args.audio_root if os.path.isabs(args.audio_root) else os.path.join(root, args.audio_root)
    transcripts_tsv = args.transcripts_tsv if os.path.isabs(args.transcripts_tsv) else os.path.join(root, args.transcripts_tsv)
    out_txt = args.out_txt if os.path.isabs(args.out_txt) else os.path.join(root, args.out_txt)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(root, args.out_json)

    guard_path = os.path.join(root, "fyp model")
    if guard_path not in sys.path:
        sys.path.insert(0, guard_path)

    from quran_guard import compute_cer, compute_wer, guard_inference, load_all_ayat_json, normalize_arabic

    all_ayat_json = _find_existing(
        [
            args.all_ayat_json if args.all_ayat_json else "",
            os.path.join(root, "fyp model", "all_ayat.json"),
            os.path.join(audio_root, "all_ayat.json"),
        ]
    )
    if not all_ayat_json:
        raise SystemExit("Could not find all_ayat.json")

    if not os.path.isdir(model_dir):
        raise SystemExit(f"Model folder not found: {model_dir}")
    if not os.path.isfile(transcripts_tsv):
        raise SystemExit(f"transcripts.tsv not found: {transcripts_tsv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    ayah_map = load_all_ayat_json(all_ayat_json)

    print(f"Device: {device}")
    print(f"Model: {model_dir}")
    print(f"Transcripts: {transcripts_tsv}")
    print(f"Ayah map: {all_ayat_json}")

    rows: List[EvalRow] = []
    attempted = 0
    with open(transcripts_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for item in reader:
            if args.max_samples > 0 and attempted >= args.max_samples:
                break
            tsv_path = (item.get("PATH") or "").strip()
            ref = normalize_arabic((item.get("TRANSCRIPT") or "").strip())
            if not tsv_path or not ref:
                continue

            local_audio, reciter, surah, ayah = _resolve_audio_path(audio_root, tsv_path)
            if args.reciter and args.reciter.lower() not in reciter.lower():
                continue
            if not os.path.isfile(local_audio):
                continue

            attempted += 1
            try:
                audio_np = _load_audio_16k(local_audio)
                raw = _decode_raw_text(model, processor, audio_np, device)
                guard = guard_inference(
                    raw_text=raw,
                    ayah_map=ayah_map,
                    surah=surah,
                    expected_ayah=ayah,
                    lookahead=args.lookahead,
                    window_back=args.window_back,
                    correction_mode=args.correction_mode,
                    allow_auto_correct=True,
                    preserve_reciter=not args.allow_reference_replacement,
                    use_sequence_match=True,
                    sequence_max_ayahs=args.sequence_max_ayahs,
                )
                corrected = guard["corrected_text"]

                raw_cer = float(compute_cer(raw, ref))
                raw_wer = float(compute_wer(raw, ref))
                corr_cer = float(compute_cer(corrected, ref))
                corr_wer = float(compute_wer(corrected, ref))
                rows.append(
                    EvalRow(
                        reciter=reciter,
                        audio_path=local_audio,
                        surah=surah,
                        ayah=ayah,
                        ref=ref,
                        raw=normalize_arabic(raw),
                        corrected=normalize_arabic(corrected),
                        raw_cer=raw_cer,
                        raw_wer=raw_wer,
                        corrected_cer=corr_cer,
                        corrected_wer=corr_wer,
                        improvement_wer=raw_wer - corr_wer,
                        verdict=str(guard.get("verdict") or ""),
                        confidence_level=str(guard.get("confidence_level") or ""),
                    )
                )
            except Exception as exc:
                print(f"[warn] failed {local_audio}: {exc}")

            if attempted % 10 == 0:
                print(f"Processed {attempted} samples...")

    if not rows:
        raise SystemExit("No samples evaluated. Check dataset paths and filters.")

    avg_raw_cer = mean(r.raw_cer for r in rows)
    avg_raw_wer = mean(r.raw_wer for r in rows)
    avg_corr_cer = mean(r.corrected_cer for r in rows)
    avg_corr_wer = mean(r.corrected_wer for r in rows)
    avg_wer_gain = mean(r.improvement_wer for r in rows)

    by_reciter: Dict[str, List[EvalRow]] = {}
    for r in rows:
        by_reciter.setdefault(r.reciter, []).append(r)

    reciter_summary = []
    for reciter, items in sorted(by_reciter.items(), key=lambda kv: kv[0]):
        reciter_summary.append(
            {
                "reciter": reciter,
                "count": len(items),
                "raw_wer": mean(x.raw_wer for x in items),
                "corrected_wer": mean(x.corrected_wer for x in items),
                "wer_gain": mean(x.improvement_wer for x in items),
            }
        )

    worst = sorted(rows, key=lambda x: x.corrected_wer, reverse=True)[:20]

    report_lines: List[str] = []
    report_lines.append("=" * 96)
    report_lines.append("Whole-Quran ASR Evaluation (Sequence-Aware Guard)")
    report_lines.append("=" * 96)
    report_lines.append(f"Samples evaluated: {len(rows)}")
    report_lines.append(f"Model: {model_dir}")
    report_lines.append(f"Audio root: {audio_root}")
    report_lines.append(f"Correction mode: {args.correction_mode}")
    report_lines.append(f"Preserve reciter: {not args.allow_reference_replacement}")
    report_lines.append(f"Sequence guard: True")
    report_lines.append("")
    report_lines.append("Overall metrics")
    report_lines.append(f"Raw CER: {avg_raw_cer:.4f}")
    report_lines.append(f"Raw WER: {avg_raw_wer:.4f}")
    report_lines.append(f"Corrected CER: {avg_corr_cer:.4f}")
    report_lines.append(f"Corrected WER: {avg_corr_wer:.4f}")
    report_lines.append(f"Average WER improvement: {avg_wer_gain:.4f}")
    report_lines.append("")
    report_lines.append("Per-reciter summary")
    for s in reciter_summary:
        report_lines.append(
            f"- {s['reciter']}: n={s['count']} raw_wer={s['raw_wer']:.4f} "
            f"corr_wer={s['corrected_wer']:.4f} gain={s['wer_gain']:.4f}"
        )
    report_lines.append("")
    report_lines.append("Worst corrected-WER samples")
    for idx, row in enumerate(worst, 1):
        report_lines.append(
            f"{idx:02d}. {row.reciter} {os.path.basename(row.audio_path)} "
            f"surah={row.surah} ayah={row.ayah} corr_wer={row.corrected_wer:.4f} "
            f"raw_wer={row.raw_wer:.4f} verdict={row.verdict}"
        )
        report_lines.append(f"    REF: {row.ref[:180]}")
        report_lines.append(f"    RAW: {row.raw[:180]}")
        report_lines.append(f"    COR: {row.corrected[:180]}")

    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    json_payload = {
        "config": {
            "model_dir": model_dir,
            "audio_root": audio_root,
            "transcripts_tsv": transcripts_tsv,
            "all_ayat_json": all_ayat_json,
            "samples": len(rows),
            "correction_mode": args.correction_mode,
            "preserve_reciter": not args.allow_reference_replacement,
            "sequence_max_ayahs": args.sequence_max_ayahs,
        },
        "metrics": {
            "raw_cer": avg_raw_cer,
            "raw_wer": avg_raw_wer,
            "corrected_cer": avg_corr_cer,
            "corrected_wer": avg_corr_wer,
            "wer_gain": avg_wer_gain,
        },
        "per_reciter": reciter_summary,
        "worst_samples": [asdict(x) for x in worst],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved report: {out_txt}")
    print(f"Saved metrics JSON: {out_json}")


if __name__ == "__main__":
    main()
