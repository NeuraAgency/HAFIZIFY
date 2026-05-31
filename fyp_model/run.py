import argparse
import io
import os
import re
import sys

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from quran_guard import (
    guard_inference,
    load_all_ayat_json,
    parse_surah_ayah_from_filename,
    normalize_arabic,
    safe_display_text,
)


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


import librosa

def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = librosa.load(path, sr=16000, mono=True)
    return wav.astype(np.float32)


def _iter_audio_chunks(audio: np.ndarray, chunk_seconds: float, overlap_seconds: float, sr: int = 16000):
    chunk_size = max(1, int(chunk_seconds * sr))
    overlap = max(0, int(overlap_seconds * sr))
    step = max(1, chunk_size - overlap)

    idx = 0
    start = 0
    n = len(audio)
    while start < n:
        end = min(n, start + chunk_size)
        yield idx, start, end, audio[start:end]
        if end >= n:
            break
        idx += 1
        start += step


def postprocess_for_display(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _decode_raw_text(model, processor, audio_np: np.ndarray, device: str) -> str:
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return normalize_arabic(processor.batch_decode(pred_ids)[0])


def resolve_audio_path(audio_value: str, script_dir: str) -> str:
    candidates = [
        audio_value,
        os.path.join(os.getcwd(), audio_value),
        os.path.join(script_dir, audio_value),
        os.path.join(script_dir, "inputs", audio_value),
        os.path.join(os.path.dirname(script_dir), audio_value),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(audio_value)


def main():
    ap = argparse.ArgumentParser(description="Run Quran ASR inference with packaged model")
    ap.add_argument("--audio", required=False, help="Path or filename of audio (.mp3/.wav/.flac/.ogg)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--ayah_json", default=None, help="Path to all_ayat.json for ayah-aware guard")
    ap.add_argument("--surah", type=int, default=None, help="Force surah number")
    ap.add_argument("--expected_ayah", type=int, default=None, help="Force expected ayah")
    ap.add_argument("--lookahead", type=int, default=2, help="Ayah lookahead window")
    ap.add_argument("--window_back", type=int, default=1, help="How many previous ayahs to include in matching window")
    ap.add_argument(
        "--correction_mode",
        type=str,
        default="balanced",
        choices=["safe", "balanced", "aggressive"],
        help="Correction policy: safe (minimal), balanced (default), aggressive (full replacement on medium/high confidence).",
    )
    ap.add_argument("--progress_ayah", action="store_true", help="Enable chunked expected-ayah progression for long recitations")
    ap.add_argument("--chunk_seconds", type=float, default=12.0, help="Chunk size in seconds for progressive mode")
    ap.add_argument("--chunk_overlap_seconds", type=float, default=1.0, help="Chunk overlap in seconds for progressive mode")
    ap.add_argument("--progress_max_cer", type=float, default=1.10, help="Max CER to allow ayah progression in chunk mode")
    ap.add_argument("--progress_min_coverage", type=float, default=0.20, help="Min coverage to allow ayah progression in chunk mode")
    ap.add_argument("--max_progress_jump", type=int, default=1, help="Maximum ayah jump per accepted chunk unless confidence is very high")
    ap.add_argument("--allow_auto_correct", action="store_true", help="Enable high-confidence ayah auto-correct")
    ap.add_argument(
        "--sequence_guard",
        action="store_true",
        help="Enable contiguous ayah-sequence matching for long recitations.",
    )
    ap.add_argument(
        "--sequence_max_ayahs",
        type=int,
        default=12,
        help="Maximum contiguous ayahs considered in sequence matching.",
    )
    ap.add_argument(
        "--allow_reference_replacement",
        action="store_true",
        help="Allow full ayah replacement when confidence is high (default is strict reciter-preserving mode).",
    )
    ap.add_argument(
        "--disable_auto_fallback",
        action="store_true",
        help="Disable fallback to full-pass decode when progressive chunk confidence is consistently low.",
    )
    args = ap.parse_args()

    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, "model")
    if not os.path.isdir(model_dir):
        # Fallback to 'weights' folder if 'model' doesn't exist
        model_dir = os.path.join(script_dir, "weights")
    
    if not os.path.isdir(model_dir):
        raise SystemExit(f"Model folder not found: {model_dir}")

    audio_value = args.audio
    if not audio_value:
        audio_value = input("Enter audio path or filename (e.g., 1.mp3): ").strip()
    if not audio_value:
        raise SystemExit("No audio file provided.")

    try:
        audio_path = resolve_audio_path(audio_value, script_dir)
    except FileNotFoundError:
        raise SystemExit(
            f"Audio file not found: {audio_value}\n"
            f"Tip: provide full path, place file in '{os.path.join(script_dir, 'inputs')}', "
            f"or pass only filename if file is in the same folder."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ayah_map = None
    ayah_json_path = args.ayah_json
    if not ayah_json_path:
        local_ayah = os.path.join(script_dir, "all_ayat.json")
        if os.path.isfile(local_ayah):
            ayah_json_path = local_ayah
    if ayah_json_path:
        ayah_map = load_all_ayat_json(ayah_json_path)

    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    audio = load_audio_16k(audio_path)
    file_surah, file_ayah = parse_surah_ayah_from_filename(audio_path)
    use_surah = args.surah if args.surah is not None else file_surah
    use_expected_ayah = args.expected_ayah if args.expected_ayah is not None else file_ayah
    preserve_reciter = not args.allow_reference_replacement

    if args.progress_ayah and ayah_map is not None and use_surah is not None and use_expected_ayah is not None:
        current_ayah = use_expected_ayah
        raw_chunks = []
        corrected_chunks = []
        chunk_cers = []
        chunk_wers = []
        chunk_confidences = []
        chunk_alignments = []
        low_conf_chunks = 0
        total_chunks = 0
        print(
            f"Progressive ayah mode: surah={use_surah}, start_ayah={use_expected_ayah}, "
            f"chunk={args.chunk_seconds}s, overlap={args.chunk_overlap_seconds}s"
        )
        for idx, start, end, chunk in _iter_audio_chunks(audio, args.chunk_seconds, args.chunk_overlap_seconds):
            chunk_raw = _decode_raw_text(model, processor, chunk, device)
            g = guard_inference(
                raw_text=chunk_raw,
                ayah_map=ayah_map,
                surah=use_surah,
                expected_ayah=current_ayah,
                lookahead=args.lookahead,
                window_back=args.window_back,
                correction_mode=args.correction_mode,
                allow_auto_correct=args.allow_auto_correct,
                preserve_reciter=preserve_reciter,
                use_sequence_match=args.sequence_guard,
                sequence_max_ayahs=args.sequence_max_ayahs,
            )
            total_chunks += 1
            conf_val = float(g.get("confidence") or 0.0)
            if g.get("verdict") == "error" or conf_val < 0.20:
                low_conf_chunks += 1
            raw_chunks.append(g["raw_asr"])
            corrected_chunks.append(g["corrected_text"])
            if g.get("cer") is not None:
                chunk_cers.append(float(g["cer"]))
            if g.get("wer") is not None:
                chunk_wers.append(float(g["wer"]))
            if g.get("confidence") is not None:
                chunk_confidences.append(float(g["confidence"]))
            if g.get("alignment_score") is not None:
                chunk_alignments.append(float(g["alignment_score"]))
            start_s = start / 16000.0
            end_s = end / 16000.0
            print(
                f"  chunk#{idx:02d} [{start_s:6.2f}-{end_s:6.2f}s] "
                f"exp={current_ayah} best={g['matched_ayah']} "
                f"cer={g['cer']} wer={g.get('wer')} cov={g['coverage']} "
                f"conf={g.get('confidence')} align={g.get('alignment_score')} "
                f"level={g.get('confidence_level')} verdict={g['verdict']}"
            )
            print(f"    raw: {safe_display_text(g['raw_asr'])[:120]}")
            print(f"    corrected: {safe_display_text(g['corrected_text'])[:120]}")
            can_advance = False
            if g["matched_ayah"] is not None:
                if g["verdict"] in ("ok", "minor"):
                    can_advance = True
                elif (
                    g["cer"] is not None
                    and g["coverage"] is not None
                    and float(g["cer"]) <= float(args.progress_max_cer)
                    and float(g["coverage"]) >= float(args.progress_min_coverage)
                ):
                    can_advance = True

            if can_advance:
                proposed = max(current_ayah, int(g["matched_ayah"]) + 1)
                max_allowed = current_ayah + max(1, int(args.max_progress_jump))
                if proposed > max_allowed and float(g.get("confidence") or 0.0) < 0.92:
                    proposed = max_allowed
                current_ayah = proposed

        avg_cer = (sum(chunk_cers) / len(chunk_cers)) if chunk_cers else None
        avg_wer = (sum(chunk_wers) / len(chunk_wers)) if chunk_wers else None
        avg_conf = (sum(chunk_confidences) / len(chunk_confidences)) if chunk_confidences else None
        avg_align = (sum(chunk_alignments) / len(chunk_alignments)) if chunk_alignments else None
        guard = {
            "raw_asr": " ".join(raw_chunks),
            "corrected_text": " ".join(corrected_chunks),
            "matched_ayah": current_ayah - 1,
            "cer": round(avg_cer, 4) if avg_cer is not None else None,
            "wer": round(avg_wer, 4) if avg_wer is not None else None,
            "coverage": None,
            "confidence": round(avg_conf, 4) if avg_conf is not None else None,
            "alignment_score": round(avg_align, 4) if avg_align is not None else None,
            "confidence_level": None,
            "verdict": "progressive",
            "correction_applied": args.allow_auto_correct,
        }

        low_conf_ratio = (float(low_conf_chunks) / float(total_chunks)) if total_chunks else 1.0
        if not args.disable_auto_fallback and low_conf_ratio >= 0.60:
            print(
                "Progressive confidence is too low "
                f"({low_conf_chunks}/{total_chunks} low-confidence chunks). "
                "Falling back to full-pass guarded decode."
            )
            full_raw = _decode_raw_text(model, processor, audio, device)
            guard = guard_inference(
                raw_text=full_raw,
                ayah_map=None,
                surah=None,
                expected_ayah=None,
                lookahead=args.lookahead,
                window_back=args.window_back,
                correction_mode=args.correction_mode,
                allow_auto_correct=args.allow_auto_correct,
                preserve_reciter=preserve_reciter,
                use_sequence_match=False,
                sequence_max_ayahs=args.sequence_max_ayahs,
            )
            guard["verdict"] = f"fallback_fullpass_rule_only:{guard.get('verdict')}"
    else:
        raw_text = _decode_raw_text(model, processor, audio, device)

        guard = guard_inference(
            raw_text=raw_text,
            ayah_map=ayah_map,
            surah=use_surah,
            expected_ayah=use_expected_ayah,
            lookahead=args.lookahead,
            window_back=args.window_back,
            correction_mode=args.correction_mode,
            allow_auto_correct=args.allow_auto_correct,
            preserve_reciter=preserve_reciter,
            use_sequence_match=args.sequence_guard,
            sequence_max_ayahs=args.sequence_max_ayahs,
        )
    display_text = safe_display_text(guard["corrected_text"])

    print("Model:", model_dir)
    print("Audio:", audio_path)
    print("Device:", device)
    print("Raw (for error detection):", guard["raw_asr"])
    print("Corrected (guarded):", guard["corrected_text"])
    print("Display:", display_text)
    if ayah_map is not None and (
        guard.get("matched_ayah") is not None
        or guard.get("cer") is not None
        or guard.get("wer") is not None
    ):
        print(
            "Match:",
            f"surah={use_surah}",
            f"expected_ayah={use_expected_ayah}",
            f"sequence={guard.get('is_sequence_match')}",
            f"start_ayah={guard.get('matched_start_ayah')}",
            f"best_ayah={guard['matched_ayah']}",
            f"cer={guard['cer']}",
            f"wer={guard.get('wer')}",
            f"coverage={guard['coverage']}",
            f"confidence={guard.get('confidence')}",
            f"align={guard.get('alignment_score')}",
            f"level={guard.get('confidence_level')}",
            f"verdict={guard['verdict']}",
            f"correction_applied={guard['correction_applied']}",
        )
    elif ayah_map is not None:
        print("Match: skipped (fallback rule-only mode for stable reciter-preserving correction)")


if __name__ == "__main__":
    main()
