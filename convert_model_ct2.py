"""
CTranslate2 Model Converter & LoRA Merger
-----------------------------------------
Merges a PEFT/LoRA adapter with its base Whisper model and converts it to
an optimized CTranslate2 (faster-whisper) format with target quantization
(e.g., int8_float32 for CPU or float16 for GPU).

Usage:
  python convert_model_ct2.py --base tarteel-ai/whisper-base-ar-quran \
                              --adapter whisper-base-quran-lora \
                              --output whisper-base-quran-lora-ct2 \
                              --quantization int8_float32
"""

import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to CTranslate2 format.")
    parser.add_argument("--base", type=str, default="tarteel-ai/whisper-base-ar-quran",
                        help="Base Whisper model path or HF model ID")
    parser.add_argument("--adapter", type=str, default="whisper-base-quran-lora",
                        help="LoRA adapter path or HF adapter ID")
    parser.add_argument("--merged_dir", type=str, default="./merged-quran-whisper",
                        help="Intermediate directory for merged HF model")
    parser.add_argument("--output", type=str, default="./whisper-base-quran-lora-ct2",
                        help="Final CTranslate2 output directory")
    parser.add_argument("--quantization", type=str, default="int8_float32",
                        choices=["float16", "int8", "int8_float16", "int8_float32", "float32"],
                        help="CTranslate2 quantization format")
    args = parser.parse_args()

    print(f"=== 1. Loading Base Model ({args.base}) & LoRA Adapter ({args.adapter}) ===")
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        from peft import PeftModel
    except ImportError:
        print("ERROR: transformers and peft must be installed.")
        sys.exit(1)

    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.base)
    processor = AutoProcessor.from_pretrained(args.base)

    if os.path.exists(args.adapter):
        print(f"Loading local LoRA adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, args.adapter)
    else:
        print(f"Loading HF LoRA adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(base_model, args.adapter)

    print("=== 2. Merging LoRA weights into base model ===")
    merged_model = model.merge_and_unload()

    os.makedirs(args.merged_dir, exist_ok=True)
    merged_model.save_pretrained(args.merged_dir)
    processor.save_pretrained(args.merged_dir)
    print(f"Merged model saved to {args.merged_dir}")

    print(f"=== 3. Converting merged model to CTranslate2 ({args.quantization}) ===")
    cmd = [
        "ct2-transformers-converter",
        "--model", args.merged_dir,
        "--output_dir", args.output,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", args.quantization,
        "--force"
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"✅ Conversion complete! CTranslate2 model ready at: {args.output}")
        if os.path.exists(args.merged_dir):
            shutil.rmtree(args.merged_dir)
            print(f"Cleaned up intermediate folder: {args.merged_dir}")
    else:
        print(f"❌ Conversion failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
