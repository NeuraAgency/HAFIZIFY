"""
Train a KenLM 5-gram ARPA model using the kenlm Python package.

This is the Windows-friendly fallback that does NOT require compiling
KenLM from source. It uses subprocess to call lmplz/build_binary if
they are on PATH, or falls back to a pure-Python n-gram ARPA generator.

Usage:
    python train_kenlm_python.py
    python train_kenlm_python.py --input quran_lm.txt --order 5
"""
import argparse
import collections
import math
import os
import shutil
import subprocess
import sys


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _train_with_cli(input_path: str, order: int, arpa_path: str, bin_path: str):
    """Use KenLM CLI tools (lmplz + build_binary)."""
    print(f"  lmplz -o {order} < {input_path} > {arpa_path}")
    with open(input_path, "r", encoding="utf-8") as fin, open(
        arpa_path, "w", encoding="utf-8"
    ) as fout:
        subprocess.run(
            ["lmplz", "-o", str(order), "--discount_fallback"],
            stdin=fin,
            stdout=fout,
            check=True,
        )
    print(f"  build_binary {arpa_path} {bin_path}")
    subprocess.run(["build_binary", arpa_path, bin_path], check=True)


def _build_arpa_python(input_path: str, order: int, arpa_path: str):
    """Build a simple ARPA n-gram model in pure Python (no smoothing)."""
    print(f"  Building {order}-gram ARPA in pure Python …")

    with open(input_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Count n-grams (with <s> and </s> markers)
    ngram_counts: dict[int, collections.Counter] = {
        n: collections.Counter() for n in range(1, order + 1)
    }
    for sent in sentences:
        tokens = sent.split()
        padded = ["<s>"] + tokens + ["</s>"]
        for n in range(1, order + 1):
            for i in range(len(padded) - n + 1):
                gram = tuple(padded[i : i + n])
                ngram_counts[n][gram] += 1

    # Write ARPA
    with open(arpa_path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(ngram_counts[n])}\n")
        f.write("\n")

        for n in range(1, order + 1):
            f.write(f"\\{n}-grams:\n")
            total = sum(ngram_counts[n].values())
            for gram, count in sorted(ngram_counts[n].items()):
                log_prob = math.log10(count / total) if total > 0 else -99
                gram_str = " ".join(gram)
                backoff = "\t0.0" if n < order else ""
                f.write(f"{log_prob:.6f}\t{gram_str}{backoff}\n")
            f.write("\n")

        f.write("\\end\\\n")

    print(f"  ARPA file written: {arpa_path}")


def _convert_arpa_to_binary(arpa_path: str, bin_path: str):
    """Try build_binary, fall back to just using ARPA directly."""
    if _has_tool("build_binary"):
        print(f"  build_binary {arpa_path} {bin_path}")
        subprocess.run(["build_binary", arpa_path, bin_path], check=True)
    else:
        # (pyctcdecode can load .arpa files, just slower startup)
        print(f"  LM path: {arpa_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--order", type=int, default=5)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input or os.path.join(base, "quran_lm.txt")
    arpa_path = os.path.join(base, "quran_5gram.arpa")
    bin_path = os.path.join(base, "quran_5gram.bin")

    if not os.path.isfile(input_path):
        sys.exit(f"Input not found: {input_path}\nRun prepare_quran_lm.py first.")

    if _has_tool("lmplz") and _has_tool("build_binary"):
        print("Using KenLM CLI tools (lmplz + build_binary)")
        _train_with_cli(input_path, args.order, arpa_path, bin_path)
    else:
        print("KenLM CLI not found — using pure-Python ARPA builder")
        _build_arpa_python(input_path, args.order, arpa_path)
        _convert_arpa_to_binary(arpa_path, bin_path)

    print("\n✓ Language model ready!")
    if os.path.isfile(bin_path):
        sz = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"  Binary : {bin_path} ({sz:.1f} MB)")
    if os.path.isfile(arpa_path):
        sz = os.path.getsize(arpa_path) / (1024 * 1024)
        print(f"  ARPA   : {arpa_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
