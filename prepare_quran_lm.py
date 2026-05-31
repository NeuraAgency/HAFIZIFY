"""
Prepare Quran text for KenLM language model training.

Reads quran-simple.txt (pipe-delimited, with diacritics), strips tashkeel,
removes punctuation, normalises whitespace. Outputs one clean ayah per line
to quran_lm.txt — ready for `lmplz`.

Usage:
    python prepare_quran_lm.py
    python prepare_quran_lm.py --input path/to/quran-simple.txt --output quran_lm.txt
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fyp_model"))
from quran_guard import normalize_arabic  # noqa: E402


# Extra Unicode symbols that appear in some Quran text files
_EXTRA_STRIP = re.compile(
    r"[\u06D6-\u06ED\u0610-\u061A\u064B-\u065F\u0670\u0640"
    r"\u00AB\u00BB\u200F\u200E\u202A-\u202E\uFEFF\u06DD\u06DE"
    r"۩۞۝]+",
)


def clean_line(raw: str) -> str:
    """Parse one line of quran-simple.txt and return clean Arabic."""
    parts = raw.strip().split("|", 2)
    if len(parts) < 3:
        return ""
    text = parts[2].strip()
    # Remove Quran-specific marks not caught by the generic normaliser
    text = _EXTRA_STRIP.sub("", text)
    text = normalize_arabic(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="Prepare Quran text for LM training")
    ap.add_argument(
        "--input",
        default=None,
        help="Path to quran-simple.txt (auto-detected if omitted)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output path for cleaned text (default: quran_lm.txt next to this script)",
    )
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    # Auto-detect input
    if args.input:
        input_path = args.input
    else:
        candidates = [
            os.path.join(base, "quran-simple.txt"),
            os.path.join(base, "..", "quran-simple.txt"),
            os.path.join(base, "..", "..", "quran-simple.txt"),
        ]
        input_path = next((p for p in candidates if os.path.isfile(p)), None)
        if input_path is None:
            sys.exit(
                "Could not find quran-simple.txt. "
                "Pass --input path/to/quran-simple.txt"
            )

    output_path = args.output or os.path.join(base, "quran_lm.txt")

    print(f"Reading : {os.path.abspath(input_path)}")
    print(f"Writing : {os.path.abspath(output_path)}")

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    cleaned = []
    for line in lines:
        text = clean_line(line)
        if text:
            cleaned.append(text)

    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write("\n".join(cleaned) + "\n")

    print(f"Done — {len(cleaned)} ayahs written to {output_path}")


if __name__ == "__main__":
    main()
