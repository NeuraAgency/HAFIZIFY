"""
evaluate_tajweed_model.py
--------------------------
Evaluates the three QDAT tajweed rule classifiers (madd_munfasil, ghunnah,
ikhfa) against a labeled test set — e.g. the QDAT dataset itself
(https://www.kaggle.com/datasets/annealdahi/quran-recitation), or a held-out
split of it that wasn't used during training.

Produces, per rule AND combined:
  - Confusion matrix (printed + saved as PNG)
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - A full sklearn classification_report

Reuses the exact same feature extraction + inference code from
tajweed_model.py so the numbers reflect real app behavior.

-------------------------------------------------------------------------
SETUP — edit these before running
-------------------------------------------------------------------------
1. Download/extract the QDAT dataset somewhere on disk. It should give you:
     - A folder of .wav audio clips
     - A CSV/XLSX file with one row per clip and a label column for each
       of the three rules (0 = incorrect, 1 = correct)

2. Set DATA_DIR and LABELS_FILE below (or pass --data_dir / --labels_file
   on the command line).

3. If the label file uses different column names than the auto-detect
   patterns below expect, either rename the columns in the CSV or edit
   COLUMN_PATTERNS below.

4. IMPORTANT: only evaluate on clips that were NOT used to train the
   models in qdat_models/. If you trained on the full QDAT set with no
   held-out split, these numbers will be optimistic (train-set accuracy,
   not a fair test). If you have a train/test split file or list of
   held-out filenames from when you trained, point HOLDOUT_IDS_FILE at it
   (one filename per line) to restrict evaluation to that subset.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    pip install scikit-learn matplotlib --break-system-packages

    python scripts/evaluate_tajweed_model.py ^
        --data_dir "C:\\path\\to\\QDAT\\audio" ^
        --labels_file "C:\\path\\to\\QDAT\\labels.csv" ^
        --out_dir "eval_results"

    # Optional: only evaluate a held-out subset
    python scripts/evaluate_tajweed_model.py ^
        --data_dir "..." --labels_file "..." ^
        --holdout_ids_file "C:\\path\\to\\test_split.txt"
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Make sure we can import tajweed_model.py and its MFCC/inference code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa

from tajweed_model import (
    RULE_LABELS,
    SAMPLE_RATE,
    _extract_mfcc,
    _run_single_rule,
)

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    raise ImportError(
        "scikit-learn is required for this script.\n"
        "Install with: pip install scikit-learn --break-system-packages"
    )

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save PNGs
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    print("[WARN] matplotlib not installed — confusion matrix plots will be "
          "skipped (metrics will still print/save). "
          "pip install matplotlib --break-system-packages to enable plots.")


# ---------------------------------------------------------------------------
# EDIT THESE DEFAULTS (or pass as CLI args — CLI args take priority)
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\path\to\QDAT\audio"          # folder of .wav clips
LABELS_FILE = r"C:\path\to\QDAT\labels.csv"  # csv/xlsx with labels
OUT_DIR = "eval_results"
HOLDOUT_IDS_FILE = None                       # optional .txt, one filename per line

# Column-name auto-detection patterns (case-insensitive substring match).
# Add to these lists if your QDAT copy uses different header names.
COLUMN_PATTERNS = {
    "filename": ["file", "wav", "audio", "path", "id", "name"],
    "madd_munfasil": ["madd", "mad", "stretch", "munfasil"],
    "ghunnah": ["ghunn", "ghonna", "ghonn", "noon", "nasal"],
    "ikhfa": ["ikhfa", "ikhfaa", "hide", "conceal"],
}


def _find_column(df: pd.DataFrame, patterns: list, exclude: list = None) -> str:
    exclude = exclude or []
    for col in df.columns:
        col_l = str(col).lower()
        if col in exclude:
            continue
        if any(p in col_l for p in patterns):
            return col
    return None


def load_labels(labels_file: str) -> pd.DataFrame:
    if labels_file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(labels_file)
    else:
        df = pd.read_csv(labels_file)

    col_map = {}
    used = []
    for key, patterns in COLUMN_PATTERNS.items():
        found = _find_column(df, patterns, exclude=used)
        if found is None:
            raise ValueError(
                f"Could not auto-detect a column for '{key}' in {labels_file}.\n"
                f"Columns found: {list(df.columns)}\n"
                f"Edit COLUMN_PATTERNS in this script to match your file's "
                f"actual header names, or rename the columns in the CSV."
            )
        col_map[key] = found
        used.append(found)

    print(f"[Labels] Detected columns: {col_map}")
    df = df.rename(columns={v: k for k, v in col_map.items()})
    return df[["filename"] + list(RULE_LABELS.keys())]


def resolve_audio_path(data_dir: str, filename: str) -> str:
    """Handle labels files that store bare filenames without extension,
    or with a different extension than what's on disk."""
    candidate = os.path.join(data_dir, filename)
    if os.path.isfile(candidate):
        return candidate

    stem = os.path.splitext(str(filename))[0]
    matches = glob.glob(os.path.join(data_dir, stem + ".*"))
    matches = [m for m in matches if m.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    if matches:
        return matches[0]

    # Last resort: recursive search (QDAT is sometimes nested in subfolders)
    matches = glob.glob(os.path.join(data_dir, "**", stem + ".*"), recursive=True)
    matches = [m for m in matches if m.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))]
    return matches[0] if matches else None


def load_holdout_ids(path: str) -> set:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def run_inference(audio_path: str) -> dict:
    """Returns {rule_tag: predicted_label (0/1)} for one audio file, using
    the exact same MFCC extraction as the live app (tajweed_model.py)."""
    audio_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    features = _extract_mfcc(audio_np)
    input_tensor = np.expand_dims(features, axis=0).astype(np.float32)

    preds = {}
    for rule_tag in RULE_LABELS:
        prob = _run_single_rule(rule_tag, input_tensor)
        preds[rule_tag] = 1 if prob >= 0.5 else 0
    return preds


def plot_confusion_matrix(cm: np.ndarray, rule_tag: str, rule_name: str, out_dir: str):
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{rule_name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Incorrect (0)", "Correct (1)"])
    ax.set_yticklabels(["Incorrect (0)", "Correct (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"confusion_matrix_{rule_tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--labels_file", default=LABELS_FILE)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--holdout_ids_file", default=HOLDOUT_IDS_FILE)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate the first N rows (for a quick smoke test)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] data_dir does not exist: {args.data_dir}")
        print("Set --data_dir to the folder containing the QDAT .wav clips.")
        sys.exit(1)
    if not os.path.isfile(args.labels_file):
        print(f"[ERROR] labels_file does not exist: {args.labels_file}")
        print("Set --labels_file to the QDAT labels CSV/XLSX.")
        sys.exit(1)

    df = load_labels(args.labels_file)

    holdout_ids = load_holdout_ids(args.holdout_ids_file)
    if holdout_ids:
        before = len(df)
        df = df[df["filename"].astype(str).isin(holdout_ids)]
        print(f"[Holdout] Restricted to {len(df)}/{before} rows via {args.holdout_ids_file}")
    else:
        print("[WARN] No --holdout_ids_file given — evaluating against the FULL "
              "labels file. If these clips were also used for training, these "
              "metrics will be optimistic, not a fair test-set score.")

    if args.limit:
        df = df.head(args.limit)

    y_true = {rule: [] for rule in RULE_LABELS}
    y_pred = {rule: [] for rule in RULE_LABELS}
    skipped = []

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        filename = getattr(row, "filename")
        audio_path = resolve_audio_path(args.data_dir, filename)
        if audio_path is None:
            skipped.append(filename)
            continue

        try:
            preds = run_inference(audio_path)
        except Exception as e:
            print(f"[WARN] Inference failed for {filename}: {e}")
            skipped.append(filename)
            continue

        for rule in RULE_LABELS:
            true_label = int(getattr(row, rule))
            y_true[rule].append(true_label)
            y_pred[rule].append(preds[rule])

        if i % 50 == 0 or i == total:
            print(f"[Progress] {i}/{total}")

    if skipped:
        print(f"[WARN] Skipped {len(skipped)} clip(s) — file not found or inference error.")
        if len(skipped) <= 20:
            print(f"  {skipped}")

    # ---- Metrics ----
    report_lines = []
    summary_rows = []

    for rule_tag, rule_name in RULE_LABELS.items():
        yt, yp = y_true[rule_tag], y_pred[rule_tag]
        if not yt:
            print(f"[SKIP] No evaluated samples for {rule_tag}")
            continue

        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        report = classification_report(
            yt, yp, labels=[0, 1],
            target_names=["Incorrect (0)", "Correct (1)"],
            zero_division=0,
        )

        section = (
            f"\n{'=' * 60}\n"
            f"{rule_name}  ({rule_tag})\n"
            f"{'=' * 60}\n"
            f"Samples evaluated: {len(yt)}\n"
            f"Accuracy:  {acc:.4f}\n"
            f"Precision: {prec:.4f}\n"
            f"Recall:    {rec:.4f}\n"
            f"F1 Score:  {f1:.4f}\n\n"
            f"Confusion Matrix (rows=true, cols=pred, order=[0,1]):\n{cm}\n\n"
            f"Classification Report:\n{report}"
        )
        print(section)
        report_lines.append(section)

        summary_rows.append({
            "rule": rule_tag,
            "rule_name": rule_name,
            "n_samples": len(yt),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        })

        plot_confusion_matrix(cm, rule_tag, rule_name, args.out_dir)

    # ---- Combined (macro-average across all 3 rules, all samples pooled) ----
    all_true = sum((y_true[r] for r in RULE_LABELS if y_true[r]), [])
    all_pred = sum((y_pred[r] for r in RULE_LABELS if y_true[r]), [])
    if all_true:
        combined = (
            f"\n{'=' * 60}\n"
            f"COMBINED (all 3 rules pooled)\n"
            f"{'=' * 60}\n"
            f"Total predictions: {len(all_true)}\n"
            f"Accuracy:  {accuracy_score(all_true, all_pred):.4f}\n"
            f"Precision: {precision_score(all_true, all_pred, zero_division=0):.4f}\n"
            f"Recall:    {recall_score(all_true, all_pred, zero_division=0):.4f}\n"
            f"F1 Score:  {f1_score(all_true, all_pred, zero_division=0):.4f}\n"
        )
        print(combined)
        report_lines.append(combined)

    # ---- Save outputs ----
    report_path = os.path.join(args.out_dir, "tajweed_eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n[Saved] Full text report -> {report_path}")

    summary_path = os.path.join(args.out_dir, "tajweed_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Summary JSON -> {summary_path}")

    summary_csv = os.path.join(args.out_dir, "tajweed_eval_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[Saved] Summary CSV -> {summary_csv}")


if __name__ == "__main__":
    main()
