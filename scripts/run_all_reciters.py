import subprocess, time, os

PYTHON = "D:/WORK/Projects/NeuraAgency/FYP/Hafizify/.venv/Scripts/python.exe"
TRAIN_SCRIPT = "D:/WORK/Projects/NeuraAgency/FYP/Hafizify/wav2vec2_train_single.py"
READERLIST = "D:/WORK/Projects/NeuraAgency/FYP/Hafizify/Quran/readerlist.tsv"
DATASET_PATH = "D:/WORK/Projects/NeuraAgency/FYP/Hafizify/Quran/audio_data"
BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

with open(READERLIST, encoding="utf-8") as f:
    readers = [line.strip() for line in f if line.strip() and not line.lower().startswith("reader")]

print(f"📖 Total reciters: {len(readers)}")

for i, reciter in enumerate(readers, 1):
    # Skip problematic children repeating reciters
    if "children" in reciter.lower() or "teacher" in reciter.lower(): 
        print(f"⏭ Skipping {reciter} (children repeating / partial mus-haf)")
        continue

    print(f"\n🚀 [{i}/{len(readers)}] Training: {reciter}")
    output_dir = os.path.join("D:/WORK/Projects/NeuraAgency/FYP/Hafizify/wav2vec2-quran-lora", reciter)

    cmd = [
        PYTHON,
        TRAIN_SCRIPT,
        "--dataset_path", DATASET_PATH,
        "--reciter", reciter,
        "--base_model", BASE_MODEL,
        "--output_dir", output_dir,
        "--epochs", "3",
        "--batch", "1",
        "--accum_steps", "8",
        "--max_seconds", "15",
        "--fp16"
    ]

    subprocess.run(cmd)
    print(f"✅ Finished: {reciter}")
    print("🧊 Cooling GPU for 30s...\n")
    time.sleep(30)

print("🎉 ALL RECITERS FINISHED")
