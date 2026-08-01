from huggingface_hub import HfApi

api = HfApi()
repo_id = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"
info = api.model_info(repo_id, files_metadata=True)
total = 0
for f in info.siblings:
    size = f.size or 0
    total += size
    print(f"{f.rfilename}\t{size}")
print(f"TOTAL_BYTES\t{total}")
