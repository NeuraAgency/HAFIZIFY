import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_id = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"
target = os.path.join(BASE_DIR, "whisper-l-v3-turbo-quran-lora-dataset-mix")

print(f"Downloading {repo_id} -> {target}")
path = snapshot_download(
    repo_id=repo_id,
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"DONE: {path}")
