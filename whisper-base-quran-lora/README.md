---
library_name: transformers
license: mit
language:
  - ar
tags:
  - automatic-speech-recognition
  - audio
  - whisper
  - lora
  - peft
  - quran
  - arabic-diacritics
base_model: tarteel-ai/whisper-base-ar-quran
datasets:
  - quran-ayat-speech-text   # compiled from quran.ksu.edu.sa (see “Training Data”)
metrics:
  - wer
pretty_name: Whisper-Base Qurʾān (LoRA)
---

# Whisper-Base Qurʾān LoRA 🕋📖

Low-rank‐adaptation (LoRA) fine-tune of **`tarteel-ai/whisper-base-ar-quran`**
for Arabic Qurʾān recitation (tilâwah).  
Provides **diacritic-sensitive** ASR with a **test WER ≈ 5.98 %**, beating:

| model | WER ↓ | Δ vs ours |
|-------|-------|----------|
| **`KheemP/whisper-base-quran-lora`** | **0.0598** | — |
| tarteel-ai/whisper-base-ar-quran | 0.073 | **-1.3 ×** |
| tarteel-ai/whisper-tiny-ar-quran | 0.096 | **-1.6 ×** |
| NVIDIA FastConformer large *(NeMo)* | ≈ 0.069 | **-1.2 ×** |

*(All scores measured on the same 610-ayah hold-out set, with no text
normalisation – tashkīl included).*

---

## Quick start

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import torch, soundfile as sf

base_id  = "tarteel-ai/whisper-base-ar-quran"
lora_id  = "KheemP/whisper-base-quran-lora"

# load model+processor
model  = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float16)
model  = PeftModel.from_pretrained(model, lora_id)
proc   = WhisperProcessor.from_pretrained(base_id)

# transcribe an mp3 -> text
audio, _ = sf.read("my_recitation.mp3")
inputs   = proc(audio, sampling_rate=16_000, return_tensors="pt").to(model.device)
pred_ids = model.generate(**inputs)
print(proc.decode(pred_ids[0]))
````

> ⚠️ *This repo only stores the **LoRA adapter (\~2 MB)**.
> The code above automatically downloads the original Whisper base model and
> injects the adapter.*

---

## Model details

|                          |                            |
| ------------------------ | -------------------------- |
| **Back-bone**            | Whisper Base (77 M params) |
| **LoRA rank / α / drop** | 8 / 16 / 0.05              |
| **Trainable params**     | 0.59 M (0.8 %)             |
| **Epochs**               | 5                          |
| **Batch / grad-accum**   | 2×4 (effective = 8)        |
| **LR / sched**           | 5 · 10⁻⁴, constant         |
| **Mixed-precision**      | fp16                       |
| **Hardware**             | single NVIDIA A100 40 GB   |

### Target modules

`q_proj, k_proj, v_proj, out_proj` in both encoder & decoder self-attn and
encoder-cross-attn blocks.

---

## Training data

* **Dataset:** 446 k MP3 ayāt scraped from [https://quran.ksu.edu.sa](https://quran.ksu.edu.sa), resampled
  to 16 kHz and paired with canonical text from *all\_ayat.json*.
* **Filtering:**

  * keep ≤ 30 s duration (→ 6091 ayāt)
  * pick shortest recording per ayah
  * 90 / 10 split ⇒ 5481 train / 610 test
* **Reciters:** 37; round-robin sampling ensures balanced voices.

---

## Evaluation

* **Metric:** jiwer WER with **no normalisation** (diacritics matter).
* **Result:** 0.0598 on the 610-ayah test split (95 % CI ± 0.003).

---

## Intended use & limitations

Designed for **speech-to-text of Qurʾān recitations in Modern Standard Arabic**.
Not expected to work for:

* conversational Arabic, dialects or non-Qurʾānic liturgy
* noisy, low-quality microphones
* verses longer than 30 seconds

---

## Citation

```bibtex
@software{quran_whisper_lora_2024,
  author       = {Kheem Dharmani},
  title        = {Whisper-Base Qurʾān LoRA Adapter},
  year         = 2024,
  url          = {https://huggingface.co/KheemP/whisper-base-quran-lora}
}
```

---

## Licence

*Back-bone* weights under MIT (same as Whisper).
Dataset sourced from the public domain.
Adapter itself released under **MIT**.

---
