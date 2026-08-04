"""
test_ct2_conversion.py
-----------------------
Side-by-side comparison: current PyTorch/transformers turbo model
(hybrid_diacritic_pipeline.py's existing path) vs. a CTranslate2-converted
version of the same model (produced by convert_model_ct2.py or a direct
`ct2-transformers-converter` run — see the comment at the top of this file
for the exact command).

Purpose: verify diacritic (harakaat) output quality survives CT2's int8
quantization BEFORE switching Combined Mode's live pipeline over to it.
This is a standalone sanity check — it does not modify
hybrid_diacritic_pipeline.py or any live code path.

Usage:
    python test_ct2_conversion.py path/to/some_recitation.wav [ct2_model_dir]

    ct2_model_dir defaults to ./whisper-turbo-quran-ct2 (matching the
    --output_dir used in the conversion command below).

Conversion command this script expects you to have already run:
    ct2-transformers-converter \\
        --model MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix \\
        --output_dir whisper-turbo-quran-ct2 \\
        --quantization int8_float32 \\
        --copy_files tokenizer.json preprocessor_config.json

What to look at in the output:
    - Do the two transcripts have the same words? (consonant-level parity —
      should be near-identical, CT2 quantization rarely changes word choice)
    - Do the two transcripts have the same diacritics (harakaat) on those
      words? This is the part actually worth scrutinizing — int8
      quantization COULD subtly change which vowel the model is most
      confident about on a borderline case, even if it never has before
      on the base model's conversion.
    - Decode time for each — this is the number that actually justifies
      doing this at all.
"""

import sys
import time

import numpy as np
import soundfile as sf


PYTORCH_MODEL_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"
DEFAULT_CT2_DIR = "whisper-turbo-quran-ct2"


def _load_audio_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def transcribe_pytorch(audio: np.ndarray) -> tuple[str, float]:
    """Current production path — same call shape as
    hybrid_diacritic_pipeline.py's transcribe_local(), but standalone so
    this script doesn't need Groq/a full Combined Mode session running."""
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

    device_available = torch.cuda.is_available()
    dtype = torch.float16 if device_available else torch.float32

    print(f"[PyTorch] Loading {PYTORCH_MODEL_ID} (device={'cuda' if device_available else 'cpu'})...")
    processor = AutoProcessor.from_pretrained(PYTORCH_MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        PYTORCH_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device_available else None,
    )
    if not device_available:
        model = model.to("cpu")

    engine = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=2,
        batch_size=1,
        device=0 if device_available else -1,
    )

    t0 = time.time()
    result = engine(
        {"raw": audio, "sampling_rate": 16000},
        generate_kwargs={"task": "transcribe", "language": "arabic"},
    )
    elapsed = time.time() - t0
    return result["text"].strip(), elapsed


def transcribe_ct2(audio: np.ndarray, ct2_dir: str) -> tuple[str, float]:
    """Candidate CT2 path — same instantiation pattern realtime_streamer.py
    already uses for the other CT2 models in _MODEL_REGISTRY (WhisperModel
    + int8_float32 on CPU / float16 on GPU)."""
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8_float32"

    print(f"[CT2] Loading {ct2_dir} (device={device}, compute_type={compute_type})...")
    model = WhisperModel(ct2_dir, device=device, compute_type=compute_type)

    t0 = time.time()
    segments, _ = model.transcribe(
        audio,
        language="ar",
        beam_size=1,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=False,
    )
    text = " ".join(s.text for s in segments).strip()
    elapsed = time.time() - t0
    return text, elapsed


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ct2_conversion.py path/to/audio.wav [ct2_model_dir]")
        sys.exit(1)

    audio_path = sys.argv[1]
    ct2_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CT2_DIR

    print(f"Loading audio: {audio_path}")
    audio = _load_audio_16k(audio_path)
    print(f"Audio: {len(audio) / 16000:.2f}s\n")

    pytorch_text, pytorch_time = transcribe_pytorch(audio)
    print(f"\n[PyTorch] {pytorch_time:.2f}s -> {pytorch_text}\n")

    ct2_text, ct2_time = transcribe_ct2(audio, ct2_dir)
    print(f"[CT2]     {ct2_time:.2f}s -> {ct2_text}\n")

    print("=" * 70)
    print(f"Speedup: {pytorch_time / max(ct2_time, 0.001):.1f}x faster with CT2")
    print("PyTorch:", pytorch_text)
    print("CT2:    ", ct2_text)
    print("(eyeball both lines above for matching words AND matching harakaat --")
    print(" no automated diff here on purpose, since a naive string compare would")
    print(" flag trivial whitespace/normalization differences as false alarms)")
    print("=" * 70)


if __name__ == "__main__":
    main()
