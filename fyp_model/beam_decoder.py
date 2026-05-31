"""
Beam Search Decoder with KenLM for Wav2Vec2 Quran ASR.

Replaces greedy argmax decoding with pyctcdecode beam search backed by a
KenLM n-gram language model trained on Quranic text.

Key design decisions
--------------------
* Vocab is extracted directly from the Wav2Vec2Processor in **index order**
  so it exactly matches the model's output logits dimension.
* ``<pad>`` (index 0) is treated as the CTC **blank** token.
* ``|`` (index 4) is replaced with ``" "`` (space) so pyctcdecode can
  detect word boundaries.
* Logits are converted to log-probabilities via ``log_softmax`` before
  being passed to the decoder — pyctcdecode expects log probs.

Usage
-----
>>> from beam_decoder import load_beam_decoder, decode_beam
>>> decoder = load_beam_decoder(processor, "quran_5gram.arpa")
>>> text = decode_beam(model, processor, decoder, audio_np, device)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def get_vocab_list(processor) -> List[str]:
    """Return the vocabulary as a list ordered by token index.

    pyctcdecode requires ``labels[i]`` to correspond to logits column ``i``.
    We also replace the word-boundary token ``|`` with a literal space ``" "``
    because that is how pyctcdecode detects word breaks.
    """
    # The processor stores vocab as {token: index}
    vocab: Dict[str, int] = processor.tokenizer.get_vocab()
    # Sort by index → list[str]
    sorted_tokens = sorted(vocab.items(), key=lambda kv: kv[1])
    labels = []
    for token, _idx in sorted_tokens:
        if token == "|":
            labels.append(" ")     # word-boundary → space
        else:
            labels.append(token)
    return labels


# ---------------------------------------------------------------------------
# Decoder construction
# ---------------------------------------------------------------------------

def load_beam_decoder(
    processor,
    kenlm_model_path: Optional[str] = None,
    alpha: float = 0.5,
    beta: float = 1.5,
    hotwords: Optional[List[str]] = None,
    hotword_weight: float = 10.0,
):
    """Build a ``BeamSearchDecoderCTC`` with optional KenLM.

    Parameters
    ----------
    processor : Wav2Vec2Processor
        HuggingFace processor whose tokenizer provides the vocabulary.
    kenlm_model_path : str | None
        Path to a KenLM ``.arpa`` or ``.bin`` file. If *None* the decoder
        runs without a language model (still better than greedy because of
        beam search).
    alpha : float
        LM weight (higher → trust the LM more).
    beta : float
        Word insertion bonus (higher → prefer more words / fewer merges).
    hotwords : list[str] | None
        Qur'anic keywords to boost, e.g. ``["الله", "الرحمن"]``.
    hotword_weight : float
        How much to boost hotwords (default 10.0).

    Returns
    -------
    BeamSearchDecoderCTC
    """
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        raise ImportError(
            "pyctcdecode is required for beam search decoding.\n"
            "Install with: pip install pyctcdecode"
        )

    labels = get_vocab_list(processor)

    # Resolve KenLM path
    kenlm_resolved = None
    if kenlm_model_path and os.path.isfile(kenlm_model_path):
        try:
            import kenlm
            kenlm_resolved = kenlm_model_path
        except ImportError:
            print("[Warning] kenlm is not installed. Falling back to LM-free beam search.")

    # Monkeypatch open() so pyctcdecode defaults to utf-8 on Windows
    import builtins
    _orig_open = builtins.open

    def _utf8_open(*args, **kwargs):
        if len(args) >= 2:
            mode = args[1]
        else:
            mode = kwargs.get("mode", "r")
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _orig_open(*args, **kwargs)

    builtins.open = _utf8_open
    try:
        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=kenlm_resolved,
            alpha=alpha,
            beta=beta,
        )
    finally:
        builtins.open = _orig_open

    # Store hotwords on the decoder object for later use
    decoder._hotwords = hotwords or []
    decoder._hotword_weight = hotword_weight

    return decoder


# ---------------------------------------------------------------------------
# Default hotwords (common Qur'anic terms)
# ---------------------------------------------------------------------------

QURAN_HOTWORDS: List[str] = [
    "الله",
    "الرحمن",
    "الرحيم",
    "الصراط",
    "المستقيم",
    "الحمد",
    "العالمين",
    "بسم",
    "رب",
    "يوم",
    "الدين",
    "اهدنا",
    "نعبد",
    "نستعين",
]


# ---------------------------------------------------------------------------
# Beam-search decode
# ---------------------------------------------------------------------------

def decode_beam(
    model,
    processor,
    decoder,
    audio_np: np.ndarray,
    device: str = "cpu",
    beam_width: int = 100,
) -> str:
    """Run Wav2Vec2 + beam search decoding on a single audio array.

    Parameters
    ----------
    model : Wav2Vec2ForCTC
        The fine-tuned CTC model (already on *device*).
    processor : Wav2Vec2Processor
        Corresponding processor.
    decoder : BeamSearchDecoderCTC
        Decoder built by :func:`load_beam_decoder`.
    audio_np : np.ndarray
        1-D float32 waveform at 16 kHz.
    device : str
        ``"cuda"`` or ``"cpu"``.
    beam_width : int
        Number of beams.  Higher → better quality, slower.
        Recommended: 50–200 for production, 10–30 for quick tests.

    Returns
    -------
    str
        Decoded Arabic text.
    """
    # 1. Feature extraction
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    # 2. Forward pass → logits
    with torch.no_grad():
        logits = model(input_values).logits  # (1, T, V)

    # 3. Convert to log probabilities (pyctcdecode expects log probs)
    log_probs = F.log_softmax(logits, dim=-1)

    # 4. Squeeze batch dim, move to CPU, convert to numpy
    log_probs_np = log_probs.squeeze(0).cpu().numpy()  # (T, V)

    # 5. Beam search
    hotwords = getattr(decoder, "_hotwords", [])
    hotword_weight = getattr(decoder, "_hotword_weight", 10.0)

    if hotwords:
        text = decoder.decode(
            log_probs_np,
            beam_width=beam_width,
            hotwords=hotwords,
            hotword_weight=hotword_weight,
        )
    else:
        text = decoder.decode(
            log_probs_np,
            beam_width=beam_width,
        )

    return text.strip()


def decode_beam_with_alternatives(
    model,
    processor,
    decoder,
    audio_np: np.ndarray,
    device: str = "cpu",
    beam_width: int = 100,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k beam hypotheses with scores.

    Returns
    -------
    list[tuple[str, float]]
        Each tuple is ``(text, log_probability)``.
    """
    inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_np = log_probs.squeeze(0).cpu().numpy()

    beams = decoder.decode_beams(
        log_probs_np,
        beam_width=beam_width,
    )

    results = []
    for beam in beams[:top_k]:
        text = beam[0]  # hypothesis text
        # beam format: (text, frames, logit_score, lm_score)
        am_score = beam[3] if len(beam) > 3 else 0.0
        lm_score = beam[4] if len(beam) > 4 else 0.0
        total = am_score + lm_score
        results.append((text.strip(), total))
    return results
