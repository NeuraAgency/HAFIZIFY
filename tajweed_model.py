"""
tajweed_model.py — Loads the three QDAT tajweed rule classifiers
(madd_munfasil, ghunnah, ikhfa) and runs inference on a 16kHz mono
audio array, in the same MFCC format used during training.
"""

import os
import numpy as np
import librosa
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "qdat_models")

MAX_PAD_LEN = 200
N_MFCC = 26
SAMPLE_RATE = 16000

RULE_LABELS = {
    "madd_munfasil": "Separate Stretching (Madd Munfasil)",
    "ghunnah": "Tight Noon (Ghunnah)",
    "ikhfa": "Concealment (Ikhfa)",
}

_interpreters = {}  # lazy-loaded cache: rule_tag -> tf.lite.Interpreter


def _load_interpreter(rule_tag: str):
    if rule_tag in _interpreters:
        return _interpreters[rule_tag]

    model_path = os.path.join(MODELS_DIR, f"{rule_tag}.tflite")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Tajweed model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    _interpreters[rule_tag] = interpreter
    return interpreter


def _extract_mfcc(audio_np: np.ndarray) -> np.ndarray:
    """Same preprocessing as training: 26 MFCCs, padded/truncated to 200 timesteps."""
    mfcc = librosa.feature.mfcc(y=audio_np.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    features = mfcc.T  # (time, features)

    if features.shape[0] < MAX_PAD_LEN:
        pad_amount = MAX_PAD_LEN - features.shape[0]
        features = np.pad(features, ((0, pad_amount), (0, 0)), mode="constant")
    else:
        features = features[:MAX_PAD_LEN, :]

    return features.astype(np.float32)


def _run_single_rule(rule_tag: str, input_tensor: np.ndarray) -> float:
    """Returns the raw sigmoid probability (1 = correct) for one rule."""
    interpreter = _load_interpreter(rule_tag)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]["index"])[0][0]
    return float(prob)


def classify_tajweed(audio_np: np.ndarray) -> dict:
    """
    Run all three tajweed rule classifiers on a 16kHz mono float32 audio array.

    Returns a dict like:
    {
        "madd_munfasil": {"rule_name": "Separate Stretching (Madd Munfasil)",
                           "correct": True, "confidence": 87.3},
        "ghunnah": {...},
        "ikhfa": {...},
    }
    """
    try:
        features = _extract_mfcc(audio_np)
        input_tensor = np.expand_dims(features, axis=0).astype(np.float32)  # (1, 200, 26)
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}

    results = {}
    for rule_tag, rule_name in RULE_LABELS.items():
        try:
            prob = _run_single_rule(rule_tag, input_tensor)
            results[rule_tag] = {
                "rule_name": rule_name,
                "correct": prob >= 0.5,
                "confidence": round(prob * 100, 2),
            }
        except Exception as e:
            results[rule_tag] = {"rule_name": rule_name, "error": str(e)}

    return results
