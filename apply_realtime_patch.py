"""Run this once: python apply_realtime_patch.py"""
import os, sys

TARGET = os.path.join(os.path.dirname(__file__), "realtime_streamer.py")
with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

# Pass confidence to process_verdict
old = """            correction_result = self.correction_engine.process_verdict(
                verdict=correction_verdict,
                raw_asr=raw_text,
                correct_ayah_text=guard_result.get("matched_ayah_text", ""),
                ayah_num=guard_result.get("matched_ayah"),
                surah_num=effective_surah,
                wrong_words=guard_result.get("wrong_words", []),
                correction_spans=guard_result.get("correction_spans", []),
            )"""

new = """            correction_result = self.correction_engine.process_verdict(
                verdict=correction_verdict,
                raw_asr=raw_text,
                correct_ayah_text=guard_result.get("matched_ayah_text", ""),
                ayah_num=guard_result.get("matched_ayah"),
                surah_num=effective_surah,
                wrong_words=guard_result.get("wrong_words", []),
                correction_spans=guard_result.get("correction_spans", []),
                confidence=float(guard_result.get("confidence") or 0.0),
            )"""

if "confidence=float(guard_result" in content:
    print("realtime_streamer.py already patched.")
elif old in content:
    content = content.replace(old, new, 1)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(content)
    print("realtime_streamer.py patched — confidence now passed to CorrectionEngine.")
else:
    print("ERROR: process_verdict call site not found — check realtime_streamer.py manually.")
    sys.exit(1)
