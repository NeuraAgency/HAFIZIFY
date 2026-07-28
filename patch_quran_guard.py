"""
Run this ONCE from your hafizify folder:
    python patch_quran_guard.py

It patches fyp_model/quran_guard.py with expanded correction rules
for words the CTranslate2 model consistently gets wrong.
"""
import os, re, sys

TARGET = os.path.join(os.path.dirname(__file__), "fyp_model", "quran_guard.py")

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

# ── 1. Expand _PHRASE_FIXES ────────────────────────────────────────────────
NEW_ENTRIES = """
    # Consistent failures from CTranslate2 model (observed in logs)
    # أنعمت عليهم — model outputs multiple garbled forms
    ("أن عمتا ليهم", "أنعمت عليهم"),
    ("أنا أمتى ليهم", "أنعمت عليهم"),
    ("أنا أمتى لهم", "أنعمت عليهم"),
    ("أن عمتا لهم", "أنعمت عليهم"),
    ("أن عمت ليهم", "أنعمت عليهم"),
    ("عمتا ليهم", "أنعمت عليهم"),
    ("أنعمتا ليهم", "أنعمت عليهم"),
    # غير المغضوب — model outputs أويد / أيد
    ("أويد المغضوب", "غير المغضوب"),
    ("أيد المغضوب", "غير المغضوب"),
    ("أويد المقبوب", "غير المغضوب"),
    # مالك يوم الدين — يوم gets garbled
    ("مالكيع من الدين", "مالك يوم الدين"),
    ("مالك من الدين", "مالك يوم الدين"),
    ("مالكين الدين", "مالك يوم الدين"),
    # إياك — commonly split or garbled
    ("إيا كان", "إياك"),
    ("إيا كنا", "إياك"),
    ("وأياهاك", "وإياك"),
    ("الوأيى", "وإياك"),
    # الصراط — common substitution
    ("السراط", "الصراط"),
    ("الصيراط", "الصراط"),
    ("سراط", "صراط"),
    # اهدنا — common garbling
    ("اهتنا", "اهدنا"),
    ("إهدنا", "اهدنا"),
    ("إهتنا", "اهدنا"),
    # الضالين
    ("الضالن", "الضالين"),
    ("الضالي", "الضالين"),
"""

# Find closing paren of _PHRASE_FIXES tuple and insert before it
marker = "_PHRASE_FIXES"
start = content.find(marker)
if start == -1:
    print("ERROR: _PHRASE_FIXES not found"); sys.exit(1)

# Find the closing ) of the tuple
close = content.find("\n)", start)
if close == -1:
    print("ERROR: closing ) of _PHRASE_FIXES not found"); sys.exit(1)

# Check if already patched
if "أن عمتا ليهم" in content:
    print("_PHRASE_FIXES already patched, skipping.")
else:
    content = content[:close] + NEW_ENTRIES + content[close:]
    print("_PHRASE_FIXES expanded.")

# ── 2. Expand _TOKEN_FIXES ─────────────────────────────────────────────────
NEW_TOKEN_ENTRIES = """    "السراط": "الصراط",
    "الصيرات": "الصراط",
    "اهتنا": "اهدنا",
    "إهتنا": "اهدنا",
    "المغبوب": "المغضوب",
    "المغووب": "المغضوب",
    "الضالن": "الضالين",
    "نعبدو": "نعبد",
    "وأياهاك": "وإياك",
    "مالكيع": "مالك",
    "ويللمغموب": "المغضوب",
    "ويللمغووب": "المغضوب",
"""

tok_marker = "_TOKEN_FIXES"
tok_start = content.find(tok_marker)
if tok_start == -1:
    print("ERROR: _TOKEN_FIXES not found"); sys.exit(1)

tok_close = content.find("\n}", tok_start)
if tok_close == -1:
    print("ERROR: closing } of _TOKEN_FIXES not found"); sys.exit(1)

if "المغبوب" in content:
    print("_TOKEN_FIXES already patched, skipping.")
else:
    content = content[:tok_close] + "\n" + NEW_TOKEN_ENTRIES + content[tok_close:]
    print("_TOKEN_FIXES expanded.")

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(content)

print("\nDone — fyp_model/quran_guard.py patched successfully.")
print("You can delete this script after running it.")
