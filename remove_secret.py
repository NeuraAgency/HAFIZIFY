#!/usr/bin/env python3
"""
Rewrite git history to remove the hardcoded Groq API key from groq_transcriber.py.
Run: python remove_secret.py
"""

import subprocess
import os
import sys

def main():
    # Check that we're in the right directory
    if not os.path.exists("groq_transcriber.py"):
        print("Error: Run this script from the repository root.")
        sys.exit(1)

    # Use filter-branch to scrub the API key from all commits
    print("Rewriting git history to remove the hardcoded API key...")
    print("WARNING: This will rewrite all commits that touch groq_transcriber.py")
    print("You will need to force-push after this.")

    # The key to remove (replace PASTE_YOUR_KEY_HERE with the actual key before running)
    # WARNING: Do NOT commit this file with the actual key in it!
    old_key = 'PASTE_YOUR_KEY_HERE'

    # Use Python as the filter to replace the key
    cmd = [
        "git", "filter-branch", "--force", "--tree-filter",
        f'python -c "import os; path=\'groq_transcriber.py\'; '
        f'if os.path.exists(path): '
        f'  with open(path, \'r\') as f: content = f.read(); '
        f'  new_content = content.replace(\'{old_key}\', \'# API key removed — use GROQ_API_KEY env var\'); '
        f'  if content != new_content: '
        f'    with open(path, \'w\') as f: f.write(new_content)"',
        "--", "HEAD~3..HEAD"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print("\nHistory rewritten successfully!")
    print("Run: git push --force-with-lease origin main")

if __name__ == "__main__":
    main()