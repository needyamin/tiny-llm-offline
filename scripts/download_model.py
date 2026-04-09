#!/usr/bin/env python3
"""Run once on a machine with internet; copies EleutherAI/gpt-neo-125M into ./models/gpt-neo-125m for offline use."""
import os
import sys
import time

from huggingface_hub import snapshot_download

OUT = os.getenv("MODEL_OUT", os.path.join(os.path.dirname(__file__), "..", "models", "model"))
OUT = os.path.abspath(OUT)
REPO = os.getenv("MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
RETRIES = int(os.getenv("HF_DOWNLOAD_RETRIES", "5"))
DELAY = int(os.getenv("HF_DOWNLOAD_DELAY_SEC", "15"))


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Downloading {REPO} -> {OUT}")
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            snapshot_download(REPO, local_dir=OUT)
            print("Done.")
            return
        except Exception as e:
            last = e
            print(f"Attempt {attempt}/{RETRIES} failed: {e}", file=sys.stderr)
            if attempt < RETRIES:
                print(f"Retrying in {DELAY}s…", file=sys.stderr)
                time.sleep(DELAY)
    raise last


if __name__ == "__main__":
    main()
    sys.exit(0)
