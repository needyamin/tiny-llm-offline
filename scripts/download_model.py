#!/usr/bin/env python3
"""Run once on a machine with internet; copies EleutherAI/gpt-neo-125M into ./models/gpt-neo-125m for offline use."""
import os
import sys

from huggingface_hub import snapshot_download

OUT = os.getenv("MODEL_OUT", os.path.join(os.path.dirname(__file__), "..", "models", "gpt-neo-125m"))
OUT = os.path.abspath(OUT)
REPO = os.getenv("MODEL_REPO", "EleutherAI/gpt-neo-125M")


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"Downloading {REPO} -> {OUT}")
    snapshot_download(REPO, local_dir=OUT)
    print("Done. Copy the folder to your offline machine under models/gpt-neo-125m")


if __name__ == "__main__":
    main()
    sys.exit(0)
