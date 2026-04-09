#!/bin/sh
set -e
MODEL_DIR=/models/gpt-neo-125m
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "tinyllm: first start — downloading GPT-Neo 125M into Docker volume (needs internet + ~2 GB free where Docker stores volumes)..."
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  export MODEL_OUT="$MODEL_DIR"
  export HF_HOME="${HF_HOME:-/models/.hf}"
  mkdir -p "$MODEL_DIR" "$HF_HOME"
  python /app/scripts/download_model.py
fi

exec "$@"
