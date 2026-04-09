#!/bin/sh
set -e
MODEL_DIR="${MODEL_PATH:-/models/model}"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "tinyllm: first start — downloading ${MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct} to ${MODEL_DIR} (needs internet + ~8 GB free)..."
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  export MODEL_OUT="$MODEL_DIR"
  export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct}"
  export HF_HOME="${HF_HOME:-/models/.hf}"
  mkdir -p "$MODEL_DIR" "$HF_HOME"
  python /app/scripts/download_model.py
fi

exec "$@"
