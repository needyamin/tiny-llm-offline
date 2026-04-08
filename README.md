<div align="center">

# Tiny LLM API

**Local [GPT-Neo 125M](https://huggingface.co/EleutherAI/gpt-neo-125M) · [FastAPI](https://fastapi.tiangolo.com/) · offline inference**

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

Port **`22122`** · [Full documentation →](documentation.html)

<br/>

</div>

## Overview

Serve a tiny causal LM from **files on disk** — no Hugging Face Hub at runtime when `HF_HUB_OFFLINE=1`.  
**Inference only** (no training). Dataset files can be listed/read via the API.

## Requirements

| | Minimum | Recommended |
|:--|:--|:--|
| **CPU** | x86_64, 2+ cores | **4+** (default image is CPU; no GPU required) |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | ~10 GB free | **~15 GB+** (Docker + PyTorch + model) |

## Layout

```
models/gpt-neo-125m/   ← full HF snapshot (populate via script below)
data/                  ← optional .txt .md .jsonl .csv
app/main.py            ← API
```

## Quick start

1. **Model** — On a machine with internet (repo root):

   ```bash
   pip install huggingface_hub && python scripts/download_model.py
   ```

   Or: `huggingface-cli download EleutherAI/gpt-neo-125M --local-dir models/gpt-neo-125m`

2. **Docker** — `docker compose up --build -d` → API at `http://<host>:22122`  
   *First build needs network for dependencies unless you load a pre-built image.*

3. **Without Docker** — Install CPU PyTorch, `pip install -r requirements.txt`, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, then run `uvicorn app.main:app --host 0.0.0.0 --port 22122`.

## Endpoints

| Method | Path | Description |
|:--|:--|:--|
| GET | `/health` | Status, model path, offline flag, threads, `data_dir` |
| POST | `/v1/generate` | `prompt`; optional `max_new_tokens`, `temperature`, `top_p`, `do_sample` |
| GET | `/v1/dataset/files` | List files in `data/` |
| GET | `/v1/dataset/file?name=` | Read one file (size capped) |

Interactive: **`/docs`** · **`/redoc`** · **`/openapi.json`**

```bash
curl -s http://localhost:22122/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":64}'
```

## Environment (Docker defaults)

| Variable | Value |
|:--|:--|
| `MODEL_PATH` | `/models/gpt-neo-125m` |
| `DATA_DIR` | `/data` |
| `HF_HUB_OFFLINE` | `1` |
| `TRANSFORMERS_OFFLINE` | `1` |
| `TORCH_NUM_THREADS` | `4` |
| `MAX_PROMPT_CHARS` | `6000` |
| `MAX_DATA_READ_BYTES` | `2097152` |

Volumes: `./models` → `/models`, `./data` → `/data` (read-only). Limits in `docker-compose.yml` — adjust for your host.

## Troubleshooting

| Issue | What to check |
|:--|:--|
| Container exits | `models/gpt-neo-125m` exists with `config.json`, tokenizer, weights (`.safetensors` or `.bin`) |
| Hub access at runtime | `HF_HUB_OFFLINE=1` and a complete local model folder |
| Slow replies | Expected on CPU; reduce `max_new_tokens` |

## License

Weights and terms: **[EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M)** — follow their model card. This repository is a thin integration layer.
