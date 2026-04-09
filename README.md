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

Serve a tiny causal LM from **weights on a Docker volume** (or a local folder without Docker). After the first download, no Hugging Face Hub at **runtime** when `HF_HUB_OFFLINE=1`.  
**Inference only** (no training). Dataset files can be listed/read via the API.

## Requirements

| | Minimum | Recommended |
|:--|:--|:--|
| **CPU** | x86_64, 2+ cores | **4+** (default image is CPU; no GPU required) |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | ~10 GB free | **~15 GB+** (Docker + PyTorch + model) |

## Layout

```
Docker volume → /models/gpt-neo-125m  ← filled on first container start
./data/             ← optional dataset (bind mount)
app/main.py         ← API
```

## Quick start (Docker)

1. **Build & run** — Image build is **small** (no model in the Dockerfile). The model is downloaded **on first container start** into a Docker volume (`tinyllm_model` → `/models`). You need **internet on first start** and **~2 GB free** where Docker stores data (often `/var/lib/docker`).

   ```bash
   docker compose up --build -d
   docker compose logs -f   # first start downloads the model; then uvicorn serves
   ```

2. **Dataset (optional)** — Files in `./data/` (mounted at `/data`).

3. **Without Docker** — `python scripts/download_model.py` → `models/gpt-neo-125m`, then `uvicorn` as before.

If a previous build failed with **no space left on device**, run `docker builder prune` / `docker system prune` to free space, then rebuild.

## Endpoints

| Method | Path | Description |
|:--|:--|:--|
| GET | `/health` | Status, model path, offline flag, threads, `data_dir` |
| POST | `/v1/generate` | `prompt`; optional `max_new_tokens`, `temperature`, `top_p`, `do_sample` |
| GET | `/v1/dataset/files` | List files in `data/` |
| GET | `/v1/dataset/file?name=` | Read one file (size capped) |

Chat UI: **`/`** (Bootstrap) · API docs: **`/docs`** · **`/redoc`** · **`/openapi.json`**

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

Volumes: `tinyllm_model` → `/models` (model files), `./data` → `/data` (read-only). Tune CPU/RAM in `docker-compose.yml`.

## Troubleshooting

| Issue | What to check |
|:--|:--|
| **`docker build` TLS timeout pulling base** | Network/CDN. **Retry** later. Default base: **AWS Public ECR** (`PYTHON_IMAGE` in `Dockerfile`). To use Docker Hub: `docker compose build --build-arg PYTHON_IMAGE=python:3.11-slim`. |
| **No space during build** | Model is **not** in the Dockerfile anymore; free disk / run `docker system prune`. First download happens at **container start**, not build. |
| **First start slow / errors** | Needs **internet** until `config.json` exists under the volume; check `docker logs`. |
| Hub at runtime | Default: `HF_HUB_OFFLINE=1` — model is already in the image; no Hub download when running. |
| Slow replies | Expected on CPU; reduce `max_new_tokens` |

## License

Weights and terms: **[EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M)** — follow their model card. This repository is a thin integration layer.
