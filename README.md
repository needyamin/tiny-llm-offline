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

Serve a tiny causal LM from **weights in the Docker image** (or a local folder if you run without Docker). No Hugging Face Hub at **runtime** when `HF_HUB_OFFLINE=1`.  
**Inference only** (no training). Dataset files can be listed/read via the API.

## Requirements

| | Minimum | Recommended |
|:--|:--|:--|
| **CPU** | x86_64, 2+ cores | **4+** (default image is CPU; no GPU required) |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | ~10 GB free | **~15 GB+** (Docker + PyTorch + model) |

## Layout

```
Docker image: /models/gpt-neo-125m  ← baked in at docker build
./data/              ← optional dataset (mounted into container)
app/main.py          ← API
```

## Quick start (Docker — model inside image)

1. **Build & run** (needs **internet during `docker compose build`** to download GPT-Neo 125M into the image):

   ```bash
   docker compose up --build -d
   ```

   API: `http://<host>:22122` — no host `models/` folder required.

2. **Dataset (optional)** — Put files under `./data/` (mounted read-only at `/data`).

3. **Run without Docker** — Download the model to `models/gpt-neo-125m` (see `scripts/download_model.py`), install CPU PyTorch + `pip install -r requirements.txt`, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, then `uvicorn app.main:app --host 0.0.0.0 --port 22122`.

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

Volume: `./data` → `/data` (read-only). Model lives **inside the image** at `/models/gpt-neo-125m`. Tune CPU/RAM in `docker-compose.yml`.

## Troubleshooting

| Issue | What to check |
|:--|:--|
| **`docker build` TLS timeout pulling base** | Network/CDN. **Retry** later. Default base: **AWS Public ECR** (`PYTHON_IMAGE` in `Dockerfile`). To use Docker Hub: `docker compose build --build-arg PYTHON_IMAGE=python:3.11-slim`. |
| **Build fails downloading model** | Build must reach **Hugging Face** to bake in weights. Retry or fix proxy/firewall. |
| Container exits (old setup) | If you mounted an **empty** `./models` over the image, remove that mount — current compose only mounts `./data`. |
| Hub at runtime | Default: `HF_HUB_OFFLINE=1` — model is already in the image; no Hub download when running. |
| Slow replies | Expected on CPU; reduce `max_new_tokens` |

## License

Weights and terms: **[EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M)** — follow their model card. This repository is a thin integration layer.
