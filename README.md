# Tiny LLM API (GPT-Neo 125M, offline)

FastAPI service that runs **EleutherAI/gpt-neo-125M** from **local files only** (no Hugging Face download at runtime). Docker exposes port **22122**.

## Hardware requirements (check this first)

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|--------|
| **CPU** | x86_64, 2+ cores | **4+ cores** (e.g. i5/i7 class) | Inference is **CPU-only** in the default Docker image; generation is usable but not instant. |
| **RAM** | 8 GB (tight) | **16 GB** | Host RAM for OS + Docker + PyTorch + 125M model peak. |
| **Storage** | ~10 GB free | **~15 GB+ free** on SSD | Docker image + PyTorch CPU + model (~0.5–1 GB) + updates; leave headroom for logs/cache. |
| **GPU** | — | Optional | **Not required.** Default stack uses CPU PyTorch. Add CUDA builds + NVIDIA only if you extend the image. |

If you are below minimum, expect **OOM**, **failed builds**, or **very slow** responses. Fine-tuning or large datasets need **more RAM and disk** than this API-only setup.

## What this is

| Piece | Role |
|--------|------|
| **Model** | Full HF snapshot in `models/gpt-neo-125m/` (config, tokenizer, weights). |
| **Dataset** | Your files in `data/` — listed/read via API (`.txt`, `.md`, `.jsonl`, `.csv`). |
| **API** | Text generation + dataset listing/reading. **No training** — add your own scripts if you fine-tune. |

## Prerequisites (software)

- **Docker** + Docker Compose (Ubuntu server or dev PC).
- **Hardware**: see **Hardware requirements** above.

## Project layout

```
tinny-slm/
├── app/main.py           # FastAPI app
├── data/                 # Your dataset files (read-only in container)
├── models/gpt-neo-125m/  # Model snapshot (you populate — see below)
├── scripts/download_model.py
├── docker-compose.yml
├── Dockerfile
├── documentation.html    # Same topics in HTML
└── requirements.txt
```

## Setup (summary)

### 1) Get the model (needs internet once)

From the repo root:

```bash
pip install huggingface_hub
python scripts/download_model.py
```

Or:

```bash
huggingface-cli download EleutherAI/gpt-neo-125M --local-dir models/gpt-neo-125m
```

Copy the entire `models/gpt-neo-125m` folder to the offline machine if needed.

### 2) Dataset (optional)

Put files under `data/`. Allowed extensions: `.txt`, `.md`, `.jsonl`, `.csv`.

### 3) Run with Docker

```bash
docker compose up --build -d
```

- API: `http://<host>:22122`
- First **build** downloads PyTorch/pip packages (needs network unless you use a saved image).

### 4) Run without Docker (optional)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Ensure models/gpt-neo-125m exists; set HF_HUB_OFFLINE=1 for offline
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
uvicorn app.main:app --host 0.0.0.0 --port 22122
```

(Linux/macOS: `export` instead of `set`.)

## Environment variables

| Variable | Default (Docker) | Meaning |
|----------|-------------------|---------|
| `MODEL_PATH` | `/models/gpt-neo-125m` | Directory with model + tokenizer. |
| `DATA_DIR` | `/data` | Dataset directory. |
| `HF_HUB_OFFLINE` | `1` | No Hub access when `1` / `true`. |
| `TRANSFORMERS_OFFLINE` | `1` | Aligns Transformers with offline mode. |
| `TORCH_NUM_THREADS` | `4` | PyTorch CPU threads. |
| `TORCH_INTEROP_THREADS` | `1` | Inter-op threads. |
| `MAX_PROMPT_CHARS` | `6000` | Max prompt length (chars). |
| `MAX_DATA_READ_BYTES` | `2097152` | Max bytes for `/v1/dataset/file`. |

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status, `model_path`, `offline`, CUDA, threads, `data_dir`. |
| `POST` | `/v1/generate` | JSON: `prompt` (required), `max_new_tokens`, `temperature`, `top_p`, `do_sample`. |
| `GET` | `/v1/dataset/files` | List dataset files. |
| `GET` | `/v1/dataset/file?name=file.txt` | Read one file (size limit applies). |

Example:

```bash
curl -s http://localhost:22122/v1/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Hello\",\"max_new_tokens\":64}"
```

(Linux/macOS: use `\` line continuation and single quotes for JSON.)

## Docker Compose notes

- Binds: `./models` → `/models`, `./data` → `/data` (read-only).
- Limits: `cpus: 4`, `mem_limit: 12g`, `shm_size: 512mb` — tune for your host.

## Troubleshooting

- **Container exits on start**: `models/gpt-neo-125m` missing or incomplete — need `config.json`, tokenizer files, and weights (`.bin` or `.safetensors`).
- **Still tries to download**: Ensure `HF_HUB_OFFLINE=1` and files are complete locally.
- **Slow generation**: Expected on CPU; reduce `max_new_tokens` or use a GPU image + CUDA PyTorch if you add a GPU.

## License

This repo is a thin wrapper. **GPT-Neo** weights and terms are from [EleutherAI on Hugging Face](https://huggingface.co/EleutherAI/gpt-neo-125M); follow their license and model card.
