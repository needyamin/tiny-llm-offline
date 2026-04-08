# Tiny LLM API

**GPT-Neo 125M** behind **FastAPI**, fully **offline** at runtime: model files live on disk, no Hugging Face calls when `HF_HUB_OFFLINE=1`. Default port **22122**. Full detail: **`documentation.html`**.

---

### Hardware (sanity check)

| | Minimum | Comfortable |
|---|--------|-------------|
| **CPU** | x86_64, 2+ cores | **4+ cores** (CPU inference; default image has no GPU) |
| **RAM** | 8 GB | **16 GB** |
| **Disk** | ~10 GB free | **~15 GB+** (image + PyTorch + ~0.5–1 GB model) |

Below that: OOM, failed builds, or very slow text generation. This stack **infers** only — it does **not** train.

---

### What goes where

| Path | Purpose |
|------|---------|
| `models/gpt-neo-125m/` | Full HF snapshot (you download once — see below) |
| `data/` | Your `.txt` / `.md` / `.jsonl` / `.csv` (optional; exposed via API) |

---

### Quick start

1. **Download model** (any machine with internet), repo root:

   `pip install huggingface_hub && python scripts/download_model.py`  
   or: `huggingface-cli download EleutherAI/gpt-neo-125M --local-dir models/gpt-neo-125m`

2. **Run:** `docker compose up --build -d` → API at `http://<host>:22122`  
   (First build needs network for `pip` unless you use a pre-built image.)

3. **Optional — no Docker:** install CPU PyTorch + `pip install -r requirements.txt`, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, then `uvicorn app.main:app --host 0.0.0.0 --port 22122`.

---

### API (short)

| Method | Path | Role |
|--------|------|------|
| GET | `/health` | Model path, offline flag, threads, `data_dir` |
| POST | `/v1/generate` | Body: `prompt`, optional `max_new_tokens`, `temperature`, `top_p`, `do_sample` |
| GET | `/v1/dataset/files` | List files in `data/` |
| GET | `/v1/dataset/file?name=…` | Read one file (size capped) |

Live OpenAPI: `/docs`, `/redoc`, `/openapi.json`.

```bash
curl -s http://localhost:22122/v1/generate -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_new_tokens":64}'
```

---

### Env vars (Docker defaults)

`MODEL_PATH=/models/gpt-neo-125m` · `DATA_DIR=/data` · `HF_HUB_OFFLINE=1` · `TRANSFORMERS_OFFLINE=1` · `TORCH_NUM_THREADS=4` · `MAX_PROMPT_CHARS=6000` · `MAX_DATA_READ_BYTES=2097152`

Compose also maps `./models` and `./data` read-only and sets CPU/RAM limits — tune in `docker-compose.yml`.

---

### Troubleshooting

- **Won’t start:** `models/gpt-neo-125m` must exist with `config.json`, tokenizer, and weights (`.safetensors` or `.bin`).
- **Hub download at runtime:** Keep `HF_HUB_OFFLINE=1` and a complete local folder.
- **Slow:** Normal on CPU; lower `max_new_tokens`.

**License:** [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) — use per their model card; this repo is a thin wrapper.
