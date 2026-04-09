import os
import re
from pathlib import Path

# i7-7700T: 4 physical cores — match threads before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "4"))

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

THREADS = int(os.getenv("TORCH_NUM_THREADS", "4"))
INTEROP = int(os.getenv("TORCH_INTEROP_THREADS", "1"))
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(INTEROP)

# Offline: set before HF stack touches the network
def _offline_mode() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


if _offline_mode():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Local model folder (full snapshot: config, tokenizer, weights)
MODEL_PATH = os.getenv("MODEL_PATH") or str(_project_root() / "models" / "gpt-neo-125m")
DATA_DIR = Path(os.getenv("DATA_DIR") or str(_project_root() / "data")).resolve()
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "6000"))
MAX_DATA_READ = int(os.getenv("MAX_DATA_READ_BYTES", str(2 * 1024 * 1024)))
DATA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$")
ALLOWED_EXT = {".txt", ".md", ".jsonl", ".csv"}

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    mp = Path(MODEL_PATH).resolve()
    if not mp.is_dir():
        raise RuntimeError(f"MODEL_PATH is not a directory: {mp}")
    device = 0 if torch.cuda.is_available() else -1
    # Do not pass local_files_only in model_kwargs/tokenizer_kwargs — newer Transformers merges
    # them with pipeline defaults and raises "multiple values for local_files_only".
    # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE already block Hub when offline.
    pipe = pipeline(
        "text-generation",
        model=str(mp),
        tokenizer=str(mp),
        device=device,
        torch_dtype=torch.float32,
    )
    yield
    pipe = None


app = FastAPI(title="Tiny LLM API (offline-ready)", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(96, ge=1, le=256)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: bool = True


class GenerateResponse(BaseModel):
    text: str
    model: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "offline": _offline_mode(),
        "cuda": torch.cuda.is_available(),
        "cpu_threads": THREADS,
        "interop_threads": INTEROP,
        "data_dir": str(DATA_DIR),
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest):
    if pipe is None:
        raise HTTPException(503, "Model not loaded")
    out = pipe(
        body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        do_sample=body.do_sample,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    text = out[0].get("generated_text", "")
    return GenerateResponse(text=text, model=MODEL_PATH)


@app.get("/v1/dataset/files")
def list_dataset_files():
    if not DATA_DIR.is_dir():
        return {"data_dir": str(DATA_DIR), "files": [], "error": "DATA_DIR missing"}
    files = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            files.append({"name": p.name, "size": p.stat().st_size})
    return {"data_dir": str(DATA_DIR), "files": files}


@app.get("/v1/dataset/file")
def read_dataset_file(name: str):
    if not DATA_NAME_RE.match(name):
        raise HTTPException(400, "Invalid file name")
    path = (DATA_DIR / name).resolve()
    try:
        path.relative_to(DATA_DIR)
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not path.is_file() or path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(404, "File not found")
    if path.stat().st_size > MAX_DATA_READ:
        raise HTTPException(413, "File too large")
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"name": name, "size": path.stat().st_size, "text": text}

