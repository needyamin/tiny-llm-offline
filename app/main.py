import os
import re
from pathlib import Path
from typing import List, Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.getenv("TORCH_NUM_THREADS", "2"))

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))
INTEROP = int(os.getenv("TORCH_INTEROP_THREADS", "1"))
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(INTEROP)


def _offline_mode() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")


if _offline_mode():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


STATIC_DIR = _project_root() / "static"

MODEL_PATH = os.getenv("MODEL_PATH") or str(_project_root() / "models" / "model")
MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
DATA_DIR = Path(os.getenv("DATA_DIR") or str(_project_root() / "data")).resolve()
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "6000"))
MAX_DATA_READ = int(os.getenv("MAX_DATA_READ_BYTES", str(2 * 1024 * 1024)))
DATA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$")
ALLOWED_EXT = {".txt", ".md", ".jsonl", ".csv"}

model = None
tokenizer = None
_device = None


def _load_pretrained_kw():
    if torch.cuda.is_available():
        dt = torch.bfloat16
    else:
        dt = torch.float16  # CPU: half precision, lower RAM
    kw = {"torch_dtype": dt}
    if _offline_mode():
        kw["local_files_only"] = True
    return kw


def _tok_kw():
    t = {}
    if _offline_mode():
        t["local_files_only"] = True
    return t


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, _device
    mp = Path(MODEL_PATH).resolve()
    if not mp.is_dir():
        raise RuntimeError(f"MODEL_PATH is not a directory: {mp}")
    kw = _load_pretrained_kw()
    tok = AutoTokenizer.from_pretrained(str(mp), **_tok_kw())
    mdl = AutoModelForCausalLM.from_pretrained(str(mp), **kw)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    mdl.eval()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(_device)
    tokenizer = tok
    model = mdl
    yield
    model = None
    tokenizer = None


app = FastAPI(title="Tiny LLM API (chat instruct)", lifespan=lifespan)


@app.get("/")
def chat_page():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "static/index.html missing")
    return FileResponse(index)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=512)
    temperature: float = Field(0.75, ge=0.1, le=2.0)
    top_p: float = Field(0.92, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.08, ge=1.0, le=2.0)


class ChatCompletionResponse(BaseModel):
    choices: list
    model: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS)
    max_new_tokens: int = Field(96, ge=1, le=512)
    temperature: float = Field(0.85, ge=0.1, le=2.0)
    top_p: float = Field(0.92, ge=0.0, le=1.0)
    do_sample: bool = True
    repetition_penalty: float = Field(1.18, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(3, ge=0, le=10)


class GenerateResponse(BaseModel):
    text: str
    model: str


def _generate_from_ids(input_ids: torch.Tensor, body: ChatCompletionRequest) -> str:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kw = dict(
        max_new_tokens=body.max_tokens,
        do_sample=True,
        temperature=body.temperature,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kw)
    new_tok = out[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tok, skip_special_tokens=True)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(body: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    msgs = [m.model_dump() for m in body.messages]
    if getattr(tokenizer, "chat_template", None):
        try:
            raw = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = raw if isinstance(raw, torch.Tensor) else raw["input_ids"]
        except Exception as e:
            raise HTTPException(400, f"chat_template failed: {e}") from e
    else:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=min(2048, MAX_PROMPT_CHARS))
        input_ids = enc["input_ids"]
    input_ids = input_ids.to(_device)
    reply = _generate_from_ids(input_ids, body)
    return ChatCompletionResponse(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply.strip()},
                "finish_reason": "stop",
            }
        ],
        model=MODEL_PATH,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_repo": MODEL_REPO,
        "offline": _offline_mode(),
        "cuda": torch.cuda.is_available(),
        "cpu_threads": THREADS,
        "interop_threads": INTEROP,
        "data_dir": str(DATA_DIR),
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    enc = tokenizer(
        body.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=min(2048, MAX_PROMPT_CHARS),
    )
    input_ids = enc["input_ids"].to(_device)
    gen_kw = dict(
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        do_sample=body.do_sample,
        repetition_penalty=body.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if body.no_repeat_ngram_size > 0:
        gen_kw["no_repeat_ngram_size"] = body.no_repeat_ngram_size
    with torch.no_grad():
        out = model.generate(input_ids, **gen_kw)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return GenerateResponse(text=full, model=MODEL_PATH)


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
