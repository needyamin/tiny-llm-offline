# Override: docker compose build --build-arg PYTHON_IMAGE=python:3.11-slim
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.11-slim
FROM ${PYTHON_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV MODEL_PATH=/models/gpt-neo-125m \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_NUM_THREADS=4 \
    TORCH_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    MAX_PROMPT_CHARS=6000
EXPOSE 22122

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "22122"]
