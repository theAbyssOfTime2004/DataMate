# Offline DataMate image for 2 x RTX 5090 (Blackwell / sm_120).
# Base = the official vLLM OpenAI image at the same version as deploy/modal_vllm.py and the
# local docker compose, so serving behaves the same as in development. The vLLM images of
# this line ship CUDA >= 12.8 wheels (Blackwell); confirm on a rented 5090 (docker/README.md).
# Build (from repo root, after docker/fetch_model.sh):
#   docker build -f docker/Dockerfile -t datamate:e5 .

ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0
FROM ${VLLM_IMAGE}

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DATAMATE_PROJECT_ROOT=/app \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_MODEL=qwen3-14b \
    LLM_API_KEY=EMPTY \
    LLM_DISABLE_THINKING=true

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY evals /app/evals
COPY rules /app/rules
# Same interpreter as vLLM; the package only needs light deps (duckdb, pandas, pydantic...).
RUN python3 -m pip install --no-cache-dir . \
    && python3 -c "import duckdb, vllm; from datamate.config import PROJECT_ROOT as R; assert (R / 'rules' / 'train.json').is_file(), R; print('ok', vllm.__version__, R)"

# Weights downloaded by docker/fetch_model.sh (not in git).
COPY docker/models /models
COPY docker/predict.sh /app/docker/predict.sh
RUN chmod +x /app/docker/predict.sh

ENTRYPOINT ["/app/docker/predict.sh"]
