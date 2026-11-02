"""Serve the dev LLM on Modal with the same image and model as the local vLLM.

Only the model runs in the cloud: data, ETL and the pipeline stay on the local machine,
which calls this endpoint through LLM_BASE_URL / LLM_API_KEY in .env.

One-time setup (run these yourself; keys never go into chat or git):
    .venv\\Scripts\\pip install modal
    .venv\\Scripts\\modal setup
    .venv\\Scripts\\modal secret create datamate-vllm VLLM_API_KEY=<a long random string>

Deploy 4B (defaults):
    .venv\\Scripts\\modal deploy deploy/modal_vllm.py
    .venv\\Scripts\\modal app stop datamate-vllm

Deploy 14B alongside 4B (different app name, so they do not overwrite each other).
PowerShell:
    $env:DATAMATE_MODEL="Qwen/Qwen3-14B-FP8"
    $env:DATAMATE_SERVED_NAME="qwen3-14b"
    $env:DATAMATE_APP="datamate-vllm-14b"
    .venv\\Scripts\\modal deploy deploy/modal_vllm.py

Then point .env at the 14B URL with LLM_MODEL=qwen3-14b and LLM_DISABLE_THINKING=true.

GPU defaults to L40S (FP8 native, 48 GB); override with DATAMATE_GPU=L4 or H100 at deploy.
The container scales to zero after SCALEDOWN_S idle seconds, so an idle endpoint costs nothing.

Env overrides (defaults keep the original 4B deploy):
    DATAMATE_MODEL, DATAMATE_SERVED_NAME, DATAMATE_APP, DATAMATE_MAX_LEN, DATAMATE_GPU
"""

from __future__ import annotations

import os
import subprocess

import modal

MODEL = os.environ.get("DATAMATE_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
SERVED_NAME = os.environ.get("DATAMATE_SERVED_NAME", "qwen3-4b")  # must match LLM_MODEL
APP_NAME = os.environ.get("DATAMATE_APP", "datamate-vllm")
# 16k: a thinking parse (prompt up to ~6k + ~3k reasoning) does not fit in 8k.
MAX_MODEL_LEN = os.environ.get("DATAMATE_MAX_LEN", "16384")
VLLM_VERSION = "0.32.0"  # same version as the local docker compose (vllm-openai:v0.32.0)
GPU = os.environ.get("DATAMATE_GPU", "L40S")
PORT = 8000
SCALEDOWN_S = 10 * 60
MAX_CONCURRENT_REQUESTS = 64

# The vllm-openai image hides its Python from Modal, so install vLLM on a CUDA base that
# Modal provisions Python for (Modal's own vLLM recipe).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Read locally at deploy time. The container re-imports this module without the
            # local env, so it must find the same values here or it falls back to the 4B.
            "DATAMATE_MODEL": MODEL,
            "DATAMATE_SERVED_NAME": SERVED_NAME,
            "DATAMATE_MAX_LEN": str(MAX_MODEL_LEN),
        }
    )
)
hf_cache = modal.Volume.from_name("datamate-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("datamate-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    # torch.compile and CUDA graph capture are CPU-bound; Modal's default fraction of a core
    # stretches cold start to tens of minutes while the GPU idles (and bills).
    cpu=8.0,
    memory=32768,
    secrets=[modal.Secret.from_name("datamate-vllm")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    timeout=24 * 60 * 60,
    scaledown_window=SCALEDOWN_S,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve() -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_NAME,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        os.environ["VLLM_API_KEY"],
        "--enable-prefix-caching",
        # Lets a thinking request reason before the JSON schema constraint starts.
        "--reasoning-parser",
        "qwen3",
    ]
    subprocess.Popen(cmd)
