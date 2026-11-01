# DataMate offline Docker (E5)

Image for the competition machine: **2 × RTX 5090** (Blackwell / sm_120), **no network**, `/data` read-only, write to `/result`.

## One-time: fetch weights

Weights are **not** in git. On a machine with network:

```bash
# needs huggingface_hub (pip) or huggingface-cli
bash docker/fetch_model.sh
```

This fills `docker/models/Qwen3-14B-FP8/` (~14B FP8). Then build:

```bash
docker build -f docker/Dockerfile -t datamate:e5 .
```

Base: `vllm/vllm-openai:v0.32.0` (same vLLM as `deploy/modal_vllm.py` and the local compose). Override with `--build-arg VLLM_IMAGE=...` if the organizers require their own base image.

## Run offline locally

Mount the **split directory** (the folder that contains `questions.jsonl`, `data/`, `calendar.csv`) at `/data`, and a writable folder at `/result`:

```bash
docker run --rm -it --network none --gpus all \
  -v /path/to/public_test:/data:ro \
  -v /path/to/out:/result \
  datamate:e5
```

Entrypoint `docker/predict.sh`:

1. Starts vLLM on `127.0.0.1:8000` with `--tensor-parallel-size 2`, served name `qwen3-14b`.
2. Waits up to 15 minutes for `/v1/models`.
3. Runs `python3 -m datamate.predict --data-dir /data --out-dir /result`.
4. On vLLM start failure: writes an all-abstain `submission.jsonl` and exits 0.

Outputs under `/result`:

| Path | Content |
|---|---|
| `submission.jsonl` | One line per question (order of `questions.jsonl`) |
| `logs/<qid>.json` | Per-question trace |
| `timing.json` | Wall time, workers, timed-out count |
| `work/` | LLM / DuckDB caches (never under `/data`) |
| `vllm.log` | Server log |

Env overrides: `DATAMATE_WORKERS`, `DATAMATE_TIME_BUDGET_S` (default 18900), `DATAMATE_TP_SIZE`, `LLM_MODEL`.

## Compare two runs

Against a Modal (or previous Docker) run:

```powershell
.\.venv\Scripts\python evals/compare_runs.py `
  --a runs/20261101-054554_pipeline_llm_all `
  --b D:\path\to\docker_out `
  --split public_test
```

Reports answer agreement %, per-type diffs, and PA when `answers.jsonl` exists for `--split` (e.g. `train --subset val` via scoring the overlapping ids). Target: ≥ 97% agreement vs Modal; two Docker runs on the same machine should differ by ≤ 0.03 PA on `train` val.
