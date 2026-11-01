#!/usr/bin/env bash
# Download Qwen3-14B-FP8 weights into docker/models/ (not committed).
# Run once on a machine with network before `docker build`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${ROOT}/models/Qwen3-14B-FP8"
REPO="${DATAMATE_HF_REPO:-Qwen/Qwen3-14B-FP8}"

mkdir -p "${DEST}"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${REPO}" --local-dir "${DEST}" --local-dir-use-symlinks False
elif command -v hf >/dev/null 2>&1; then
  hf download "${REPO}" --local-dir "${DEST}"
else
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${REPO}", local_dir=r"${DEST}", local_dir_use_symlinks=False)
print("downloaded ${REPO} -> ${DEST}")
PY
fi

echo "Model ready at ${DEST}"
echo "Next: docker build -f docker/Dockerfile -t datamate:e5 ."
