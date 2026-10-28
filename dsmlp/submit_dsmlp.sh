#!/usr/bin/env bash
# Launch from the DSMLP LOGIN node, never from your Mac terminal.
# See docs/dsmlp-qwen7b.md. This submits a batch job; it does not run locally.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$HERE/dsmlp.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE; copy dsmlp.env.example and set your paths." >&2
  exit 1
fi

# dsmlp.env is user-owned configuration and must not contain access tokens.
source "$ENV_FILE"

for var in PROJECT_DIR_ON_DSMLP OFFICIAL_DIR_ON_DSMLP GPU_CLASS CPUS MEM_GB GPUS RUN_MODE VENV_DIR HF_HOME_DIR RUN_LOG_DIR; do
  value="${!var:-}"
  if [[ -z "$value" || "$value" == *FILL_IN* ]]; then
    echo "Please set $var in dsmlp.env (currently $value)." >&2
    exit 1
  fi
done

if [[ "$RUN_MODE" != "preflight" && "$RUN_MODE" != "extract" ]]; then
  echo "RUN_MODE must be preflight or extract." >&2
  exit 1
fi

if ! command -v launch-scipy-ml.sh >/dev/null 2>&1; then
  echo "launch-scipy-ml.sh not found. Run this script on DSMLP's login node." >&2
  exit 1
fi

echo "Mode=$RUN_MODE GPU class=$GPU_CLASS CPUs=$CPUS RAM=${MEM_GB}GB GPUs=$GPUS"
echo "Student repo (container): $PROJECT_DIR_ON_DSMLP"
echo "Upstream repo (container): $OFFICIAL_DIR_ON_DSMLP"
echo "Confirm DSMLP queue limits and GPU class entitlement before submitting."
launch-scipy-ml.sh -c "$CPUS" -m "$MEM_GB" -g "$GPUS" -l "gpu-class=$GPU_CLASS" \
  -B -- bash "$PROJECT_DIR_ON_DSMLP/dsmlp/run_job.sh"

echo "Submitted DSMLP $RUN_MODE job. Logs will appear in:"
echo "$RUN_LOG_DIR"
