#!/usr/bin/env bash
# Runs INSIDE a DSMLP scipy-ml job pod, called by submit_dsmlp.sh.
# Does not modify the authors' numerical algorithms, dataset or parameters.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/dsmlp.env"

mkdir -p "$RUN_LOG_DIR" "$HF_HOME_DIR"
LOG_FILE="$RUN_LOG_DIR/pain-axis-${RUN_MODE}-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'rc=$?; echo "Exit code: $rc; log: $LOG_FILE"; exit $rc' EXIT

echo "START: $(date -u)"
echo "Run mode: $RUN_MODE"
echo "Student repo: $PROJECT_DIR_ON_DSMLP"
echo "Pinned upstream clone: $OFFICIAL_DIR_ON_DSMLP"
echo "Storage/quota inspection (quota is not necessarily reflected by df):"
df -h "$PROJECT_DIR_ON_DSMLP" "$OFFICIAL_DIR_ON_DSMLP" || true
if command -v quota >/dev/null 2>&1; then
  quota -s || true
fi
echo "CAUTION: verify your actual PERSONAL/COURSE quota with UCSD Disk Quota Service." 
echo "Some users have 10GB, while standard course allocations may be higher." 
echo "Unmodified bf16 7B checkpoint (~15GB) and activations need >10GB storage."
test -f "$PROJECT_DIR_ON_DSMLP/scripts/prepare_official_extraction.py" || {
  echo "Student repository not mounted at expected path." >&2; exit 1;
}
test -f "$OFFICIAL_DIR_ON_DSMLP/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py" || {
  echo "Original paper repository not mounted at expected path." >&2; exit 1;
}

EXPECTED_SHA=4d75cd90e206ea962f7a9101e65c85efea56723b
CURRENT_SHA="$(git -C "$OFFICIAL_DIR_ON_DSMLP" rev-parse HEAD)"
if [[ "$CURRENT_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Wrong original-repository revision: $CURRENT_SHA, expected $EXPECTED_SHA" >&2
  exit 1
fi

# Reuse the DSMLP image's CUDA-compatible torch, like the team's E-STEER job.
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install -r "$PROJECT_DIR_ON_DSMLP/dsmlp/requirements-dsmlp.txt"

export HF_HOME="$HF_HOME_DIR"
export HF_HUB_OFFLINE=0
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys
import torch
import transformers
import transformer_lens
print("Python:", sys.version)
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable; refusing CPU fallback")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM bytes:", torch.cuda.get_device_properties(0).total_memory)
PY

python -m pip freeze > "$RUN_LOG_DIR/pip-freeze-$(date -u +%Y%m%dT%H%M%SZ).txt"
nvidia-smi

# Prepare a copy from a pinned author script; never edit the original.
python "$PROJECT_DIR_ON_DSMLP/scripts/prepare_official_extraction.py" \
  --upstream-dir "$OFFICIAL_DIR_ON_DSMLP"

if [[ "$RUN_MODE" == "preflight" ]]; then
  echo "PREFLIGHT PASSED. No model loaded and no activations calculated."
  echo "To perform extraction, edit dsmlp.env: RUN_MODE=extract, and resubmit."
  exit 0
fi

if [[ "$RUN_MODE" != "extract" ]]; then
  echo "Unsupported run mode: $RUN_MODE" >&2
  exit 1
fi

cd "$OFFICIAL_DIR_ON_DSMLP"
echo "EXTRACTION START: $(date -u)"
python -u scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors_qwen7b_paper.py

echo "EXTRACTION FINISHED: $(date -u)"
python "$PROJECT_DIR_ON_DSMLP/scripts/check_reproduced_extraction.py" \
  --output-dir "$OFFICIAL_DIR_ON_DSMLP/results/Qwen_2.5_7B_instruct"

echo "Success: the NEW Qwen7B results are in $OFFICIAL_DIR_ON_DSMLP/results/Qwen_2.5_7B_instruct"
