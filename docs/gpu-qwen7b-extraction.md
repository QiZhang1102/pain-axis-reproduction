# Stage 2 — Strict Qwen2.5-7B Pain Direction Reproduction

This stage performs **new GPU model inference**, not a reanalysis of the paper's saved results.

**Status: launcher prepared and unit-tested. Full GPU experiment not yet executed.**

## GPU requirements

Use Linux, NVIDIA CUDA, a working CUDA PyTorch environment, enough system RAM/VRAM, and persistent disk. The upstream script stores all-layer activations for every included dataset, so 7B may still require significant CPU/GPU memory. A 48–80 GB VRAM GPU is a reasonable planning choice, not a verified minimum.

## Run on a CUDA GPU (e.g. RunPod with persistent /workspace)

~~~bash
cd /workspace
git clone https://github.com/QiZhang1102/pain-axis-reproduction.git
git clone --filter=blob:none https://github.com/valen-research/Pain-axis.git Pain-axis-official
git -C Pain-axis-official checkout 4d75cd90e206ea962f7a9101e65c85efea56723b

python pain-axis-reproduction/scripts/prepare_official_extraction.py \
  --upstream-dir Pain-axis-official

cd Pain-axis-official
python -m pip install -r requirements.txt
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
~~~

If PyTorch, TransformerLens or Python versions conflict, resolve the environment problem and record the versions before altering scientific code. The original requirements are not strictly version-locked.

## Execute the original extraction algorithm

~~~bash
cd /workspace/Pain-axis-official
set -o pipefail
python scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors_qwen7b_paper.py \
  2>&1 | tee qwen7b_extraction_console.log
~~~

The helper produces a COPY of the pinned upstream script, changing exactly three configuration/runtime details: (1) only Qwen/Qwen2.5-7B-Instruct in MODELS, (2) output location under the local ./results directory, and (3) disabling deletion of the full Hugging Face cache. The original script remains untouched.

Scientific methods kept intact: author's original S1/S2/control/sadness datasets, original hidden state extraction (TransformerLens residual stream), 5-fold CV grouped by sentence set, fixed seed 42, Difference-in-Means, denoising with leading PCA components that cumulatively explain 50% of control variance, layer selection, and the original evaluation logic.

A provenance JSON file is generated in the upstream checkout, containing original and copied script SHA-256 and pinned Git revision.

## Check fresh results against published benchmarks

~~~bash
cd /workspace/Pain-axis-official
python ../pain-axis-reproduction/scripts/check_reproduced_extraction.py \
  --output-dir results/Qwen_2.5_7B_instruct
~~~

Expected files for this NEW run:
- results/Qwen_2.5_7B_instruct/activations.pt
- results/Qwen_2.5_7B_instruct/layer_curves.csv
- results/Qwen_2.5_7B_instruct/layer_curves.png
- results/Qwen_2.5_7B_instruct/summary.json
- results/Qwen_2.5_7B_instruct/final_token/pain_vectors.pt
- results/Qwen_2.5_7B_instruct/mean/pain_vectors.pt

**Published author values, NOT measurements from this project:** final-token best layer 24, mean-token best layer 26, S2 1P AUC 0.9565 at L24, and S2 3P AUC 0.9650 at L24. The checker reports differences without forcing a match.

Do not commit generated activations, model weights, credentials or other large artifacts to the public student repository. Preserve the exact upstream commit, model revision, GPU specifications, package versions, seeds and generated result metadata.

Next: run the author's similarity validation script against your newly generated activations to evaluate fear and sadness. Actual behavior steering needs the matching fine-tuned LoRA adapter and an additional GPU run.
