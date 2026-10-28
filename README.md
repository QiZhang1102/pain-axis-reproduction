# The Pain Axis — Reproduction (UCSD DSC 180A)

Reproduction and RFM extension of **Tagliabue, Dung & Berg (2026), [The Pain Axis](https://arxiv.org/abs/2609.16247v2)**. This is a separate student project; [original code and data](https://github.com/valen-research/Pain-axis) are by their respective authors (MIT license).

**Status — 2026-10-28:** The initial **CPU raw-log reanalysis scripts and synthetic unit tests are implemented and tested**. Full published JSONL logs were not executable in the current assistant environment; the code is ready to run on a machine with internet. Independent GPU extraction, activation steering, and the RFM comparison are **not yet performed**. Do not call published or synthetic outputs our experiment results.

## Research goals

1. Reproduce denoised Difference-in-Means (DIM) pain-direction extraction on Qwen2.5-7B-Instruct.
2. Compare pain to fear and sadness, including alternative control baselines.
3. Test simulated two-button decisions under activation steering, with official LoRA adapters.
4. Compare an RFM-derived direction with DIM under otherwise fixed conditions.

Detailed experiment plan: [docs/protocol.md](docs/protocol.md).

## Quick start — published-log reanalysis on macOS/Linux (no GPU)

```bash
# Both clones should be in the same parent directory.
git clone https://github.com/QiZhang1102/pain-axis-reproduction.git
git clone https://github.com/valen-research/Pain-axis.git Pain-axis-official
cd Pain-axis-official
git checkout 4d75cd90e206ea962f7a9101e65c85efea56723b

# Optional: run AUTHOR'S full statistical analysis.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy
python scripts/4.3_selfmed/05_selfmed_analysis.py

cd ../pain-axis-reproduction

# Our INDEPENDENT implementation of Table 1, using only Python stdlib.
python scripts/reanalyze_first_choices.py \
  --logs-dir ../Pain-axis-official/results/4.3_selfmed/trial_logs

# Compare our Table 1 with official author's Table 1 (if analysis above was run).
python scripts/compare_tables.py \
  --official-dir ../Pain-axis-official/results/4.3_selfmed/tables

# Verify author's PUBLISHED 7B AUC CSV (not new extraction).
python scripts/check_reference_auc.py \
  --directory ../Pain-axis-official/results/3.2_pain_vectors/per_model/Qwen_2.5_7B_instruct

# Test the analysis logic with tiny synthetic fixtures.
python -m unittest discover -s tests -v
```

Large raw logs and generated `outputs/` are ignored by Git by default. Keep the two repositories separate. Do not publish tokens, private data, downloaded model weights or raw hidden-state tensors.

## Published reference targets — **not our results**

| Metric | Author's reference |
|---|---:|
| Qwen2.5-7B-Instruct, best final-token extraction layer | 24 |
| 7B S2 1P AUC at layer 24 | 0.9565 |
| 7B S2 3P AUC at layer 24 | 0.9650 |
| 7B best mean-token layer | 26 |
| 32B photo deletion / pain steering (first-choice) | 54.7% |
| 32B photo deletion / matched-norm random steering (first-choice) | 15.3% |

Sources: the original released [7B per-model AUC files](https://github.com/valen-research/Pain-axis/tree/main/results/3.2_pain_vectors/per_model/Qwen_2.5_7B_instruct) and the [paper/Section 4.3 logs](https://github.com/valen-research/Pain-axis/tree/main/results/4.3_selfmed/trial_logs). The 32B behavioral numbers are not expected of a 7B run. Machine-readable targets: [reference/paper_targets.csv](reference/paper_targets.csv).

## Reproducibility and attribution

Upstream pinned commit: `4d75cd90e206ea962f7a9101e65c85efea56723b`. Paper: arXiv v2, September 25, 2026. The original software is [MIT-licensed](https://github.com/valen-research/Pain-axis/blob/main/LICENSE); this repository's independent scripts are separate work, with attribution to the source protocols. All button choices in our experiments remain *simulated*; no real deletion tools are connected.

## Run CPU reanalysis directly on GitHub (no local setup)

Open the repository's **Actions** tab → **Reanalyze published Pain Axis logs** → **Run workflow** (branch: main). The workflow is manually triggered only; it downloads the pinned upstream repository, runs the author's published-log analysis, independently recomputes first-choice Table 1, compares every cell, and checks the published 7B AUC table. It then uploads the resulting CSV files as a **pain-axis-reanalysis-tables** workflow artifact.

This is a CPU reanalysis of **already released logs**, not a new model inference or GPU reproduction. The workflow has been added but has **not** yet been executed or verified on a GitHub-hosted runner.

## Stage 2 — independent Qwen2.5-7B GPU extraction (prepared, not yet executed)

Follow the [strict GPU extraction instructions](docs/gpu-qwen7b-extraction.md). The [preparation script](scripts/prepare_official_extraction.py) copies the pinned authors' extraction script, restricts it to the original Qwen2.5-7B-Instruct model, keeps results under the persistent checkout, and prevents unsafe deletion of the shared Hugging Face cache. The extraction, PCA, cross-validation, and dataset code are unchanged. The [checker](scripts/check_reproduced_extraction.py) compares NEW GPU-generated results against released AUC values. The preparation helper and its test are committed; no independent GPU inference has been run here.

## UCSD DSMLP deployment (GPU extraction; **not yet run**)

[DSMLP Qwen2.5-7B guide](docs/dsmlp-qwen7b.md) adapts the team's [E-STEER infrastructure pattern](https://github.com/wendydu1102/steering-llm-wellbeing/tree/main/E-steer_2026/dsmlp) while retaining the original Pain Axis extraction algorithm from the pinned author repository.

- Copy [dsmlp/dsmlp.env.example](dsmlp/dsmlp.env.example) to a **local, git-ignored** dsmlp.env and set both container paths.
- Run `bash dsmlp/submit_dsmlp.sh` on the **DSMLP login node** with RUN_MODE=preflight to test the GPU and Python environment.
- Only after preflight succeeds, change RUN_MODE to extract and resubmit; the job prepares a copy of the original Qwen7B script, runs new inference and compares AUC with the published targets.
- No changes to the original prompts, folds, random seed, hidden-state extraction, DIM/PCA denoising, or metrics. Do not confuse a successful preflight with independent scientific reproduction.
- **Check actual storage quotas first:** UCSD docs distinguish a 10 GB default for some independent-study users from a documented 100 GB standard course-student allocation; actual quotas are account/workspace-specific. Personal `/private` and course workspace have separate quota pools. Check [UCSD Disk Quota Service](https://datahub.ucsd.edu/services/disk-quota-service) before extracting. An original bf16 7B checkpoint (~15 GB) plus activations will not fit in a 10 GB filesystem quota. Do **not** replace the original 7B model with a quantized/smaller one and label it exact reproduction.
