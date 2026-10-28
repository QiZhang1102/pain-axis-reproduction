#!/usr/bin/env python3
"""Create a single-model runner from the authors' pinned Pain Axis extraction code.

Only three campaign/runtime changes: MODELS=Qwen2.5-7B-Instruct,
save to local ./results, and avoid deleting the entire Hugging Face cache.
The original extraction, denoising and validation methods are untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_SHA = "4d75cd90e206ea962f7a9101e65c85efea56723b"
SOURCE_REL = Path("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
OUTPUT_REL = Path("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors_qwen7b_paper.py")


def replace_once(source: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, lambda match: replacement, source, flags=re.M | re.S)
    if count != 1:
        raise ValueError("Expected one " + label + " in pinned upstream; found " + str(count))
    return result


def prepare(original: str) -> str:
    source = replace_once(
        original,
        r'^MODELS = \[\n.*?^\]\n',
        'MODELS = [("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct")]\n',
        "MODELS list",
    )
    source = replace_once(
        source,
        r'^# On RunPod, /workspace is the persistent network volume\.\n.*?^    print\("Local environment - saving to ./results"\)\n',
        'OUTPUT_DIR = Path("results")\nprint("Saving to ./results; keep the author repository in persistent storage")\n',
        "output directory",
    )
    source = replace_once(
        source,
        r'^def clear_gpu_and_cache\(\):\n.*?\n(?=^def save_progress\()',
        'def clear_gpu_and_cache():\n'
        '    """Clear unused CUDA allocations without deleting the shared HF cache."""\n'
        '    gc.collect()\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.empty_cache()\n'
        '        torch.cuda.synchronize()\n\n\n',
        "whole-cache cleanup",
    )
    compile(source, str(OUTPUT_REL), "exec")
    return source


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.upstream_dir.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_SHA:
        parser.error("Pinned upstream revision required: " + UPSTREAM_SHA + ", found " + head)
    original_bytes = (root / SOURCE_REL).read_bytes()
    output_bytes = prepare(original_bytes.decode("utf-8")).encode("utf-8")
    dest = root / OUTPUT_REL
    if dest.exists() and dest.read_bytes() != output_bytes:
        parser.error("Refusing to overwrite a different existing runner: " + str(dest))
    dest.write_bytes(output_bytes)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": head,
        "original_script": str(SOURCE_REL),
        "original_sha256": sha256(original_bytes),
        "generated_runner": str(OUTPUT_REL),
        "generated_sha256": sha256(output_bytes),
        "changes": [
            "only Qwen/Qwen2.5-7B-Instruct in MODELS",
            "save relative to ./results",
            "disable whole Hugging Face cache deletion",
        ],
        "scientific_methods_modified": False,
        "status": "prepared_only_not_executed",
    }
    metadata = root / "qwen7b_extraction_provenance.json"
    metadata.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Created:", dest)
    print("Provenance:", metadata)
    print("5 folds, seed 42, DIM, 50% PCA denoising, original hidden-state hooks retained.")
    print("This only prepares the GPU run: extraction has not executed.")


if __name__ == "__main__":
    main()
