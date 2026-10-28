#!/usr/bin/env python3
"""Compare newly generated GPU outputs with published 7B AUC targets.

This script intentionally refuses to report a reproduction if required new files
are absent. Published numbers are references, never substituted as results.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

TARGETS = {
    ("S2_1P", "final_token", 24): 0.9565,
    ("S2_3P", "final_token", 24): 0.9650,
    ("S2_1P", "mean", 26): 0.9570,
    ("S2_3P", "mean", 26): 0.9400,
}


def report(output_dir: Path) -> None:
    required = [
        output_dir / "summary.json", output_dir / "layer_curves.csv",
        output_dir / "final_token" / "pain_vectors.pt", output_dir / "activations.pt",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Independent extraction not complete. Missing:\n" + "\n".join(missing))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "layer_curves.csv").open(newline="", encoding="utf-8") as fh:
        rows = {
            (r["dataset"], r["extraction"], int(r["layer"])): float(r["auc_vs_all_controls"])
            for r in csv.DictReader(fh)
        }
    print("INDEPENDENT GPU EXTRACTION compared with published author references")
    print("Model:", summary.get("model"))
    print("Best final-token layer:", summary.get("best_layer_final_token"), "| published: 24")
    print("Best mean-token layer:", summary.get("best_layer_mean"), "| published: 26")
    for key, expected in TARGETS.items():
        observed = rows.get(key)
        if observed is None:
            print("MISSING", key)
        else:
            print("{}: observed={:.4f} published={:.4f} delta={:+.4f}".format(
                key, observed, expected, observed - expected
            ))
    print("These differences must be reported, not tuned away.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report(args.output_dir)


if __name__ == "__main__":
    main()
