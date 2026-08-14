"""Aggregate all frozen spectral-CNN summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "aggregate"


def main() -> None:
    files = sorted(RESULTS.glob("seed*/**/fit_summary.json"))
    if len(files) != 90:
        raise SystemExit(f"expected 90 summaries, found {len(files)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "all_fit_summaries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for split in ("repetition_fold", "specimen_fold"):
        for modality in ("vibration", "current", "fusion"):
            selected = [row for row in rows if row["split"] == split and row["modality"] == modality]
            item = {"split": split, "modality": modality, "fits": len(selected)}
            for metric in ("balanced_accuracy", "macro_f1", "accuracy", "elapsed_s"):
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"mean_{metric}"] = float(values.mean())
                item[f"std_{metric}"] = float(values.std(ddof=1))
                item[f"min_{metric}"] = float(values.min())
                item[f"median_{metric}"] = float(np.median(values))
                item[f"max_{metric}"] = float(values.max())
            summaries.append(item)

    gaps = []
    for modality in ("vibration", "current", "fusion"):
        paired = []
        for seed in (42, 1001, 1002):
            for fold in range(1, 6):
                repetition = next(
                    row for row in rows
                    if row["seed"] == seed and row["fold" if "fold" in row else "test_fold"] == fold
                    and row["modality"] == modality and row["split"] == "repetition_fold"
                )
                specimen = next(
                    row for row in rows
                    if row["seed"] == seed and row["fold" if "fold" in row else "test_fold"] == fold
                    and row["modality"] == modality and row["split"] == "specimen_fold"
                )
                paired.append(float(repetition["balanced_accuracy"]) - float(specimen["balanced_accuracy"]))
        values = np.asarray(paired)
        gaps.append(
            {
                "modality": modality,
                "paired_seed_fold_cells": int(values.size),
                "mean_repetition_minus_specimen": float(values.mean()),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "fraction_positive": float(np.mean(values > 0)),
            }
        )
    evidence = {
        "protocol": "paderborn-spectral-cnn-confirmatory-protocol-20260811",
        "summaries": summaries,
        "paired_gaps": gaps,
    }
    (OUTPUT / "SPECTRAL_CNN_CONFIRMATORY_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    (OUTPUT / "SPECTRAL_CNN_AGGREGATION_COMPLETE.ok").write_text(
        "aggregation complete\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
