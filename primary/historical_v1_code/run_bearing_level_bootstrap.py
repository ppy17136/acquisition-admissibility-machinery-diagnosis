"""Compute paired bearing-level bootstrap intervals for the split-policy gap."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results" / "frozen_logistic_baselines" / "record_predictions.csv"
OUTPUT = ROOT / "results" / "bearing_level_bootstrap"
CLASSES = ("healthy", "real_inner_race", "real_outer_race")
MODALITIES = ("vibration", "current", "fusion")
REPLICATES = 10_000
RNG_SEED = 20260811


def balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float(np.mean(predicted[true == label] == label)) for label in CLASSES]
    return float(np.mean(recalls))


def main() -> None:
    raw = list(csv.DictReader(PREDICTIONS.open(encoding="utf-8")))
    selected = [row for row in raw if row["task"] == "three_class_diagnosis"]
    classes_by_bearing: dict[str, str] = {}
    for row in selected:
        classes_by_bearing[row["bearing_id"]] = row["true_label"]
    bearings_by_class: dict[str, list[str]] = defaultdict(list)
    for bearing, label in sorted(classes_by_bearing.items()):
        bearings_by_class[label].append(bearing)
    if any(len(bearings_by_class[label]) != 5 for label in CLASSES):
        raise SystemExit(f"unexpected bearings by class: {dict(bearings_by_class)}")

    summaries: list[dict[str, object]] = []
    per_bearing: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED)

    for modality in MODALITIES:
        by_split: dict[str, dict[str, list[dict[str, str]]]] = {}
        for split in ("repetition_fold", "specimen_fold"):
            mapping: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in selected:
                if row["modality"] == modality and row["split"] == split:
                    mapping[row["bearing_id"]].append(row)
            if any(len(mapping[bearing]) != 80 for bearing in classes_by_bearing):
                raise SystemExit(f"missing predictions for {modality} {split}")
            by_split[split] = mapping

        for bearing in sorted(classes_by_bearing):
            item: dict[str, object] = {
                "modality": modality,
                "bearing_id": bearing,
                "class_label": classes_by_bearing[bearing],
            }
            for split in by_split:
                records = by_split[split][bearing]
                item[f"accuracy__{split}"] = float(
                    np.mean(
                        [row["true_label"] == row["predicted_label"] for row in records]
                    )
                )
            item["repetition_minus_specimen"] = (
                float(item["accuracy__repetition_fold"])
                - float(item["accuracy__specimen_fold"])
            )
            per_bearing.append(item)

        observed: dict[str, float] = {}
        for split in by_split:
            records = [row for group in by_split[split].values() for row in group]
            observed[split] = balanced_accuracy(
                np.asarray([row["true_label"] for row in records]),
                np.asarray([row["predicted_label"] for row in records]),
            )

        bootstrap = np.empty((REPLICATES, 3), dtype=float)
        for replicate in range(REPLICATES):
            sampled_bearings: list[str] = []
            for label in CLASSES:
                sampled_bearings.extend(
                    rng.choice(bearings_by_class[label], size=5, replace=True).tolist()
                )
            values: list[float] = []
            for split in ("repetition_fold", "specimen_fold"):
                records = [
                    row
                    for bearing in sampled_bearings
                    for row in by_split[split][bearing]
                ]
                values.append(
                    balanced_accuracy(
                        np.asarray([row["true_label"] for row in records]),
                        np.asarray([row["predicted_label"] for row in records]),
                    )
                )
            bootstrap[replicate] = (values[0], values[1], values[0] - values[1])

        names = ("repetition_balanced_accuracy", "specimen_balanced_accuracy", "paired_gap")
        summary: dict[str, object] = {
            "modality": modality,
            "observed_repetition_balanced_accuracy": observed["repetition_fold"],
            "observed_specimen_balanced_accuracy": observed["specimen_fold"],
            "observed_paired_gap": observed["repetition_fold"] - observed["specimen_fold"],
        }
        for column, name in enumerate(names):
            summary[f"{name}_q025"] = float(np.quantile(bootstrap[:, column], 0.025))
            summary[f"{name}_median"] = float(np.quantile(bootstrap[:, column], 0.5))
            summary[f"{name}_q975"] = float(np.quantile(bootstrap[:, column], 0.975))
        summaries.append(summary)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_bearing_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_bearing[0]))
        writer.writeheader()
        writer.writerows(per_bearing)
    evidence = {
        "protocol": "paderborn-bearing-bootstrap-spec-20260811",
        "rng_seed": RNG_SEED,
        "replicates": REPLICATES,
        "resampling_unit": "physical bearing stratified by true class",
        "summaries": summaries,
    }
    (OUTPUT / "BEARING_LEVEL_BOOTSTRAP_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
