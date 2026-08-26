"""Compile the gated public-benchmark evidence into one JSON and Markdown record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis" / "compiled_evidence"


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def select(rows: list[dict], **criteria: object) -> dict:
    matched = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matched) != 1:
        raise ValueError(f"expected one row for {criteria}, found {len(matched)}")
    return matched[0]


def main() -> None:
    inventory = read_json("analysis/signal_inventory/PADERBORN_SIGNAL_INVENTORY_AUDIT.json")
    grid = read_json("analysis/common_time_grid/COMMON_TIME_GRID_AUDIT.json")
    features = read_json("derived/PADERBORN_FEATURE_BUILD.json")
    baseline = read_json(
        "results/frozen_logistic_baselines/FROZEN_LOGISTIC_BASELINE_EVIDENCE.json"
    )
    pseudolabel = read_json(
        "results/pseudolabel_accuracy_v2/PSEUDOLABEL_ACCURACY_V2_EVIDENCE.json"
    )
    transfer = read_json("results/operating_transfer/OPERATING_TRANSFER_EVIDENCE.json")
    bootstrap = read_json(
        "results/bearing_level_bootstrap/BEARING_LEVEL_BOOTSTRAP_EVIDENCE.json"
    )

    actual: dict[str, dict[str, float]] = {}
    identity: dict[str, float] = {}
    for modality in ("vibration", "current", "fusion"):
        repetition = select(
            baseline["summaries"], task="three_class_diagnosis",
            modality=modality, split="repetition_fold"
        )
        specimen = select(
            baseline["summaries"], task="three_class_diagnosis",
            modality=modality, split="specimen_fold"
        )
        actual[modality] = {
            "repetition_balanced_accuracy": repetition["mean_balanced_accuracy"],
            "specimen_balanced_accuracy": specimen["mean_balanced_accuracy"],
            "absolute_gap": (
                repetition["mean_balanced_accuracy"] - specimen["mean_balanced_accuracy"]
            ),
            "repetition_macro_f1": repetition["mean_macro_f1"],
            "specimen_macro_f1": specimen["mean_macro_f1"],
        }
        identity_row = select(
            baseline["summaries"], task="bearing_identity",
            modality=modality, split="repetition_fold"
        )
        identity[modality] = identity_row["mean_accuracy"]

    metadata_row = select(
        baseline["summaries"], task="metadata_only_diagnosis",
        modality="operating_condition_and_repeat", split="specimen_fold"
    )
    pseudo: dict[str, dict[str, float]] = {}
    for modality in ("vibration", "current", "fusion"):
        repetition = select(
            pseudolabel["summaries"], modality=modality, split="repetition_fold"
        )
        specimen = select(
            pseudolabel["summaries"], modality=modality, split="specimen_fold"
        )
        paired = select(
            pseudolabel["paired_repetition_minus_specimen"], modality=modality
        )
        pseudo[modality] = {
            "repetition_mean_accuracy": repetition["mean_accuracy"],
            "specimen_mean_accuracy": specimen["mean_accuracy"],
            "mean_paired_gap": paired["mean_repetition_minus_specimen"],
            "paired_gap_q025": paired["q025"],
            "paired_gap_q975": paired["q975"],
            "fraction_positive": paired["fraction_positive"],
        }

    joint_transfer: dict[str, dict[str, float]] = {}
    for modality in ("vibration", "current", "fusion"):
        row = select(
            transfer["summaries"], design="joint_unseen_specimen_and_setting",
            modality=modality, metric="balanced_accuracy"
        )
        joint_transfer[modality] = {
            "cells": row["cells"],
            "mean": row["mean"],
            "min": row["min"],
            "median": row["median"],
            "max": row["max"],
        }

    evidence = {
        "protocol": "provenance-gated-public-positive-benchmark-20260811",
        "inventory": inventory,
        "common_grid": grid,
        "feature_build": features,
        "actual_three_class_diagnosis": actual,
        "repetition_split_bearing_identity_accuracy": identity,
        "metadata_only_specimen_split_balanced_accuracy": metadata_row[
            "mean_balanced_accuracy"
        ],
        "balanced_specimen_pseudolabel_control_200_permutations": pseudo,
        "joint_unseen_specimen_and_setting": joint_transfer,
        "bearing_level_bootstrap": bootstrap["summaries"],
        "claim_boundary": {
            "supported": [
                "Record-disjoint validation can remain specimen-leaky when repeated records from the same physical bearing occur in training and test sets.",
                "The frozen repetition split substantially overestimated unseen-bearing performance for all three modalities in this public benchmark.",
                "Near-perfect prediction of arbitrary specimen-level pseudolabels demonstrates a specimen-fingerprint shortcut rather than fault information.",
                "Acquisition provenance and deployment-unit-aligned splitting are prerequisites for interpreting diagnostic accuracy.",
            ],
            "not_supported": [
                "The exact observed accuracy gap applies to every machinery dataset.",
                "Deep learning or sensor fusion is intrinsically invalid.",
                "Fifteen bearings establish unrestricted field-deployment performance.",
            ],
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "PUBLIC_BENCHMARK_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )

    lines = [
        "# Public benchmark evidence summary",
        "",
        "## Audited data",
        "",
        f"- {inventory['bearing_count']} independent bearings and {inventory['record_count']} four-second records.",
        "- Three balanced classes (400 records each) and four settings (20 repeats per bearing-setting cell).",
        "- Vibration and two phase-current channels are synchronous; the common 64 kHz grid is covered by every record.",
        "",
        "## Actual three-class diagnosis",
        "",
        "| Modality | Repetition split | Bearing-held-out | Absolute gap |",
        "|---|---:|---:|---:|",
    ]
    for modality in ("vibration", "current", "fusion"):
        row = actual[modality]
        lines.append(
            f"| {modality} | {row['repetition_balanced_accuracy']:.4f} | "
            f"{row['specimen_balanced_accuracy']:.4f} | {row['absolute_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shortcut controls",
            "",
            f"- Repetition-split bearing identity accuracy: vibration {identity['vibration']:.4f}, current {identity['current']:.4f}, fusion {identity['fusion']:.4f}.",
            f"- Metadata-only diagnosis remained at {metadata_row['mean_balanced_accuracy']:.4f} balanced accuracy.",
            f"- Across 200 balanced random specimen-pseudolabel assignments, fusion accuracy was {pseudo['fusion']['repetition_mean_accuracy']:.4f} under repetition splitting and {pseudo['fusion']['specimen_mean_accuracy']:.4f} under bearing-held-out splitting.",
            f"- The fusion pseudolabel gap was positive in {pseudo['fusion']['fraction_positive']:.0%} of permutations.",
            "",
            "## Joint transfer",
            "",
            f"- Joint unseen-bearing/unseen-setting mean balanced accuracy: vibration {joint_transfer['vibration']['mean']:.4f}, current {joint_transfer['current']['mean']:.4f}, fusion {joint_transfer['fusion']['mean']:.4f}.",
            "",
            "## Interpretation",
            "",
            "The public result is not a claim that high diagnostic accuracy is generally false. It demonstrates that record-disjoint validation can answer the wrong deployment question when the experimental unit is a physical specimen. Acquisition provenance, specimen identity, nuisance-label controls, and deployment-unit-aligned splitting must be audited before accuracy is interpreted.",
        ]
    )
    (OUTPUT / "PUBLIC_BENCHMARK_EVIDENCE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
