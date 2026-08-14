"""Run frozen condition-only and joint specimen-condition transfer tests."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURE_TABLE = ROOT / "derived" / "paderborn_record_features.csv"
OUTPUT = ROOT / "results" / "operating_transfer"
META_COLUMNS = {
    "record_uid", "bearing_id", "class_label", "specimen_fold",
    "repetition_fold", "setting", "repeat",
}


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000, random_state=20260811),
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))
    feature_names = [name for name in rows[0] if name not in META_COLUMNS]
    full_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    target = np.asarray([row["class_label"] for row in rows])
    settings = np.asarray([row["setting"] for row in rows])
    specimen_folds = np.asarray([int(row["specimen_fold"]) for row in rows])
    modalities = {
        "vibration": np.asarray([name.startswith("vibration_1__") for name in feature_names]),
        "current": np.asarray([name.startswith("phase_current_") for name in feature_names]),
        "fusion": np.ones(len(feature_names), dtype=bool),
    }

    results: list[dict[str, object]] = []
    for modality, feature_mask in modalities.items():
        matrix = full_matrix[:, feature_mask]
        for setting in sorted(set(settings)):
            test = settings == setting
            estimator = model()
            estimator.fit(matrix[~test], target[~test])
            predicted = estimator.predict(matrix[test])
            results.append(
                {
                    "design": "condition_only_shared_bearings",
                    "modality": modality,
                    "held_out_setting": setting,
                    "specimen_fold": "shared",
                    "train_rows": int((~test).sum()),
                    "test_rows": int(test.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(target[test], predicted)),
                    "macro_f1": float(f1_score(target[test], predicted, average="macro")),
                }
            )

            for fold in range(1, 6):
                test = (settings == setting) & (specimen_folds == fold)
                train = (settings != setting) & (specimen_folds != fold)
                estimator = model()
                estimator.fit(matrix[train], target[train])
                predicted = estimator.predict(matrix[test])
                results.append(
                    {
                        "design": "joint_unseen_specimen_and_setting",
                        "modality": modality,
                        "held_out_setting": setting,
                        "specimen_fold": fold,
                        "train_rows": int(train.sum()),
                        "test_rows": int(test.sum()),
                        "balanced_accuracy": float(balanced_accuracy_score(target[test], predicted)),
                        "macro_f1": float(f1_score(target[test], predicted, average="macro")),
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "operating_transfer_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summaries: list[dict[str, object]] = []
    for design in sorted({str(row["design"]) for row in results}):
        for modality in modalities:
            selected = [
                row for row in results
                if row["design"] == design and row["modality"] == modality
            ]
            for metric in ("balanced_accuracy", "macro_f1"):
                values = np.asarray([float(row[metric]) for row in selected])
                summaries.append(
                    {
                        "design": design,
                        "modality": modality,
                        "metric": metric,
                        "cells": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "min": float(values.min()),
                        "median": float(np.median(values)),
                        "max": float(values.max()),
                    }
                )

    evidence = {
        "protocol": "paderborn-operating-transfer-spec-20260811",
        "settings": sorted(set(settings)),
        "summaries": summaries,
    }
    (OUTPUT / "OPERATING_TRANSFER_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
