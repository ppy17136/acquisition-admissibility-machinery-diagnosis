"""Run frozen diagnosis, metadata-only, and bearing-identity baselines."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURE_TABLE = ROOT / "derived" / "paderborn_record_features.csv"
OUTPUT = ROOT / "results" / "frozen_logistic_baselines"
META_COLUMNS = {
    "record_uid",
    "bearing_id",
    "class_label",
    "specimen_fold",
    "repetition_fold",
    "setting",
    "repeat",
}


def load_table() -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))
    feature_names = [name for name in rows[0] if name not in META_COLUMNS]
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    return rows, feature_names, matrix


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=20260811,
        ),
    )


def multiclass_brier(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> float:
    target = np.zeros_like(probability)
    index = {label: position for position, label in enumerate(classes)}
    for row_index, label in enumerate(y):
        target[row_index, index[label]] = 1.0
    return float(np.mean(np.sum(np.square(probability - target), axis=1)))


def evaluate(
    rows: list[dict[str, str]],
    matrix: np.ndarray,
    target: np.ndarray,
    split_column: str,
    task: str,
    modality: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fold_metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for fold in range(1, 6):
        test = np.asarray([int(row[split_column]) == fold for row in rows])
        train = ~test
        estimator = model()
        estimator.fit(matrix[train], target[train])
        predicted = estimator.predict(matrix[test])
        probability = estimator.predict_proba(matrix[test])
        classes = estimator.classes_
        true = target[test]

        fold_metrics.append(
            {
                "task": task,
                "modality": modality,
                "split": split_column,
                "fold": fold,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "accuracy": float(accuracy_score(true, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
                "macro_f1": float(f1_score(true, predicted, average="macro")),
                "log_loss": float(log_loss(true, probability, labels=classes)),
                "multiclass_brier": multiclass_brier(true, probability, classes),
            }
        )

        indices = np.flatnonzero(test)
        for local_index, global_index in enumerate(indices):
            item: dict[str, object] = {
                "task": task,
                "modality": modality,
                "split": split_column,
                "fold": fold,
                "record_uid": rows[global_index]["record_uid"],
                "bearing_id": rows[global_index]["bearing_id"],
                "setting": rows[global_index]["setting"],
                "true_label": str(true[local_index]),
                "predicted_label": str(predicted[local_index]),
            }
            for class_index, label in enumerate(classes):
                item[f"probability__{label}"] = float(probability[local_index, class_index])
            predictions.append(item)
    return fold_metrics, predictions


def metadata_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    settings = sorted({row["setting"] for row in rows})
    values: list[list[float]] = []
    for row in rows:
        vector = [float(row["setting"] == setting) for setting in settings]
        vector.extend(float(int(row["repeat"]) == repeat) for repeat in range(1, 21))
        values.append(vector)
    return np.asarray(values, dtype=float)


def aggregate(
    metrics: list[dict[str, object]], predictions: list[dict[str, object]]
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in metrics:
        key = (str(row["task"]), str(row["modality"]), str(row["split"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        summary: dict[str, object] = {
            "task": key[0],
            "modality": key[1],
            "split": key[2],
            "folds": len(values),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "log_loss",
            "multiclass_brier",
        ):
            numbers = np.asarray([float(row[metric]) for row in values])
            summary[f"mean_{metric}"] = float(numbers.mean())
            summary[f"std_{metric}"] = float(numbers.std(ddof=1))
            summary[f"min_{metric}"] = float(numbers.min())
            summary[f"max_{metric}"] = float(numbers.max())
        summaries.append(summary)

    diagnosis_predictions = [
        row for row in predictions if row["task"] == "three_class_diagnosis"
    ]
    confusion: dict[str, object] = {}
    for modality in ("vibration", "current", "fusion"):
        for split in ("repetition_fold", "specimen_fold"):
            selected = [
                row
                for row in diagnosis_predictions
                if row["modality"] == modality and row["split"] == split
            ]
            labels = ["healthy", "real_inner_race", "real_outer_race"]
            confusion[f"{modality}__{split}"] = confusion_matrix(
                [row["true_label"] for row in selected],
                [row["predicted_label"] for row in selected],
                labels=labels,
            ).tolist()

    return {"summaries": summaries, "diagnosis_confusion_matrices": confusion}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, feature_names, full_matrix = load_table()
    modalities = {
        "vibration": np.asarray(
            [name.startswith("vibration_1__") for name in feature_names]
        ),
        "current": np.asarray(
            [name.startswith("phase_current_") for name in feature_names]
        ),
        "fusion": np.ones(len(feature_names), dtype=bool),
    }
    diagnosis_target = np.asarray([row["class_label"] for row in rows])
    all_metrics: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []

    for modality, mask in modalities.items():
        for split in ("repetition_fold", "specimen_fold"):
            metrics, predictions = evaluate(
                rows,
                full_matrix[:, mask],
                diagnosis_target,
                split,
                "three_class_diagnosis",
                modality,
            )
            all_metrics.extend(metrics)
            all_predictions.extend(predictions)

    metadata = metadata_matrix(rows)
    for split in ("repetition_fold", "specimen_fold"):
        metrics, predictions = evaluate(
            rows,
            metadata,
            diagnosis_target,
            split,
            "metadata_only_diagnosis",
            "operating_condition_and_repeat",
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    identity_target = np.asarray([row["bearing_id"] for row in rows])
    for modality, mask in modalities.items():
        metrics, predictions = evaluate(
            rows,
            full_matrix[:, mask],
            identity_target,
            "repetition_fold",
            "bearing_identity",
            modality,
        )
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "fold_metrics.csv", all_metrics)
    write_csv(OUTPUT / "record_predictions.csv", all_predictions)
    evidence = aggregate(all_metrics, all_predictions)
    evidence.update(
        {
            "protocol": "paderborn-feature-baseline-spec-20260811",
            "records": len(rows),
            "class_counts": dict(Counter(diagnosis_target)),
            "fixed_C": 1.0,
            "features_by_modality": {
                name: int(mask.sum()) for name, mask in modalities.items()
            },
        }
    )
    (OUTPUT / "FROZEN_LOGISTIC_BASELINE_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
