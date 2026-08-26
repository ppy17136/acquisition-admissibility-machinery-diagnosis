from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

from v2_common import (
    CLASSES,
    FEATURE_TABLE,
    MODALITIES,
    RNG_SEED,
    SPLITS,
    DataBundle,
    environment_record,
    load_data,
    mean_summary,
    model,
    sha256,
    write_csv,
    write_json,
)


HERE = Path(__file__).resolve().parent
FULL_OUTPUT = HERE.parent / "rerun_evidence"
SMOKE_OUTPUT = HERE.parent / "rerun_smoke"
BOOTSTRAP_REPLICATES = 10_000
PSEUDOLABEL_PERMUTATIONS = 200


def multiclass_brier(y: np.ndarray, probability: np.ndarray, classes: np.ndarray) -> float:
    target = np.zeros_like(probability)
    index = {label: position for position, label in enumerate(classes)}
    for row_index, label in enumerate(y):
        target[row_index, index[label]] = 1.0
    return float(np.mean(np.sum(np.square(probability - target), axis=1)))


def fit_fold(
    matrix: np.ndarray,
    target: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimator = model()
    estimator.fit(matrix[~test], target[~test])
    return estimator.predict(matrix[test]), estimator.predict_proba(matrix[test]), estimator.classes_


def metadata_matrix(data: DataBundle) -> np.ndarray:
    settings = sorted(set(data.settings))
    values: list[list[float]] = []
    for row in data.rows:
        vector = [float(row["setting"] == setting) for setting in settings]
        vector.extend(float(int(row["repeat"]) == repeat) for repeat in range(1, 21))
        values.append(vector)
    return np.asarray(values, dtype=float)


def evaluate_baseline_task(
    data: DataBundle,
    matrix: np.ndarray,
    target: np.ndarray,
    split: str,
    task: str,
    modality: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for fold in range(1, 6):
        test = data.split_values[split] == fold
        predicted, probability, classes = fit_fold(matrix, target, test)
        true = target[test]
        metrics.append(
            {
                "task": task,
                "modality": modality,
                "split": split,
                "fold": fold,
                "train_rows": int((~test).sum()),
                "test_rows": int(test.sum()),
                "accuracy": float(accuracy_score(true, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
                "macro_f1": float(f1_score(true, predicted, average="macro")),
                "log_loss": float(log_loss(true, probability, labels=classes)),
                "multiclass_brier": multiclass_brier(true, probability, classes),
            }
        )
        for local_index, global_index in enumerate(np.flatnonzero(test)):
            item: dict[str, object] = {
                "task": task,
                "modality": modality,
                "split": split,
                "fold": fold,
                "record_uid": data.rows[global_index]["record_uid"],
                "bearing_id": data.rows[global_index]["bearing_id"],
                "setting": data.rows[global_index]["setting"],
                "true_label": str(true[local_index]),
                "predicted_label": str(predicted[local_index]),
            }
            for class_index, label in enumerate(classes):
                item[f"probability__{label}"] = float(probability[local_index, class_index])
            predictions.append(item)
    return metrics, predictions


def aggregate_baselines(
    metrics: list[dict[str, object]], predictions: list[dict[str, object]]
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metrics:
        grouped[(str(row["task"]), str(row["modality"]), str(row["split"]))].append(row)
    summaries: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        summary: dict[str, object] = {
            "task": key[0], "modality": key[1], "split": key[2], "folds": len(values)
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss", "multiclass_brier"):
            array = np.asarray([float(row[metric]) for row in values])
            summary.update(
                {
                    f"mean_{metric}": float(array.mean()),
                    f"std_{metric}": float(array.std(ddof=1)),
                    f"min_{metric}": float(array.min()),
                    f"max_{metric}": float(array.max()),
                }
            )
        summaries.append(summary)
    confusion: dict[str, object] = {}
    labels = list(CLASSES)
    for modality in MODALITIES:
        for split in SPLITS:
            selected = [
                row for row in predictions
                if row["task"] == "three_class_diagnosis"
                and row["modality"] == modality and row["split"] == split
            ]
            confusion[f"{modality}__{split}"] = confusion_matrix(
                [row["true_label"] for row in selected],
                [row["predicted_label"] for row in selected],
                labels=labels,
            ).tolist()
    return {"summaries": summaries, "diagnosis_confusion_matrices": confusion}


def run_baselines(data: DataBundle, output: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for modality in MODALITIES:
        matrix = data.matrix[:, data.modality_masks[modality]]
        for split in SPLITS:
            current_metrics, current_predictions = evaluate_baseline_task(
                data, matrix, data.diagnosis_target, split,
                "three_class_diagnosis", modality,
            )
            metrics.extend(current_metrics)
            predictions.extend(current_predictions)
    metadata = metadata_matrix(data)
    for split in SPLITS:
        current_metrics, current_predictions = evaluate_baseline_task(
            data, metadata, data.diagnosis_target, split,
            "metadata_only_diagnosis", "operating_condition_and_repeat",
        )
        metrics.extend(current_metrics)
        predictions.extend(current_predictions)
    for modality in MODALITIES:
        current_metrics, current_predictions = evaluate_baseline_task(
            data, data.matrix[:, data.modality_masks[modality]], data.bearing_target,
            "repetition_fold", "bearing_identity", modality,
        )
        metrics.extend(current_metrics)
        predictions.extend(current_predictions)
    evidence = aggregate_baselines(metrics, predictions)
    evidence.update(
        {
            "protocol": "unified-v2-logistic-baselines",
            "records": len(data.rows),
            "class_counts": dict(Counter(data.diagnosis_target)),
            "fixed_C": 1.0,
            "features_by_modality": {
                name: int(mask.sum()) for name, mask in data.modality_masks.items()
            },
        }
    )
    directory = output / "baselines"
    write_csv(directory / "fold_metrics.csv", metrics)
    write_csv(directory / "record_predictions.csv", predictions)
    write_json(directory / "BASELINE_EVIDENCE.json", evidence)
    return evidence, predictions


def evaluate_pseudolabel(
    permutation_id: int,
    modality: str,
    split: str,
    matrix: np.ndarray,
    target: np.ndarray,
    split_values: np.ndarray,
) -> dict[str, object]:
    scores: list[float] = []
    for fold in range(1, 6):
        test = split_values == fold
        predicted, _, _ = fit_fold(matrix, target, test)
        scores.append(float(accuracy_score(target[test], predicted)))
    return {
        "permutation_id": permutation_id,
        "modality": modality,
        "split": split,
        **{f"fold{fold}_accuracy": scores[fold - 1] for fold in range(1, 6)},
        "mean_accuracy": float(np.mean(scores)),
    }


def run_pseudolabels(
    data: DataBundle, output: Path, permutations: int, jobs: int
) -> dict[str, object]:
    bearings = np.asarray(sorted(set(data.bearing_target)))
    rng = np.random.default_rng(RNG_SEED)
    assignments: list[dict[str, object]] = []
    targets: dict[int, np.ndarray] = {}
    for permutation_id in range(1, permutations + 1):
        ordered = rng.permutation(bearings)
        mapping = {bearing: f"pseudo_{index // 5 + 1}" for index, bearing in enumerate(ordered)}
        targets[permutation_id] = np.asarray([mapping[bearing] for bearing in data.bearing_target])
        assignments.extend(
            {
                "permutation_id": permutation_id,
                "bearing_id": bearing,
                "pseudolabel": mapping[bearing],
            }
            for bearing in bearings
        )
    tasks = [
        (permutation_id, modality, split)
        for permutation_id in range(1, permutations + 1)
        for modality in MODALITIES for split in SPLITS
    ]
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=jobs, verbose=5)(
            delayed(evaluate_pseudolabel)(
                permutation_id, modality, split,
                data.matrix[:, data.modality_masks[modality]],
                targets[permutation_id], data.split_values[split],
            )
            for permutation_id, modality, split in tasks
        )
    summaries: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for modality in MODALITIES:
        by_split: dict[str, np.ndarray] = {}
        for split in SPLITS:
            values = np.asarray([
                float(row["mean_accuracy"]) for row in results
                if row["modality"] == modality and row["split"] == split
            ])
            by_split[split] = values
            summaries.append(
                {
                    "modality": modality, "split": split,
                    "permutations": int(values.size),
                    **mean_summary(values, "accuracy"),
                }
            )
        difference = by_split["repetition_fold"] - by_split["specimen_fold"]
        paired.append(
            {
                "modality": modality,
                "mean_repetition_minus_specimen": float(difference.mean()),
                "q025": float(np.quantile(difference, 0.025)),
                "median": float(np.median(difference)),
                "q975": float(np.quantile(difference, 0.975)),
                "fraction_positive": float(np.mean(difference > 0)),
            }
        )
    evidence = {
        "protocol": "unified-v2-balanced-specimen-pseudolabels",
        "rng_seed": RNG_SEED,
        "balanced_permutations": permutations,
        "uniform_guessing_expected_accuracy": 1 / 3,
        "summaries": summaries,
        "paired_repetition_minus_specimen": paired,
    }
    directory = output / "pseudolabels"
    write_csv(directory / "balanced_pseudolabel_assignments.csv", assignments)
    write_csv(directory / "pseudolabel_permutation_accuracy.csv", results)
    write_json(directory / "PSEUDOLABEL_EVIDENCE.json", evidence)
    return evidence


def run_class_conditional_identity(data: DataBundle, output: Path) -> dict[str, object]:
    fold_rows: list[dict[str, object]] = []
    for class_label in CLASSES:
        class_mask = data.diagnosis_target == class_label
        target = data.bearing_target[class_mask]
        split = data.split_values["repetition_fold"][class_mask]
        for modality in MODALITIES:
            matrix = data.matrix[class_mask][:, data.modality_masks[modality]]
            for fold in range(1, 6):
                test = split == fold
                predicted, _, _ = fit_fold(matrix, target, test)
                fold_rows.append(
                    {
                        "class_label": class_label,
                        "modality": modality,
                        "fold": fold,
                        "train_rows": int((~test).sum()),
                        "test_rows": int(test.sum()),
                        "five_way_identity_accuracy": float(accuracy_score(target[test], predicted)),
                    }
                )
    summaries: list[dict[str, object]] = []
    for class_label in CLASSES:
        for modality in MODALITIES:
            values = [
                float(row["five_way_identity_accuracy"]) for row in fold_rows
                if row["class_label"] == class_label and row["modality"] == modality
            ]
            summaries.append(
                {
                    "class_label": class_label, "modality": modality, "folds": len(values),
                    **mean_summary(values, "accuracy"),
                }
            )
    evidence = {
        "protocol": "unified-v2-class-conditional-bearing-identity",
        "classes": list(CLASSES),
        "chance_accuracy": 0.2,
        "summaries": summaries,
    }
    directory = output / "control_a_class_conditional_identity"
    write_csv(directory / "fold_metrics.csv", fold_rows)
    write_json(directory / "CONTROL_A_EVIDENCE.json", evidence)
    return evidence


def canonical(mapping: dict[str, int], order: list[str]) -> tuple[int, ...]:
    relabel: dict[int, int] = {}
    next_label = 0
    result: list[int] = []
    for bearing in order:
        original = mapping[bearing]
        if original not in relabel:
            relabel[original] = next_label
            next_label += 1
        result.append(relabel[original])
    return tuple(result)


def bearing_metadata(data: DataBundle) -> tuple[list[str], dict[str, dict[str, object]], dict[str, dict[int, str]]]:
    metadata: dict[str, dict[str, object]] = {}
    for row in data.rows:
        metadata.setdefault(
            row["bearing_id"],
            {
                "bearing_id": row["bearing_id"],
                "class_label": row["class_label"],
                "specimen_fold": int(row["specimen_fold"]),
            },
        )
    order = sorted(metadata)
    by_class_fold: dict[str, dict[int, str]] = defaultdict(dict)
    for row in metadata.values():
        by_class_fold[str(row["class_label"])][int(row["specimen_fold"])] = str(row["bearing_id"])
    return order, metadata, by_class_fold


def generate_orthogonal_mappings(
    by_class_fold: dict[str, dict[int, str]], order: list[str]
) -> list[dict[str, int]]:
    healthy = by_class_fold["healthy"]
    outer = by_class_fold["real_outer_race"]
    inner = by_class_fold["real_inner_race"]
    unique: dict[tuple[int, ...], dict[str, int]] = {}
    for outer_labels in itertools.permutations(range(5)):
        for inner_labels in itertools.permutations(range(5)):
            if any(len({fold - 1, outer_labels[fold - 1], inner_labels[fold - 1]}) != 3 for fold in range(1, 6)):
                continue
            mapping: dict[str, int] = {}
            for fold in range(1, 6):
                mapping[healthy[fold]] = fold - 1
                mapping[outer[fold]] = outer_labels[fold - 1]
                mapping[inner[fold]] = inner_labels[fold - 1]
            unique[canonical(mapping, order)] = mapping
    mappings = [unique[key] for key in sorted(unique)]
    if len(mappings) != 552:
        raise AssertionError(f"expected 552 mappings, got {len(mappings)}")
    return mappings


def run_orthogonal_decoder(
    data: DataBundle, predictions: list[dict[str, object]], output: Path
) -> dict[str, object]:
    order, metadata, by_class_fold = bearing_metadata(data)
    mappings = generate_orthogonal_mappings(by_class_fold, order)
    mapping_rows: list[dict[str, object]] = []
    for mapping_id, mapping in enumerate(mappings, 1):
        for bearing in order:
            mapping_rows.append(
                {
                    "mapping_id": mapping_id,
                    "bearing_id": bearing,
                    "class_label": metadata[bearing]["class_label"],
                    "specimen_fold": metadata[bearing]["specimen_fold"],
                    "pseudoclass": f"P{mapping[bearing] + 1}",
                }
            )
    identity = [row for row in predictions if row["task"] == "bearing_identity"]
    if len(identity) != 3600:
        raise AssertionError(f"expected 3600 identity predictions, got {len(identity)}")
    decoder_rows: list[dict[str, object]] = []
    for modality in MODALITIES:
        selected = [row for row in identity if row["modality"] == modality]
        for mapping_id, mapping in enumerate(mappings, 1):
            correct = sum(
                mapping[str(row["true_label"])] == mapping[str(row["predicted_label"])]
                for row in selected
            )
            decoder_rows.append(
                {
                    "mapping_id": mapping_id, "modality": modality,
                    "records": len(selected), "decoded_accuracy": correct / len(selected),
                }
            )
    summaries = []
    for modality in MODALITIES:
        values = [
            float(row["decoded_accuracy"]) for row in decoder_rows
            if row["modality"] == modality
        ]
        summaries.append(
            {"modality": modality, "mappings": len(values), **mean_summary(values, "accuracy")}
        )
    evidence = {
        "protocol": "unified-v2-fault-orthogonal-identity-decoder",
        "construction": "deterministic mapping of V2 out-of-fold bearing-identity predictions; no pseudolabel model refitting",
        "orthogonal_mapping_count": len(mappings),
        "identity_prediction_rows": len(identity),
        "summaries": summaries,
    }
    directory = output / "control_b2_orthogonal_decoder"
    write_csv(directory / "orthogonal_mapping_inventory.csv", mapping_rows)
    write_csv(directory / "orthogonal_identity_decoder.csv", decoder_rows)
    write_json(directory / "CONTROL_B2_EVIDENCE.json", evidence)
    return evidence


def evaluate_triplet(
    triplet_id: int, bearings: tuple[str, str, str], modality: str,
    matrix: np.ndarray, target: np.ndarray, bearing_target: np.ndarray,
) -> dict[str, object]:
    test = np.isin(bearing_target, np.asarray(bearings))
    predicted, _, _ = fit_fold(matrix, target, test)
    return {
        "triplet_id": triplet_id,
        "healthy_bearing": bearings[0],
        "outer_bearing": bearings[1],
        "inner_bearing": bearings[2],
        "modality": modality,
        "train_rows": int((~test).sum()),
        "test_rows": int(test.sum()),
        "balanced_accuracy": float(balanced_accuracy_score(target[test], predicted)),
        "macro_f1": float(f1_score(target[test], predicted, average="macro")),
    }


def run_partition_sensitivity(
    data: DataBundle, output: Path, jobs: int, smoke_triplets: int | None = None
) -> dict[str, object]:
    _, metadata, by_class_fold = bearing_metadata(data)
    healthy = [by_class_fold["healthy"][fold] for fold in range(1, 6)]
    outer = [by_class_fold["real_outer_race"][fold] for fold in range(1, 6)]
    inner = [by_class_fold["real_inner_race"][fold] for fold in range(1, 6)]
    all_triplets = list(itertools.product(healthy, outer, inner))
    triplets = all_triplets if smoke_triplets is None else all_triplets[:smoke_triplets]
    tasks = [
        (triplet_id, bearings, modality)
        for triplet_id, bearings in enumerate(triplets, 1) for modality in MODALITIES
    ]
    with parallel_backend("loky", inner_max_num_threads=1):
        triplet_rows = Parallel(n_jobs=jobs, verbose=5)(
            delayed(evaluate_triplet)(
                triplet_id, bearings, modality,
                data.matrix[:, data.modality_masks[modality]],
                data.diagnosis_target, data.bearing_target,
            )
            for triplet_id, bearings, modality in tasks
        )
    directory = output / "control_c_partition_sensitivity"
    write_csv(directory / "unique_triplet_metrics.csv", triplet_rows)
    if smoke_triplets is not None:
        evidence = {
            "protocol": "unified-v2-partition-sensitivity-smoke",
            "computed_unique_triplets": len(triplets),
            "total_unique_triplets": len(all_triplets),
        }
        write_json(directory / "CONTROL_C_SMOKE_EVIDENCE.json", evidence)
        return evidence

    score_lookup = {
        (str(row["healthy_bearing"]), str(row["outer_bearing"]), str(row["inner_bearing"]), str(row["modality"])): float(row["balanced_accuracy"])
        for row in triplet_rows
    }
    partition_rows: list[dict[str, object]] = []
    primary_id: int | None = None
    partition_id = 0
    for outer_order in itertools.permutations(outer):
        for inner_order in itertools.permutations(inner):
            partition_id += 1
            folds = tuple((healthy[index], outer_order[index], inner_order[index]) for index in range(5))
            is_primary = all(
                int(metadata[bearing]["specimen_fold"]) == fold_index
                for fold_index, fold in enumerate(folds, 1) for bearing in fold
            )
            if is_primary:
                if primary_id is not None:
                    raise AssertionError("multiple primary partitions")
                primary_id = partition_id
            for modality in MODALITIES:
                scores = [score_lookup[(fold[0], fold[1], fold[2], modality)] for fold in folds]
                partition_rows.append(
                    {
                        "partition_id": partition_id,
                        "modality": modality,
                        "is_prespecified_primary": is_primary,
                        "mean_balanced_accuracy": float(np.mean(scores)),
                        "minimum_fold_balanced_accuracy": float(np.min(scores)),
                        "maximum_fold_balanced_accuracy": float(np.max(scores)),
                    }
                )
    if partition_id != 14400 or primary_id is None:
        raise AssertionError((partition_id, primary_id))
    summaries: list[dict[str, object]] = []
    for modality in MODALITIES:
        values = np.asarray([
            float(row["mean_balanced_accuracy"]) for row in partition_rows
            if row["modality"] == modality
        ])
        primary = next(
            float(row["mean_balanced_accuracy"]) for row in partition_rows
            if row["modality"] == modality and row["is_prespecified_primary"]
        )
        summaries.append(
            {
                "modality": modality,
                "partitions": int(values.size),
                "prespecified_primary_mean_balanced_accuracy": primary,
                "primary_percentile_rank_weak": float(np.mean(values <= primary)),
                **mean_summary(values, "partition_mean_balanced_accuracy"),
            }
        )
    evidence = {
        "protocol": "unified-v2-exhaustive-class-balanced-specimen-partitions",
        "unique_class_balanced_test_triplets": len(all_triplets),
        "admissible_partition_count": partition_id,
        "prespecified_primary_partition_id": primary_id,
        "interpretation": "finite design sensitivity distribution, not a confidence interval",
        "summaries": summaries,
    }
    write_csv(directory / "partition_metrics.csv", partition_rows)
    write_json(directory / "CONTROL_C_EVIDENCE.json", evidence)
    return evidence


def run_operating_transfer(data: DataBundle, output: Path) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for modality in MODALITIES:
        matrix = data.matrix[:, data.modality_masks[modality]]
        for setting in sorted(set(data.settings)):
            test = data.settings == setting
            predicted, _, _ = fit_fold(matrix, data.diagnosis_target, test)
            cells.append(
                {
                    "design": "condition_only_shared_bearings", "modality": modality,
                    "held_out_setting": setting, "specimen_fold": "shared",
                    "train_rows": int((~test).sum()), "test_rows": int(test.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(data.diagnosis_target[test], predicted)),
                    "macro_f1": float(f1_score(data.diagnosis_target[test], predicted, average="macro")),
                }
            )
            for fold in range(1, 6):
                test = (data.settings == setting) & (data.split_values["specimen_fold"] == fold)
                train = (data.settings != setting) & (data.split_values["specimen_fold"] != fold)
                estimator = model()
                estimator.fit(matrix[train], data.diagnosis_target[train])
                predicted = estimator.predict(matrix[test])
                cells.append(
                    {
                        "design": "joint_unseen_specimen_and_setting", "modality": modality,
                        "held_out_setting": setting, "specimen_fold": fold,
                        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                        "balanced_accuracy": float(balanced_accuracy_score(data.diagnosis_target[test], predicted)),
                        "macro_f1": float(f1_score(data.diagnosis_target[test], predicted, average="macro")),
                    }
                )
    summaries: list[dict[str, object]] = []
    for design in sorted({str(row["design"]) for row in cells}):
        for modality in MODALITIES:
            selected = [row for row in cells if row["design"] == design and row["modality"] == modality]
            for metric in ("balanced_accuracy", "macro_f1"):
                summaries.append(
                    {
                        "design": design, "modality": modality, "metric": metric,
                        "cells": len(selected),
                        **mean_summary([float(row[metric]) for row in selected]),
                    }
                )
    evidence = {
        "protocol": "unified-v2-operating-transfer",
        "settings": sorted(set(data.settings)),
        "summaries": summaries,
    }
    directory = output / "operating_transfer"
    write_csv(directory / "operating_transfer_cells.csv", cells)
    write_json(directory / "OPERATING_TRANSFER_EVIDENCE.json", evidence)
    return evidence


def balanced_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float(np.mean(predicted[true == label] == label)) for label in CLASSES]
    return float(np.mean(recalls))


def run_bootstrap(
    predictions: list[dict[str, object]], output: Path, replicates: int
) -> dict[str, object]:
    selected = [row for row in predictions if row["task"] == "three_class_diagnosis"]
    classes_by_bearing: dict[str, str] = {}
    for row in selected:
        classes_by_bearing[str(row["bearing_id"])] = str(row["true_label"])
    bearings_by_class: dict[str, list[str]] = defaultdict(list)
    for bearing, label in sorted(classes_by_bearing.items()):
        bearings_by_class[label].append(bearing)
    summaries: list[dict[str, object]] = []
    per_bearing: list[dict[str, object]] = []
    rng = np.random.default_rng(RNG_SEED)
    for modality in MODALITIES:
        by_split: dict[str, dict[str, list[dict[str, object]]]] = {}
        for split in SPLITS:
            mapping: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in selected:
                if row["modality"] == modality and row["split"] == split:
                    mapping[str(row["bearing_id"])].append(row)
            by_split[split] = mapping
        for bearing in sorted(classes_by_bearing):
            item: dict[str, object] = {
                "modality": modality, "bearing_id": bearing,
                "class_label": classes_by_bearing[bearing],
            }
            for split in SPLITS:
                records = by_split[split][bearing]
                item[f"accuracy__{split}"] = float(np.mean([
                    row["true_label"] == row["predicted_label"] for row in records
                ]))
            item["repetition_minus_specimen"] = float(item["accuracy__repetition_fold"]) - float(item["accuracy__specimen_fold"])
            per_bearing.append(item)
        observed: dict[str, float] = {}
        for split in SPLITS:
            records = [row for group in by_split[split].values() for row in group]
            observed[split] = balanced_accuracy(
                np.asarray([row["true_label"] for row in records]),
                np.asarray([row["predicted_label"] for row in records]),
            )
        bootstrap = np.empty((replicates, 3), dtype=float)
        for replicate in range(replicates):
            sampled: list[str] = []
            for label in CLASSES:
                sampled.extend(rng.choice(bearings_by_class[label], size=5, replace=True).tolist())
            values: list[float] = []
            for split in SPLITS:
                records = [row for bearing in sampled for row in by_split[split][bearing]]
                values.append(balanced_accuracy(
                    np.asarray([row["true_label"] for row in records]),
                    np.asarray([row["predicted_label"] for row in records]),
                ))
            bootstrap[replicate] = (values[0], values[1], values[0] - values[1])
        summary: dict[str, object] = {
            "modality": modality,
            "observed_repetition_balanced_accuracy": observed["repetition_fold"],
            "observed_specimen_balanced_accuracy": observed["specimen_fold"],
            "observed_paired_gap": observed["repetition_fold"] - observed["specimen_fold"],
        }
        for column, name in enumerate(("repetition_balanced_accuracy", "specimen_balanced_accuracy", "paired_gap")):
            summary.update({
                f"{name}_q025": float(np.quantile(bootstrap[:, column], 0.025)),
                f"{name}_median": float(np.quantile(bootstrap[:, column], 0.5)),
                f"{name}_q975": float(np.quantile(bootstrap[:, column], 0.975)),
            })
        summaries.append(summary)
    evidence = {
        "protocol": "unified-v2-bearing-bootstrap",
        "rng_seed": RNG_SEED,
        "replicates": replicates,
        "resampling_unit": "physical bearing stratified by true class",
        "summaries": summaries,
    }
    directory = output / "bearing_bootstrap"
    write_csv(directory / "per_bearing_accuracy.csv", per_bearing)
    write_json(directory / "BEARING_BOOTSTRAP_EVIDENCE.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output = SMOKE_OUTPUT if args.smoke else FULL_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = load_data()
    write_json(
        output / "RUN_ENVIRONMENT.json",
        {
            "protocol": "unified-v2-environment",
            "feature_table": {
                "path": str(FEATURE_TABLE), "bytes": FEATURE_TABLE.stat().st_size,
                "sha256": sha256(FEATURE_TABLE), "records": len(data.rows),
                "features": len(data.feature_names),
            },
            "environment": environment_record(),
            "mode": "smoke" if args.smoke else "full",
            "jobs": args.jobs,
        },
    )
    print("[1/7] unified baselines", flush=True)
    baseline, predictions = run_baselines(data, output)
    print("[2/7] balanced pseudolabel control", flush=True)
    pseudolabel = run_pseudolabels(data, output, 2 if args.smoke else PSEUDOLABEL_PERMUTATIONS, args.jobs)
    print("[3/7] class-conditional identity", flush=True)
    control_a = run_class_conditional_identity(data, output)
    print("[4/7] fault-orthogonal identity decoder", flush=True)
    control_b2 = run_orthogonal_decoder(data, predictions, output)
    print("[5/7] exhaustive partition sensitivity", flush=True)
    control_c = run_partition_sensitivity(data, output, args.jobs, 4 if args.smoke else None)
    print("[6/7] operating transfer", flush=True)
    transfer = run_operating_transfer(data, output)
    print("[7/7] bearing-level bootstrap", flush=True)
    bootstrap = run_bootstrap(predictions, output, 100 if args.smoke else BOOTSTRAP_REPLICATES)
    elapsed = time.time() - started
    summary = {
        "protocol": "unified-v2-all-evidence",
        "mode": "smoke" if args.smoke else "full",
        "elapsed_s": elapsed,
        "feature_sha256": sha256(FEATURE_TABLE),
        "baselines": baseline,
        "pseudolabels": pseudolabel,
        "control_a": control_a,
        "control_b2": control_b2,
        "control_c": control_c,
        "operating_transfer": transfer,
        "bearing_bootstrap": bootstrap,
    }
    write_json(output / "UNIFIED_V2_EVIDENCE.json", summary)
    (output / "UNIFIED_V2_COMPLETE.ok").write_text(
        f"unified V2 evidence complete; elapsed_s={elapsed:.3f}\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "mode": summary["mode"], "elapsed_s": elapsed}, indent=2))


if __name__ == "__main__":
    main()
