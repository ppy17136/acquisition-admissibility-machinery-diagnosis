"""Metric-corrected pseudolabel control using predeclared ordinary accuracy.

V1 used balanced accuracy, but some frozen specimen test folds contain fewer
than all three randomly assigned pseudoclasses. Ordinary accuracy retains an
expected uniform-guessing baseline of one third regardless of that fold-level
class composition. Samples, assignments, features, model, and folds are
otherwise unchanged.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURE_TABLE = ROOT / "derived" / "paderborn_record_features.csv"
OUTPUT = ROOT / "results" / "pseudolabel_accuracy_v2"
META_COLUMNS = {
    "record_uid", "bearing_id", "class_label", "specimen_fold",
    "repetition_fold", "setting", "repeat",
}
RNG_SEED = 20260811
PERMUTATIONS = 200


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000, random_state=RNG_SEED
        ),
    )


def evaluate_one(
    permutation_id: int,
    modality: str,
    split_column: str,
    matrix: np.ndarray,
    target: np.ndarray,
    split_values: np.ndarray,
) -> dict[str, object]:
    scores: list[float] = []
    for fold in range(1, 6):
        test = split_values == fold
        estimator = model()
        estimator.fit(matrix[~test], target[~test])
        scores.append(float(accuracy_score(target[test], estimator.predict(matrix[test]))))
    return {
        "permutation_id": permutation_id,
        "modality": modality,
        "split": split_column,
        "fold1_accuracy": scores[0],
        "fold2_accuracy": scores[1],
        "fold3_accuracy": scores[2],
        "fold4_accuracy": scores[3],
        "fold5_accuracy": scores[4],
        "mean_accuracy": float(np.mean(scores)),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))
    feature_names = [name for name in rows[0] if name not in META_COLUMNS]
    full_matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    modalities = {
        "vibration": np.asarray(
            [name.startswith("vibration_1__") for name in feature_names]
        ),
        "current": np.asarray(
            [name.startswith("phase_current_") for name in feature_names]
        ),
        "fusion": np.ones(len(feature_names), dtype=bool),
    }
    bearing_ids = np.asarray([row["bearing_id"] for row in rows])
    bearings = np.asarray(sorted(set(bearing_ids)))
    split_values = {
        name: np.asarray([int(row[name]) for row in rows])
        for name in ("repetition_fold", "specimen_fold")
    }

    rng = np.random.default_rng(RNG_SEED)
    assignments: list[dict[str, object]] = []
    targets: dict[int, np.ndarray] = {}
    for permutation_id in range(1, PERMUTATIONS + 1):
        ordered = rng.permutation(bearings)
        mapping = {
            bearing: f"pseudo_{index // 5 + 1}"
            for index, bearing in enumerate(ordered)
        }
        targets[permutation_id] = np.asarray([mapping[bearing] for bearing in bearing_ids])
        for bearing in bearings:
            assignments.append(
                {
                    "permutation_id": permutation_id,
                    "bearing_id": bearing,
                    "pseudolabel": mapping[bearing],
                }
            )

    tasks = [
        (permutation_id, modality, split)
        for permutation_id in range(1, PERMUTATIONS + 1)
        for modality in modalities
        for split in split_values
    ]
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=8, verbose=5)(
            delayed(evaluate_one)(
                permutation_id,
                modality,
                split,
                full_matrix[:, modalities[modality]],
                targets[permutation_id],
                split_values[split],
            )
            for permutation_id, modality, split in tasks
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "balanced_pseudolabel_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    with (OUTPUT / "pseudolabel_permutation_accuracy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    summaries: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    for modality in modalities:
        by_split: dict[str, np.ndarray] = {}
        for split in split_values:
            values = np.asarray(
                [
                    float(row["mean_accuracy"])
                    for row in results
                    if row["modality"] == modality and row["split"] == split
                ]
            )
            by_split[split] = values
            summaries.append(
                {
                    "modality": modality,
                    "split": split,
                    "permutations": int(values.size),
                    "mean_accuracy": float(values.mean()),
                    "std_accuracy": float(values.std(ddof=1)),
                    "q025_accuracy": float(np.quantile(values, 0.025)),
                    "median_accuracy": float(np.median(values)),
                    "q975_accuracy": float(np.quantile(values, 0.975)),
                    "min_accuracy": float(values.min()),
                    "max_accuracy": float(values.max()),
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
        "protocol": "paderborn-split-unit-negative-controls-addendum-20260811",
        "implementation": "metric-corrected-v2-ordinary-accuracy",
        "v1_boundary": (
            "V1 balanced accuracy is retained as an implementation diagnostic; "
            "folds with absent pseudoclasses make it non-primary."
        ),
        "rng_seed": RNG_SEED,
        "balanced_permutations": PERMUTATIONS,
        "uniform_guessing_expected_accuracy": 1 / 3,
        "summaries": summaries,
        "paired_repetition_minus_specimen": paired,
    }
    (OUTPUT / "PSEUDOLABEL_ACCURACY_V2_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
