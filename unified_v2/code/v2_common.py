from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in THREAD_ENV_VARS:
    os.environ[_name] = "1"

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
FEATURE_TABLE = REPOSITORY / "primary" / "derived" / "paderborn_record_features.csv"
RNG_SEED = 20260811
CLASSES = ("healthy", "real_inner_race", "real_outer_race")
MODALITIES = ("vibration", "current", "fusion")
SPLITS = ("repetition_fold", "specimen_fold")
META_COLUMNS = {
    "record_uid",
    "bearing_id",
    "class_label",
    "specimen_fold",
    "repetition_fold",
    "setting",
    "repeat",
}


@dataclass(frozen=True)
class DataBundle:
    rows: list[dict[str, str]]
    feature_names: list[str]
    matrix: np.ndarray
    modality_masks: dict[str, np.ndarray]
    diagnosis_target: np.ndarray
    bearing_target: np.ndarray
    split_values: dict[str, np.ndarray]
    settings: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> DataBundle:
    rows = list(csv.DictReader(FEATURE_TABLE.open(encoding="utf-8")))
    if len(rows) != 1200:
        raise ValueError(f"expected 1200 records, found {len(rows)}")
    feature_names = [name for name in rows[0] if name not in META_COLUMNS]
    matrix = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=float
    )
    masks = {
        "vibration": np.asarray(
            [name.startswith("vibration_1__") for name in feature_names]
        ),
        "current": np.asarray(
            [name.startswith("phase_current_") for name in feature_names]
        ),
        "fusion": np.ones(len(feature_names), dtype=bool),
    }
    if any(int(mask.sum()) == 0 for mask in masks.values()):
        raise ValueError("one or more modality masks are empty")
    return DataBundle(
        rows=rows,
        feature_names=feature_names,
        matrix=matrix,
        modality_masks=masks,
        diagnosis_target=np.asarray([row["class_label"] for row in rows]),
        bearing_target=np.asarray([row["bearing_id"] for row in rows]),
        split_values={
            name: np.asarray([int(row[name]) for row in rows]) for name in SPLITS
        },
        settings=np.asarray([row["setting"] for row in rows]),
    )


def model() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=RNG_SEED,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean_summary(values: Iterable[float], prefix: str = "") -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    key = f"{prefix}_" if prefix else ""
    result = {
        f"{key}mean": float(array.mean()),
        f"{key}min": float(array.min()),
        f"{key}median": float(np.median(array)),
        f"{key}max": float(array.max()),
    }
    if array.size > 1:
        result[f"{key}std"] = float(array.std(ddof=1))
        result[f"{key}q025"] = float(np.quantile(array, 0.025))
        result[f"{key}q975"] = float(np.quantile(array, 0.975))
    return result


def environment_record() -> dict[str, object]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": _package_version("pandas"),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "threadpool_info": threadpoolctl.threadpool_info(),
    }


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"
