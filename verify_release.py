from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ("healthy", "real_outer_race", "real_inner_race")
SEEDS = (42, 1001, 1002)
MODALITIES = ("vibration", "current", "fusion")
SPLITS = ("repetition_fold", "specimen_fold")
FOLDS = (1, 2, 3, 4, 5)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 5e-12) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    true = [r["true_label"] for r in rows]
    pred = [r["predicted_label"] for r in rows]
    accuracy = sum(a == b for a, b in zip(true, pred)) / len(rows)
    recalls, f1s = [], []
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(true, pred))
        fn = sum(a == label and b != label for a, b in zip(true, pred))
        fp = sum(a != label and b == label for a, b in zip(true, pred))
        recalls.append(tp / (tp + fn))
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": accuracy, "balanced_accuracy": statistics.mean(recalls), "macro_f1": statistics.mean(f1s)}


def verify_manifest() -> None:
    with (ROOT / "SHA256SUMS.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    actual = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "SHA256SUMS.csv"
        and "__pycache__" not in p.parts
        and p.suffix.lower() != ".pyc"
    )
    assert len(rows) == len(actual), (len(rows), len(actual))
    for row in rows:
        path = ROOT / Path(row["path"])
        assert path.is_file(), path
        assert path.stat().st_size == int(row["bytes"]), path
        assert sha(path) == row["sha256"], path


def verify_cnn() -> None:
    base = ROOT / "cnn/evidence"
    paths = sorted((base / "results").glob("seed*/*/*/fold*/fit_summary.json"))
    assert len(paths) == 90, len(paths)
    expected = {(s, p, m, f) for s in SEEDS for p in SPLITS for m in MODALITIES for f in FOLDS}
    summaries, uid_sets = {}, {}
    prediction_rows = 0
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        key = (int(d["seed"]), d["split"], d["modality"], int(d["test_fold"]))
        assert key not in summaries and d["device"] == "cuda"
        summaries[key] = d
        pred_path = path.parent / "test_predictions.csv"
        with pred_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        prediction_rows += len(rows)
        assert len(rows) == d["test_rows"] == 240
        assert len({r["record_uid"] for r in rows}) == 240
        uid_sets[key] = {r["record_uid"] for r in rows}
        calc = metrics(rows)
        assert all(close(calc[k], float(d[k])) for k in calc), key
        for row in rows:
            probs = [float(row[f"p_{label}"]) for label in LABELS]
            assert all(math.isfinite(x) for x in probs) and abs(sum(probs) - 1) < 2e-6
    assert set(summaries) == expected and prediction_rows == 21600
    for split in SPLITS:
        for fold in FOLDS:
            ref = uid_sets[(42, split, "vibration", fold)]
            assert all(uid_sets[(seed, split, modality, fold)] == ref for seed in SEEDS for modality in MODALITIES)
    published = json.loads((base / "aggregate/SPECTRAL_CNN_CONFIRMATORY_EVIDENCE.json").read_text(encoding="utf-8"))
    pub = {x["modality"]: x for x in published["paired_gaps"]}
    for modality in MODALITIES:
        gaps = [summaries[(s, "repetition_fold", modality, f)]["balanced_accuracy"] - summaries[(s, "specimen_fold", modality, f)]["balanced_accuracy"] for s in SEEDS for f in FOLDS]
        assert len(gaps) == 15 and all(x > 0 for x in gaps)
        assert close(statistics.mean(gaps), pub[modality]["mean_repetition_minus_specimen"])
        assert close(min(gaps), pub[modality]["min"])


def main() -> None:
    verify_manifest()
    verify_primary()
    verify_cnn()
    print("Release verification passed: hashes, primary evidence, and 90-fit CNN confirmation agree.")


if __name__ == "__main__":
    main()
