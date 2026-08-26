from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
EVIDENCE = REPOSITORY / "unified_v2" / "evidence"
FEATURE_TABLE = REPOSITORY / "primary" / "derived" / "paderborn_record_features.csv"
EXPECTED_FEATURE_SHA256 = "82f5438e6fb83a3c1dfbbf78bd48c79722dcbe016d1b754c14cee07223729b7e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(relative: str) -> list[dict[str, str]]:
    return list(csv.DictReader((EVIDENCE / relative).open(encoding="utf-8")))


def assert_finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            assert_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def lookup(rows: list[dict], **criteria: object) -> dict:
    matched = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    assert len(matched) == 1, (criteria, len(matched))
    return matched[0]


def main() -> None:
    assert sha256(FEATURE_TABLE) == EXPECTED_FEATURE_SHA256
    evidence = json.loads((EVIDENCE / "UNIFIED_V2_EVIDENCE.json").read_text(encoding="utf-8"))
    assert_finite(evidence)
    assert evidence["feature_sha256"] == EXPECTED_FEATURE_SHA256
    expected_rows = {
        "baselines/fold_metrics.csv": 55,
        "baselines/record_predictions.csv": 13200,
        "pseudolabels/balanced_pseudolabel_assignments.csv": 3000,
        "pseudolabels/pseudolabel_permutation_accuracy.csv": 1200,
        "pseudolabels/pseudolabel_fault_association_sensitivity.csv": 600,
        "control_a_class_conditional_identity/fold_metrics.csv": 45,
        "control_b2_orthogonal_decoder/orthogonal_mapping_inventory.csv": 8280,
        "control_b2_orthogonal_decoder/orthogonal_identity_decoder.csv": 1656,
        "control_c_partition_sensitivity/unique_triplet_metrics.csv": 375,
        "control_c_partition_sensitivity/partition_metrics.csv": 43200,
        "operating_transfer/operating_transfer_cells.csv": 72,
        "bearing_bootstrap/per_bearing_accuracy.csv": 45,
    }
    for relative, expected in expected_rows.items():
        assert len(read_csv(relative)) == expected, relative
    for modality in ("vibration", "current", "fusion"):
        primary = lookup(
            evidence["baselines"]["summaries"],
            task="three_class_diagnosis", modality=modality, split="specimen_fold",
        )["mean_balanced_accuracy"]
        partition = lookup(evidence["control_c"]["summaries"], modality=modality)[
            "prespecified_primary_mean_balanced_accuracy"
        ]
        assert abs(float(primary) - float(partition)) <= 1e-15
    sensitivity = json.loads((EVIDENCE / "pseudolabels/PSEUDOLABEL_FAULT_ASSOCIATION_SENSITIVITY.json").read_text(encoding="utf-8"))
    assert sensitivity["minimum_cramers_v"] == 0.2
    assert sensitivity["minimum_cramers_v_permutation_count"] == 52
    assert all(abs(sensitivity["modality_summary"][m]["correlation_cramers_v_with_repetition_accuracy"]) < 0.09 for m in ("vibration", "current", "fusion"))
    assert all(
        float(row["fraction_positive"]) == 1.0
        for row in evidence["pseudolabels"]["paired_repetition_minus_specimen"]
    )
    assert min(float(row["accuracy_mean"]) for row in evidence["control_a"]["summaries"]) >= 0.95
    assert evidence["control_b2"]["orthogonal_mapping_count"] == 552
    assert min(float(row["accuracy_min"]) for row in evidence["control_b2"]["summaries"]) >= 0.97
    assert evidence["control_c"]["unique_class_balanced_test_triplets"] == 125
    assert evidence["control_c"]["admissible_partition_count"] == 14400
    print("Unified V2 verification passed: frozen feature hash, row counts, controls, and partition identities agree.")


if __name__ == "__main__":
    main()
