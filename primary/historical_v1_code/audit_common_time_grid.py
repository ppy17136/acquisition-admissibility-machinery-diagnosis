"""Verify that every record covers the frozen common 64 kHz, 4 s grid."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "analysis" / "signal_inventory" / "paderborn_record_inventory.csv"
OUTPUT = ROOT / "analysis" / "common_time_grid"
TARGET_HZ = 64_000
TARGET_SAMPLES = 256_000
TARGET_START = 0.0
TARGET_END = (TARGET_SAMPLES - 1) / TARGET_HZ


def main() -> None:
    rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    audits: list[dict[str, object]] = []
    errors: list[str] = []
    for row in rows:
        path = ROOT / row["relative_mat_path"]
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        root = payload[path.stem]
        channels = {str(item.Name): item for item in np.atleast_1d(root.Y)}
        xindex = int(channels["vibration_1"].XIndex)
        x = np.asarray(np.atleast_1d(root.X)[xindex - 1].Data, dtype=float)
        covers = bool(x[0] <= TARGET_START and x[-1] >= TARGET_END)
        if not covers:
            errors.append(
                f"{row['record_uid']}: x=[{x[0]:.12g},{x[-1]:.12g}] "
                f"does not cover [{TARGET_START},{TARGET_END}]"
            )
        audits.append(
            {
                "record_uid": row["record_uid"],
                "x_start_s": float(x[0]),
                "x_end_s": float(x[-1]),
                "source_samples": int(x.size),
                "median_dt_s": float(np.median(np.diff(x))),
                "covers_common_grid": covers,
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "common_time_grid_record_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    summary = {
        "target_sampling_hz": TARGET_HZ,
        "target_samples": TARGET_SAMPLES,
        "target_interval_s": [TARGET_START, TARGET_END],
        "records": len(audits),
        "records_covering_common_grid": sum(
            bool(row["covers_common_grid"]) for row in audits
        ),
        "source_sample_count_range": [
            min(int(row["source_samples"]) for row in audits),
            max(int(row["source_samples"]) for row in audits),
        ],
        "x_start_range_s": [
            min(float(row["x_start_s"]) for row in audits),
            max(float(row["x_start_s"]) for row in audits),
        ],
        "x_end_range_s": [
            min(float(row["x_end_s"]) for row in audits),
            max(float(row["x_end_s"]) for row in audits),
        ],
        "errors": errors,
        "passed": not errors and len(audits) == 1200,
    }
    (OUTPUT / "COMMON_TIME_GRID_AUDIT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
