"""Build the frozen record inventory and verify public-benchmark gate metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MANIFEST = PROJECT / "protocol" / "paderborn_primary15_download_manifest.csv"
EXTRACTED = ROOT / "extracted"
OUTPUT = ROOT / "analysis" / "signal_inventory"
REQUIRED_SYNC = ("phase_current_1", "phase_current_2", "vibration_1")
NAME_PATTERN = re.compile(
    r"^(?P<setting>N\d{2}_M\d{2}_F\d{2})_"
    r"(?P<bearing>K(?:A|I)?\d{3})_(?P<repeat>\d+)\.mat$"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="audit only extracted bearings; final publication audit must omit this",
    )
    args = parser.parse_args()

    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    manifest = {row["bearing_id"]: row for row in manifest_rows}
    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for bearing_id, metadata in manifest.items():
        folder = EXTRACTED / bearing_id
        if not folder.is_dir():
            if not args.allow_partial:
                errors.append(f"missing extracted bearing directory: {bearing_id}")
            continue

        for path in sorted(folder.glob("*.mat")):
            match = NAME_PATTERN.match(path.name)
            if not match:
                errors.append(f"{bearing_id}: unexpected MAT filename {path.name}")
                continue
            if match.group("bearing") != bearing_id:
                errors.append(f"{bearing_id}: filename bearing mismatch {path.name}")
                continue
            try:
                payload = loadmat(path, squeeze_me=True, struct_as_record=False)
                public_keys = [key for key in payload if not key.startswith("__")]
                if public_keys != [path.stem]:
                    raise ValueError(f"public keys {public_keys} != [{path.stem}]")
                root = payload[public_keys[0]]
                channels = {str(item.Name): item for item in np.atleast_1d(root.Y)}
                missing = [name for name in REQUIRED_SYNC if name not in channels]
                if missing:
                    raise ValueError(f"missing required channels {missing}")

                sync = [channels[name] for name in REQUIRED_SYNC]
                rasters = [str(item.Raster) for item in sync]
                xindices = [int(item.XIndex) for item in sync]
                lengths = [int(np.asarray(item.Data).size) for item in sync]
                if len(set(rasters)) != 1 or len(set(xindices)) != 1 or len(set(lengths)) != 1:
                    raise ValueError(
                        f"sync mismatch raster={rasters}, xindex={xindices}, length={lengths}"
                    )
                if not all(np.isfinite(np.asarray(item.Data)).all() for item in sync):
                    raise ValueError("nonfinite value in required synchronous channel")

                axes = np.atleast_1d(root.X)
                x = np.asarray(axes[xindices[0] - 1].Data, dtype=float)
                if x.size != lengths[0] or not np.all(np.diff(x) > 0):
                    raise ValueError("invalid or mismatched synchronous time axis")
                duration_s = float(x[-1] - x[0])
                sampling_hz = float((x.size - 1) / duration_s)
                repeat = int(match.group("repeat"))
                setting = match.group("setting")
                measurement_datetime = datetime.strptime(
                    str(root.Description.General.DateTime), "%d.%m.%Y %H:%M:%S"
                ).isoformat()

                rows.append(
                    {
                        "record_uid": f"{bearing_id}__{setting}__r{repeat:02d}",
                        "bearing_id": bearing_id,
                        "class_label": metadata["class_label"],
                        "specimen_fold": int(metadata["fold"]),
                        "repetition_fold": ((repeat - 1) % 5) + 1,
                        "setting": setting,
                        "repeat": repeat,
                        "measurement_datetime_audit_only": measurement_datetime,
                        "measurement_id_audit_only": int(root.Info.MeasurementID),
                        "relative_mat_path": path.relative_to(ROOT).as_posix(),
                        "mat_bytes": path.stat().st_size,
                        "mat_sha256": file_sha256(path),
                        "samples": lengths[0],
                        "duration_s": duration_s,
                        "sampling_hz": sampling_hz,
                        "raster": rasters[0],
                        "xindex": xindices[0],
                    }
                )
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["bearing_id"]), str(row["setting"]), int(row["repeat"])))
    if rows:
        with (OUTPUT / "paderborn_record_inventory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    bearing_counts = Counter(str(row["bearing_id"]) for row in rows)
    class_counts = Counter(str(row["class_label"]) for row in rows)
    settings_by_bearing: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        settings_by_bearing[str(row["bearing_id"])][str(row["setting"])] += 1

    audited_bearings = sorted(bearing_counts)
    expected_records = 80 * (len(manifest) if not args.allow_partial else len(audited_bearings))
    passed = (
        not errors
        and len(rows) == expected_records
        and all(count == 80 for count in bearing_counts.values())
        and all(
            sorted(counter.values()) == [20, 20, 20, 20]
            for counter in settings_by_bearing.values()
        )
    )
    if not args.allow_partial:
        passed = passed and audited_bearings == sorted(manifest)

    summary = {
        "protocol": "paderborn-primary15-frozen-20260811",
        "partial_mode": args.allow_partial,
        "audited_bearings": audited_bearings,
        "bearing_count": len(audited_bearings),
        "record_count": len(rows),
        "bearing_record_counts": dict(sorted(bearing_counts.items())),
        "class_record_counts": dict(sorted(class_counts.items())),
        "required_synchronous_channels": list(REQUIRED_SYNC),
        "sampling_hz_range": [
            min((float(row["sampling_hz"]) for row in rows), default=None),
            max((float(row["sampling_hz"]) for row in rows), default=None),
        ],
        "duration_s_range": [
            min((float(row["duration_s"]) for row in rows), default=None),
            max((float(row["duration_s"]) for row in rows), default=None),
        ],
        "errors": errors,
        "passed": passed,
    }
    (OUTPUT / "PADERBORN_SIGNAL_INVENTORY_AUDIT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
