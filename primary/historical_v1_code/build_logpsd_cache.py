"""Build the frozen full-record log-Welch-PSD cache for the confirmatory CNN."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "analysis" / "signal_inventory" / "paderborn_record_inventory.csv"
OUTPUT = ROOT / "derived"
TARGET_HZ = 64_000.0
TARGET_SAMPLES = 256_000
TARGET_TIME = np.arange(TARGET_SAMPLES, dtype=np.float64) / TARGET_HZ
CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    inventory = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    if len(inventory) != 1200:
        raise SystemExit(f"expected 1200 audited records, found {len(inventory)}")
    spectra = np.empty((1200, 3, 2049), dtype=np.float32)
    frequencies_reference: np.ndarray | None = None
    started = time.time()

    for index, metadata in enumerate(inventory):
        path = ROOT / metadata["relative_mat_path"]
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        root = payload[path.stem]
        channels = {str(item.Name): item for item in np.atleast_1d(root.Y)}
        axis_index = int(channels["vibration_1"].XIndex)
        source_time = np.asarray(np.atleast_1d(root.X)[axis_index - 1].Data, dtype=float)
        for channel_index, channel_name in enumerate(CHANNELS):
            source = np.asarray(channels[channel_name].Data, dtype=float)
            aligned = np.interp(TARGET_TIME, source_time, source)
            frequencies, psd = welch(
                aligned,
                fs=TARGET_HZ,
                window="hann",
                nperseg=4096,
                noverlap=2048,
                detrend="constant",
                scaling="density",
            )
            spectra[index, channel_index] = np.log10(
                psd.astype(np.float32) + np.finfo(np.float32).tiny
            )
            if frequencies_reference is None:
                frequencies_reference = frequencies.astype(np.float32)
            elif not np.array_equal(frequencies_reference, frequencies.astype(np.float32)):
                raise RuntimeError("frequency grid changed across records")
        if (index + 1) % 100 == 0:
            print(f"logpsd {index + 1}/1200 elapsed_s={time.time() - started:.1f}", flush=True)

    if not np.isfinite(spectra).all():
        raise SystemExit("nonfinite log-PSD value")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "paderborn_primary15_logpsd_float32.npz"
    np.savez_compressed(
        cache,
        log_psd=spectra,
        frequencies_hz=frequencies_reference,
        record_uid=np.asarray([row["record_uid"] for row in inventory]),
        bearing_id=np.asarray([row["bearing_id"] for row in inventory]),
        class_label=np.asarray([row["class_label"] for row in inventory]),
        specimen_fold=np.asarray([int(row["specimen_fold"]) for row in inventory], dtype=np.int8),
        repetition_fold=np.asarray([int(row["repetition_fold"]) for row in inventory], dtype=np.int8),
        setting=np.asarray([row["setting"] for row in inventory]),
        repeat=np.asarray([int(row["repeat"]) for row in inventory], dtype=np.int8),
    )
    summary = {
        "protocol": "paderborn-spectral-cnn-confirmatory-protocol-20260811",
        "records": 1200,
        "channels": list(CHANNELS),
        "shape": list(spectra.shape),
        "dtype": str(spectra.dtype),
        "frequency_bins": int(frequencies_reference.size),
        "frequency_range_hz": [float(frequencies_reference[0]), float(frequencies_reference[-1])],
        "all_finite": True,
        "cache_bytes": cache.stat().st_size,
        "cache_sha256": sha256(cache),
        "elapsed_s": time.time() - started,
    }
    (OUTPUT / "PADERBORN_LOGPSD_CACHE_AUDIT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
