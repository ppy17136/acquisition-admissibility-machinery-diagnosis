"""Build the frozen record-level time-frequency feature table."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parent
INVENTORY = ROOT / "analysis" / "signal_inventory" / "paderborn_record_inventory.csv"
OUTPUT = ROOT / "derived"
TARGET_HZ = 64_000.0
TARGET_SAMPLES = 256_000
TARGET_TIME = np.arange(TARGET_SAMPLES, dtype=np.float64) / TARGET_HZ
CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")
BANDS = (
    (0.0, 250.0),
    (250.0, 500.0),
    (500.0, 1_000.0),
    (1_000.0, 2_000.0),
    (2_000.0, 4_000.0),
    (4_000.0, 8_000.0),
    (8_000.0, 16_000.0),
    (16_000.0, 32_000.0),
)
EPS = np.finfo(np.float64).tiny


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band_name(low: float, high: float) -> str:
    return f"{int(low)}_{int(high)}hz"


def channel_features(signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    abs_signal = np.abs(signal)
    mean_abs = float(np.mean(abs_signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(abs_signal))
    root_abs_mean = float(np.mean(np.sqrt(abs_signal)))

    result = {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal, ddof=1)),
        "rms": rms,
        "mean_abs": mean_abs,
        "peak_abs": peak,
        "peak_to_peak": float(np.ptp(signal)),
        "skew": float(skew(signal, bias=False)),
        "excess_kurtosis": float(kurtosis(signal, fisher=True, bias=False)),
        "crest_factor": peak / max(rms, EPS),
        "impulse_factor": peak / max(mean_abs, EPS),
        "shape_factor": rms / max(mean_abs, EPS),
        "clearance_factor": peak / max(root_abs_mean * root_abs_mean, EPS),
    }
    for quantile in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        result[f"q{int(100 * quantile):02d}"] = float(np.quantile(signal, quantile))

    frequencies, psd = welch(
        signal,
        fs=TARGET_HZ,
        window="hann",
        nperseg=4096,
        noverlap=2048,
        detrend="constant",
        scaling="density",
    )
    total_power = float(np.trapezoid(psd, frequencies))
    for low, high in BANDS:
        if high == TARGET_HZ / 2:
            mask = (frequencies >= low) & (frequencies <= high)
        else:
            mask = (frequencies >= low) & (frequencies < high)
        power = float(np.trapezoid(psd[mask], frequencies[mask]))
        name = band_name(low, high)
        result[f"log10_power_{name}"] = math.log10(max(power, EPS))
        result[f"relative_power_{name}"] = power / max(total_power, EPS)

    psd_sum = float(np.sum(psd))
    probability = psd / max(psd_sum, EPS)
    centroid = float(np.sum(frequencies * probability))
    result["spectral_centroid_hz"] = centroid
    result["spectral_bandwidth_hz"] = float(
        np.sqrt(np.sum(np.square(frequencies - centroid) * probability))
    )
    nonzero_probability = probability[probability > 0]
    result["spectral_entropy"] = float(
        -np.sum(nonzero_probability * np.log(nonzero_probability))
        / np.log(probability.size)
    )
    result["spectral_flatness"] = float(
        np.exp(np.mean(np.log(psd + EPS))) / max(float(np.mean(psd)), EPS)
    )
    result["dominant_frequency_hz"] = float(frequencies[1 + np.argmax(psd[1:])])
    return result


def main() -> None:
    inventory = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    if len(inventory) != 1200:
        raise SystemExit(f"expected 1200 audited records, found {len(inventory)}")

    feature_rows: list[dict[str, object]] = []
    started = time.time()
    for index, metadata in enumerate(inventory, start=1):
        path = ROOT / metadata["relative_mat_path"]
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        root = payload[path.stem]
        channels = {str(item.Name): item for item in np.atleast_1d(root.Y)}
        axis_index = int(channels["vibration_1"].XIndex)
        source_time = np.asarray(np.atleast_1d(root.X)[axis_index - 1].Data, dtype=float)

        row: dict[str, object] = {
            "record_uid": metadata["record_uid"],
            "bearing_id": metadata["bearing_id"],
            "class_label": metadata["class_label"],
            "specimen_fold": int(metadata["specimen_fold"]),
            "repetition_fold": int(metadata["repetition_fold"]),
            "setting": metadata["setting"],
            "repeat": int(metadata["repeat"]),
        }
        for channel_name in CHANNELS:
            source_signal = np.asarray(channels[channel_name].Data, dtype=float)
            aligned = np.interp(TARGET_TIME, source_time, source_signal)
            for feature_name, value in channel_features(aligned).items():
                row[f"{channel_name}__{feature_name}"] = value
        feature_rows.append(row)

        if index % 50 == 0 or index == len(inventory):
            elapsed = time.time() - started
            print(f"features {index}/{len(inventory)} elapsed_s={elapsed:.1f}", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = OUTPUT / "paderborn_record_features.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    feature_names = [
        name
        for name in feature_rows[0]
        if name not in {
            "record_uid",
            "bearing_id",
            "class_label",
            "specimen_fold",
            "repetition_fold",
            "setting",
            "repeat",
        }
    ]
    summary = {
        "protocol": "paderborn-feature-baseline-spec-20260811",
        "records": len(feature_rows),
        "channels": list(CHANNELS),
        "features_per_channel": len(feature_names) // len(CHANNELS),
        "total_signal_features": len(feature_names),
        "target_sampling_hz": TARGET_HZ,
        "target_samples": TARGET_SAMPLES,
        "inventory_sha256": sha256(INVENTORY),
        "feature_table_sha256": sha256(table),
        "elapsed_s": time.time() - started,
        "all_features_finite": all(
            np.isfinite(float(row[name])) for row in feature_rows for name in feature_names
        ),
    }
    (OUTPUT / "PADERBORN_FEATURE_BUILD.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not summary["all_features_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
