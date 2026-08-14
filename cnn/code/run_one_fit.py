"""Run one frozen seed/modality/split/fold spectral-CNN fit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from spectral_cnn import FrozenSpectralCNN, fit, loader, predict, seed_everything


CLASS_TO_INT = {"healthy": 0, "real_outer_race": 1, "real_inner_race": 2}
INT_TO_CLASS = {value: key for key, value in CLASS_TO_INT.items()}
MODALITY_CHANNELS = {"vibration": [0], "current": [1, 2], "fusion": [0, 1, 2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--modality", choices=sorted(MODALITY_CHANNELS), required=True)
    parser.add_argument("--split", choices=("repetition_fold", "specimen_fold"), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "CNN_FIT_COMPLETE.ok"
    if marker.is_file():
        print(f"SKIP completed {args.output}")
        return
    seed_everything(args.seed)
    started = time.time()
    cache = np.load(args.cache, allow_pickle=False)
    channels = MODALITY_CHANNELS[args.modality]
    x = cache["log_psd"][:, channels, :].astype(np.float32, copy=False)
    y = np.asarray([CLASS_TO_INT[str(label)] for label in cache["class_label"]], dtype=np.int64)
    split_values = cache[args.split].astype(int)
    validation_fold = args.fold % 5 + 1
    test_mask = split_values == args.fold
    validation_mask = split_values == validation_fold
    train_mask = ~(test_mask | validation_mask)

    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    train_loader = loader(x[train_mask], y[train_mask], args.seed, True)
    validation_loader = loader(x[validation_mask], y[validation_mask], args.seed, False)
    test_loader = loader(x[test_mask], y[test_mask], args.seed, False)

    if os.environ.get("PADERBORN_REQUIRE_CUDA", "1") == "1" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = FrozenSpectralCNN(len(channels)).to(device)
    history, best_epoch = fit(
        network, train_loader, validation_loader, device,
        maximum_epochs=args.maximum_epochs, patience=12
    )
    probability, true = predict(network, test_loader, device)
    predicted = probability.argmax(axis=1)

    test_indices = np.flatnonzero(test_mask)
    with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "record_uid", "bearing_id", "setting", "true_label", "predicted_label",
            "p_healthy", "p_real_outer_race", "p_real_inner_race"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for local, global_index in enumerate(test_indices):
            writer.writerow(
                {
                    "record_uid": str(cache["record_uid"][global_index]),
                    "bearing_id": str(cache["bearing_id"][global_index]),
                    "setting": str(cache["setting"][global_index]),
                    "true_label": INT_TO_CLASS[int(true[local])],
                    "predicted_label": INT_TO_CLASS[int(predicted[local])],
                    "p_healthy": float(probability[local, 0]),
                    "p_real_outer_race": float(probability[local, 1]),
                    "p_real_inner_race": float(probability[local, 2]),
                }
            )
    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "seed": args.seed,
        "modality": args.modality,
        "split": args.split,
        "test_fold": args.fold,
        "validation_fold": validation_fold,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro")),
        "accuracy": float(np.mean(true == predicted)),
        "device": str(device),
        "elapsed_s": time.time() - started,
    }
    (args.output / "fit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    marker.write_text("CNN_FIT_COMPLETE\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
