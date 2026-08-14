"""Run all 90 frozen spectral-CNN fits sequentially on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 1001, 1002)
MODALITIES = ("vibration", "current", "fusion")
SPLITS = ("repetition_fold", "specimen_fold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    for seed in SEEDS:
        for split in SPLITS:
            for modality in MODALITIES:
                for fold in range(1, 6):
                    output = args.results / f"seed{seed}" / split / modality / f"fold{fold}"
                    marker = output / "CNN_FIT_COMPLETE.ok"
                    if marker.is_file():
                        print(f"SKIP {output}", flush=True)
                        continue
                    print(
                        f"START seed={seed} split={split} modality={modality} fold={fold}",
                        flush=True,
                    )
                    subprocess.run(
                        [
                            sys.executable,
                            str(root / "run_one_fit.py"),
                            "--cache", str(args.cache),
                            "--output", str(output),
                            "--seed", str(seed),
                            "--split", split,
                            "--modality", modality,
                            "--fold", str(fold),
                            "--maximum-epochs", "100",
                        ],
                        check=True,
                    )

    markers = list(args.results.glob("seed*/**/CNN_FIT_COMPLETE.ok"))
    if len(markers) != 90:
        raise SystemExit(f"expected 90 fit markers, found {len(markers)}")
    (args.results / "SPECTRAL_CNN_90_FITS_COMPLETE.ok").write_text(
        "90 frozen fits complete\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
