"""Smoke-test aggregation against the exact 90-fit directory contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import aggregate_results as aggregate


class AggregationContractTest(unittest.TestCase):
    def test_ninety_fit_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate.RESULTS = root / "results"
            aggregate.OUTPUT = root / "aggregate"
            for seed in (42, 1001, 1002):
                for split in ("repetition_fold", "specimen_fold"):
                    for modality in ("vibration", "current", "fusion"):
                        for fold in range(1, 6):
                            folder = (
                                aggregate.RESULTS / f"seed{seed}" / split /
                                modality / f"fold{fold}"
                            )
                            folder.mkdir(parents=True, exist_ok=True)
                            value = 0.9 if split == "repetition_fold" else 0.6
                            (folder / "fit_summary.json").write_text(
                                json.dumps(
                                    {
                                        "seed": seed,
                                        "modality": modality,
                                        "split": split,
                                        "test_fold": fold,
                                        "balanced_accuracy": value,
                                        "macro_f1": value,
                                        "accuracy": value,
                                        "elapsed_s": 1.0,
                                    }
                                ),
                                encoding="utf-8",
                            )
            aggregate.main()
            evidence = json.loads(
                (aggregate.OUTPUT / "SPECTRAL_CNN_CONFIRMATORY_EVIDENCE.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(len(evidence["summaries"]), 6)
            self.assertEqual(len(evidence["paired_gaps"]), 3)
            self.assertTrue(
                all(abs(row["mean_repetition_minus_specimen"] - 0.3) < 1e-12
                    for row in evidence["paired_gaps"])
            )


if __name__ == "__main__":
    unittest.main()
