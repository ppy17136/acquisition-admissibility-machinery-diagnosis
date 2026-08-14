"""GPU, cache, and protocol preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "paderborn_primary15_logpsd_float32.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cache = np.load(CACHE, allow_pickle=False)
    if cache["log_psd"].shape != (1200, 3, 2049):
        raise SystemExit(f"bad cache shape {cache['log_psd'].shape}")
    if not np.isfinite(cache["log_psd"]).all():
        raise SystemExit("cache has nonfinite values")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    evidence = {
        "cache_sha256": sha256(CACHE),
        "cache_shape": list(cache["log_psd"].shape),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "passed": True,
    }
    (ROOT / "GPU_PREFLIGHT_EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    (ROOT / "GPU_PREFLIGHT_COMPLETE.ok").write_text("GPU preflight complete\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
