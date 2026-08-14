from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "unified_v2" / "code" / "verify_public_v2.py"


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(SCRIPT)]))
