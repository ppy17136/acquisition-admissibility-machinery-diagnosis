"""Portable launcher for a complete unified V2 rerun.

On Windows, some OpenBLAS wheels expose an invalid process-module path to
``threadpoolctl.threadpool_info``.  The model fits do not use that optional
probe.  This launcher replaces only the provenance probe on Windows and then
runs the unchanged unified analysis.
"""

from __future__ import annotations

import platform


if platform.system() == "Windows":
    import threadpoolctl

    threadpoolctl.threadpool_info = lambda: [
        {
            "status": "not enumerated",
            "reason": "Windows OpenBLAS DLL-path safety guard",
        }
    ]

import run_v2_evidence


if __name__ == "__main__":
    run_v2_evidence.main()
