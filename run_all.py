from __future__ import annotations

from verify_release import verify_cnn, verify_manifest
from unified_v2.code.verify_public_v2 import main as verify_v2


def main() -> None:
    verify_manifest()
    verify_cnn()
    verify_v2()
    print("Release verification passed: manifest, 90-fit CNN, and unified V2 evidence agree.")


if __name__ == "__main__":
    main()
