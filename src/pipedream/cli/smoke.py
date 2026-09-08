"""Run the repository-level LLVM smoke test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    smoke_script = repo_root / "scripts" / "smoke_test.sh"
    completed = subprocess.run(["bash", str(smoke_script)], cwd=repo_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
