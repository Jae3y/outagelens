"""One-command launcher: installs requirements into the current interpreter and starts the web app.

    python setup_and_run.py            # install + run
    python setup_and_run.py --no-install
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    if "--no-install" not in sys.argv:
        core = [sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "numpy", "pandas", "networkx"]
        if subprocess.call(core) != 0:
            print("Core dependency install failed; check your network/pip configuration.")
            return 1
        # OpenDSS is optional: the app falls back to its NumPy physics engine if this fails.
        if subprocess.call([sys.executable, "-m", "pip", "install", "opendssdirect.py"]) != 0:
            print("opendssdirect.py could not be installed; OutageLens will use its built-in DistFlow engine.")
    subprocess.call([sys.executable, str(ROOT / "scripts" / "warm_cache.py")], cwd=ROOT)
    return subprocess.call([sys.executable, str(ROOT / "run.py")], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
