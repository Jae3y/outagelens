"""Pre-compute everything the server would otherwise build on its first requests.

    python scripts/warm_cache.py

Per feeder: trains and saves the GNN, calibrates the Bayesian posterior, and runs the default 280-outage
benchmark, all into .cache/. Run at Docker build time (see Dockerfile) or once after cloning, so the app
starts in seconds even on a small free-tier host.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import service  # noqa: E402


def main() -> None:
    for key in service.FEEDERS:
        t = time.time()
        service.system(key)  # engine cross-check, calibration and GNN (cached to disk if missing)
        service.benchmark(key)  # default benchmark (cached to disk)
        print(f"{key}: ready in {time.time() - t:.1f} s")
    print("cache:", ", ".join(sorted(p.name for p in service.CACHE_DIR.iterdir())))


if __name__ == "__main__":
    main()
