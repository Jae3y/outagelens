"""Start the OutageLens web control room.

    python run.py              # http://127.0.0.1:8000
    python run.py --port 9000
"""
from __future__ import annotations

import argparse
import os
import socket

import uvicorn


def free_port(host: str, start: int, tries: int = 20) -> int:
    """First port from ``start`` that nothing is listening on (Docker/WSL often hold 8000)."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise SystemExit(f"No free port between {start} and {start + tries - 1}; pass --port.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OutageLens web control room")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="default: $PORT, else the first free port from 8000")
    args = parser.parse_args()
    port = args.port or (int(os.environ["PORT"]) if os.environ.get("PORT") else free_port(args.host, 8000))
    print(f"OutageLens running at http://{args.host}:{port}  (Ctrl+C to stop)")
    uvicorn.run("server.api:app", host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
