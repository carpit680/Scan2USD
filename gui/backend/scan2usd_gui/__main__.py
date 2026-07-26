"""CLI entry: ``scan2usd-gui`` or ``python -m scan2usd_gui``."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan2USD GUI API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (dev only)",
    )
    args = parser.parse_args()
    os.environ["SCAN2USD_GUI_PORT"] = str(args.port)
    os.environ["SCAN2USD_GUI_HOST"] = args.host
    import uvicorn

    uvicorn.run(
        "scan2usd_gui.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
