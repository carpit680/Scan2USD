"""Convert an ASCII USD layer to crate format using Isaac Sim's bundled OpenUSD."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Usd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.input))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(args.output)):
        raise RuntimeError(f"Could not export USD stage: {args.output}")


if __name__ == "__main__":
    main()
