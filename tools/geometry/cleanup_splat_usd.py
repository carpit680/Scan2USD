"""Isaac-backed ParticleField stray-Gaussian cleanup (needs OpenUSD UsdVol)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scan2usd.reconstruction.splat_cleanup import (  # noqa: E402
    SplatCleanupParams,
    cleanup_particlefield_file,
    write_report_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--outlier-std", type=float, default=4.0)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument("--raw-backup", type=Path, default=None)
    args = parser.parse_args()
    params = SplatCleanupParams(
        enabled=True,
        outlier_std=args.outlier_std,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
    )
    report = cleanup_particlefield_file(
        args.input,
        args.output,
        params,
        raw_backup_path=args.raw_backup,
    )
    write_report_json(report, args.report)
    print(json.dumps(report.to_dict()), flush=True)


if __name__ == "__main__":
    main()
