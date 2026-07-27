"""Isaac-backed ParticleField stray-Gaussian cleanup (needs OpenUSD UsdVol)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
    parser.add_argument("--outlier-std", type=float, default=8.0)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--max-scale", type=float, default=None)
    parser.add_argument(
        "--max-scale-frac",
        type=float,
        default=None,
        help="Drop Gaussians wider than this fraction of the scene diagonal",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=None,
        help="Drop Gaussians beyond the observed volume expanded by this fraction",
    )
    parser.add_argument("--min-neighbors", type=int, default=0)
    parser.add_argument("--neighbor-radius-frac", type=float, default=0.01)
    parser.add_argument("--max-needle-ratio", type=float, default=None)
    parser.add_argument("--needle-min-length-frac", type=float, default=0.005)
    parser.add_argument("--min-view-count", type=int, default=0)
    parser.add_argument("--colmap-txt", type=Path, default=None)
    parser.add_argument("--observed-min", type=float, nargs=3, default=None)
    parser.add_argument("--observed-max", type=float, nargs=3, default=None)
    parser.add_argument("--raw-backup", type=Path, default=None)
    args = parser.parse_args()

    bounds = None
    if args.observed_min is not None and args.observed_max is not None:
        bounds = (np.asarray(args.observed_min), np.asarray(args.observed_max))

    params = SplatCleanupParams(
        enabled=True,
        outlier_std=args.outlier_std,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        max_scale_frac=args.max_scale_frac,
        crop_margin=args.crop_margin,
        min_neighbors=args.min_neighbors,
        neighbor_radius_frac=args.neighbor_radius_frac,
        max_needle_ratio=args.max_needle_ratio,
        needle_min_length_frac=args.needle_min_length_frac,
        min_view_count=args.min_view_count,
    )
    report = cleanup_particlefield_file(
        args.input,
        args.output,
        params,
        raw_backup_path=args.raw_backup,
        observed_bounds=bounds,
        colmap_txt_dir=args.colmap_txt,
    )
    write_report_json(report, args.report)
    print(json.dumps(report.to_dict()), flush=True)


if __name__ == "__main__":
    main()
