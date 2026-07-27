"""Dump Gaussian centres from a ParticleField/NuRec USD as a PLY point cloud.

Runs under Isaac's Python (it needs the OpenUSD volume schemas); the surface
reconstruction itself happens in the scan2usd venv, which has Open3D.

The point of this is alignment. Deriving collision geometry from a separate
multi-view-stereo run means registering two independent reconstructions to each
other, and any error there puts the robot's contacts in a different place from
what it sees — on the kitchen scan that drift measured 12.6 units. The Gaussian
centres already sit on the scene's surfaces in the same frame as the splat, so a
mesh built from them is aligned by construction, with no registration step to get
wrong. This is the approach Niantic describe for Scaniverse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _vt_to_numpy(values) -> np.ndarray:
    sample = values[0]
    if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
        return np.asarray([[float(x) for x in item] for item in values], dtype=np.float64)
    return np.asarray([float(x) for x in values], dtype=np.float64)


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    points = np.ascontiguousarray(points, dtype=np.float32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Splat USD/USDZ")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply")
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.3,
        help="Only keep Gaussians at least this opaque. Faint Gaussians model haze "
        "rather than surfaces, and would pull the reconstructed surface off the wall.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1_500_000,
        help="Uniformly subsample above this count to bound Poisson memory.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    from pxr import Usd

    stage = Usd.Stage.Open(str(args.input.resolve()))
    prim = None
    for candidate in stage.Traverse():
        type_name = str(candidate.GetTypeName())
        if ("ParticleField" in type_name or type_name == "Volume") and candidate.GetAttribute(
            "positions"
        ).HasValue():
            prim = candidate
            break
    if prim is None:
        raise SystemExit(f"No Gaussian positions found in {args.input}")

    positions = _vt_to_numpy(prim.GetAttribute("positions").Get())
    opacity_attr = prim.GetAttribute("opacities")
    total = len(positions)
    if opacity_attr and opacity_attr.HasValue():
        opacities = _vt_to_numpy(opacity_attr.Get()).reshape(-1)
        keep = opacities >= args.min_opacity
        positions = positions[keep]

    subsampled = False
    if len(positions) > args.max_points:
        step = np.linspace(0, len(positions) - 1, args.max_points, dtype=np.int64)
        positions = positions[step]
        subsampled = True

    write_binary_ply(args.output, positions)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "total_gaussians": int(total),
        "kept_points": int(len(positions)),
        "min_opacity": args.min_opacity,
        "subsampled": subsampled,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
