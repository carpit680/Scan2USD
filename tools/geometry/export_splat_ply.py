#!/usr/bin/env python
"""
ParticleField USD -> 3DGS PLY, so the cleaned scene can be viewed in a browser.

Checking a scene currently means launching Isaac: ~80 seconds of startup, the
whole GPU held, and no way to compare two cleanup settings side by side. A
browser viewer needs none of that, but every web viewer reads the 3DGS PLY
convention and nothing reads a ParticleField.

3DGRUT ships a PLY exporter, but it takes a live torch model, so it can only
export what training produced. The interesting artifact is the *cleaned* splat,
which exists only as USD -- previewing the raw model would defeat the purpose.

Two conversions matter, and both are silent when wrong. PLY stores
**pre-activation** values: opacity as a logit and scale as a log, which viewers
push through sigmoid and exp. USD stores them already activated. Copying either
across straight yields a scene that loads happily and looks wrong -- uniformly
opaque, or with every Gaussian collapsed to a point.

Run under Isaac's Python (it has ``pxr``)::

    workspace/isaac_env/bin/python tools/geometry/export_splat_ply.py \
        --stage workspace_bedroom/build/visual/environment_splat.usd \
        --out workspace_bedroom/build/visual/preview.ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Spherical-harmonics DC basis constant, used only to sanity check the colours.
SH_C0 = 0.28209479177387814


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _quaternion_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation part of a 4x4 as a (w, x, y, z) quaternion."""
    rotation = matrix[:3, :3]
    scale = np.linalg.norm(rotation, axis=0)
    rotation = rotation / np.where(scale > 1e-12, scale, 1.0)
    trace = float(np.trace(rotation))
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k, k])
        q = np.zeros(4)
        q[0] = (rotation[k, j] - rotation[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (rotation[j, i] + rotation[i, j]) / s
        q[k + 1] = (rotation[k, i] + rotation[i, k]) / s
        w, x, y, z = q
    return np.array([w, x, y, z], dtype=np.float64)


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z), with ``a`` applied after ``b``."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def load_transform(manifest: Path | None) -> np.ndarray | None:
    """COLMAP->USD similarity from the manifest, so the preview matches Isaac."""
    if manifest is None or not manifest.is_file():
        return None
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    for record in raw.get("transforms", []):
        if record.get("source_frame") == "colmap_world" and str(
            record.get("target_frame", "")
        ).startswith("usd_world"):
            return np.asarray(record["matrix"], dtype=np.float64)
    return None


def write_ply(path: Path, fields: dict[str, np.ndarray]) -> None:
    """Binary little-endian PLY with one float32 property per field."""
    count = len(next(iter(fields.values())))
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header += [f"property float {name}" for name in fields]
    header.append("end_header")
    dtype = np.dtype([(name, "<f4") for name in fields])
    packed = np.empty(count, dtype=dtype)
    for name, values in fields.items():
        packed[name] = values.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(packed.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="scene_manifest.json; applies the COLMAP->USD transform so the "
        "preview is Z-up with the floor at zero, like the Isaac scene.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=0,
        help="0 keeps only the DC colour (flat shading, ~4x smaller). 3 keeps "
        "full view-dependent colour and quadruples the download.",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=0,
        help="Randomly subsample to at most this many. 0 keeps all.",
    )
    args = parser.parse_args()

    from scan2usd.reconstruction.splat_cleanup import (
        _find_particle_field_prim,
        _load_gaussian_arrays,
    )

    from pxr import Usd

    stage = Usd.Stage.Open(str(args.stage))
    if stage is None:
        raise SystemExit(f"Could not open stage: {args.stage}")
    loaded = _load_gaussian_arrays(_find_particle_field_prim(stage))

    positions = np.asarray(loaded["positions"], dtype=np.float64)
    opacities = np.asarray(loaded["opacities"], dtype=np.float64).reshape(-1)
    scales = np.asarray(loaded["scales"], dtype=np.float64)
    rotations = np.asarray(loaded["orientations"], dtype=np.float64)
    sh = loaded["sh_coeffs"]
    element = int(loaded["sh_element_size"])

    if args.max_gaussians and len(positions) > args.max_gaussians:
        pick = np.random.default_rng(0).choice(
            len(positions), size=args.max_gaussians, replace=False
        )
        pick.sort()
        if sh is not None:
            # Index the SH block before positions, so `pick` still refers to the
            # original Gaussian ordering.
            sh = np.asarray(sh).reshape(len(positions), element, 3)[pick].reshape(-1, 3)
        positions, opacities, scales = positions[pick], opacities[pick], scales[pick]
        rotations = rotations[pick]

    transform = load_transform(args.manifest)
    if transform is not None:
        positions = positions @ transform[:3, :3].T + transform[:3, 3]
        rotations = _quaternion_multiply(_quaternion_matrix(transform), rotations)
        # Scale stays: the floor transform is a rigid rotation (unit scale).

    fields: dict[str, np.ndarray] = {
        "x": positions[:, 0],
        "y": positions[:, 1],
        "z": positions[:, 2],
        "nx": np.zeros(len(positions)),
        "ny": np.zeros(len(positions)),
        "nz": np.zeros(len(positions)),
    }

    count = len(positions)
    coefficients = (
        np.asarray(sh, dtype=np.float64).reshape(count, element, 3)
        if sh is not None
        else np.zeros((count, 1, 3))
    )
    for channel in range(3):
        fields[f"f_dc_{channel}"] = coefficients[:, 0, channel]

    wanted = (max(0, args.sh_degree) + 1) ** 2 - 1
    keep = min(wanted, element - 1)
    if keep > 0:
        # Channel-major, as every 3DGS PLY reader expects: all of R, then G, then B.
        rest = coefficients[:, 1 : keep + 1, :].transpose(0, 2, 1).reshape(count, keep * 3)
        for index in range(rest.shape[1]):
            fields[f"f_rest_{index}"] = rest[:, index]

    # PLY carries pre-activation values; USD carries activated ones.
    fields["opacity"] = _logit(opacities)
    for axis in range(3):
        fields[f"scale_{axis}"] = np.log(np.maximum(scales[:, axis], 1e-12))
    for index in range(4):
        fields[f"rot_{index}"] = rotations[:, index]

    write_ply(args.out, fields)

    colour = 0.5 + SH_C0 * coefficients[:, 0, :]
    print(
        json.dumps(
            {
                "gaussians": int(count),
                "sh_degree": int(np.sqrt(keep + 1) - 1) if keep else 0,
                "bytes": args.out.stat().st_size,
                "output": str(args.out.resolve()),
                # A plausible room is mostly mid-tone. Far outside 0..1 means the
                # DC coefficients were not what this assumed.
                "implied_rgb_p5_p95": [
                    round(float(v), 3) for v in np.percentile(colour, [5, 95])
                ],
                "opacity_logit_p5_p95": [
                    round(float(v), 2) for v in np.percentile(fields["opacity"], [5, 95])
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
