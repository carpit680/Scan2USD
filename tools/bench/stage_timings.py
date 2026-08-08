#!/usr/bin/env python
"""
Measure what each pipeline stage actually costs, so optimisation is evidence-led.

Every performance claim on this project so far came from reading code or from
stopwatch impressions of a background job. Both are unreliable: the last
"efficiency win" turned out to be a model that had collapsed to 8,576
Gaussians, and the VRAM figure that suggested it was an improvement was the
symptom rather than the gain.

This records wall-clock and peak RSS per operation into a JSON file, so a
change can be shown to have helped rather than argued to have helped. Peak RSS
matters as much as time here: the constraint is a 30 GB box, and the failures
that actually stopped work were memory, not speed.

    workspace/isaac_env/bin/python tools/bench/stage_timings.py \
        --config configs/bedroom_scene.yaml --label before-phase1

Ops are chosen to be read-only and side-effect free, so this is safe to run
against a live workspace at any time.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def _peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux; it is a high-water mark for the process,
    # so deltas are only meaningful when the peak actually moves.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def measure(results: list[dict], name: str, **detail):
    before_peak = _peak_rss_mb()
    start = time.perf_counter()
    error = None
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — a failed op should still be recorded
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    after_peak = _peak_rss_mb()
    entry = {
        "op": name,
        "seconds": round(elapsed, 3),
        "peak_rss_mb": round(after_peak, 1),
        "peak_rss_growth_mb": round(max(0.0, after_peak - before_peak), 1),
        **detail,
    }
    if error:
        entry["error"] = error
    results.append(entry)
    status = "FAILED" if error else "ok"
    print(
        f"  {name:<34} {elapsed:>8.2f}s  peak {after_peak:>7.0f} MB  {status}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--label", default="baseline", help="Names this run in the JSON")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from scan2usd.config import SceneConfig

    cfg = SceneConfig.load(args.config)
    workspace = Path(cfg.workspace_dir)
    visual = workspace / "build" / "visual"
    sparse = Path(cfg.nerfstudio_data_dir) / "colmap" / "sparse" / "0"

    results: list[dict] = []
    print(f"[bench] {args.label} — {args.config.name}", flush=True)

    # 1. USD Gaussian array load. The suspected dominant cost of every splat
    #    operation, and the one that scales worst with model size.
    positions = None
    splat = visual / "environment_splat.usd"
    if splat.is_file():
        from scan2usd.reconstruction.splat_cleanup import (
            _find_particle_field_prim,
            _load_gaussian_arrays,
        )
        from pxr import Usd

        with measure(results, "usd_open_stage", file_mb=round(splat.stat().st_size / 1e6, 1)):
            stage = Usd.Stage.Open(str(splat))
            prim = _find_particle_field_prim(stage)
        with measure(results, "load_gaussian_arrays"):
            loaded = _load_gaussian_arrays(prim)
            positions = loaded["positions"]
        results[-1]["gaussians"] = int(len(positions))
        results[-1]["sh_element_size"] = int(loaded["sh_element_size"])

    # 2. Free-space carve, split so the ray-construction loop is visible
    #    separately from the sampling it feeds.
    if positions is not None and (sparse / "points3D.bin").is_file():
        from scan2usd.reconstruction import free_space as fs

        with measure(results, "colmap_read_images_bin"):
            fs.read_images_bin(sparse / "images.bin")
        with measure(results, "colmap_read_points3d_bin"):
            points, tracks = fs.read_points3d_bin(sparse / "points3D.bin")
        results[-1]["sfm_points"] = int(len(points))
        results[-1]["observations"] = int(sum(len(t) for t in tracks))

        with measure(results, "carve_from_colmap", max_rays=400_000):
            carve = fs.carve_from_colmap(
                positions, sparse, resolution=256, max_rays=400_000
            )
        results[-1]["grid_voxels"] = int(carve.grid.size)

        span = float(
            __import__("numpy").linalg.norm(carve.reference[1] - carve.reference[0])
        )
        with measure(results, "within_surface_radius"):
            fs.within_surface_radius(positions, carve.points, 0.015 * span)
        with measure(results, "inside_camera_hull"):
            fs.inside_camera_hull(positions, carve.centres)
        with measure(results, "hull_voxel_indices"):
            fs.hull_voxel_indices(carve.grid, carve.centres)

    # 3. COLMAP TXT parse — 270 MB of keypoints read to use the pose lines only.
    images_txt = Path(cfg.colmap_txt_dir) / "images.txt"
    if images_txt.is_file():
        from scan2usd.reconstruction.colmap_io import parse_images_txt

        with measure(
            results, "parse_images_txt", file_mb=round(images_txt.stat().st_size / 1e6, 1)
        ):
            parse_images_txt(images_txt)

    # 4. Held-out scoring, the largest single share of a tuner trial.
    renders = workspace / "build" / "heldout_renders"
    held_out = workspace / "build" / "grut_dataset" / "held_out.json"
    if renders.is_dir() and held_out.is_file() and any(renders.glob("*.png")):
        from scan2usd.eval.photorealism import evaluate_held_out_renders

        with measure(results, "score_held_out_views", lpips=True):
            report = evaluate_held_out_renders(
                held_out_manifest=held_out,
                reference_images_dir=Path(cfg.nerfstudio_data_dir) / "images",
                render_dir=renders,
                output_path=workspace / "build" / "bench_photorealism.json",
                compute_lpips=True,
            )
        results[-1]["views"] = report.get("evaluated_views")

    payload = {
        "label": args.label,
        "config": str(args.config),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {"python": platform.python_version(), "machine": platform.machine()},
        "ops": results,
        "total_seconds": round(sum(r["seconds"] for r in results), 2),
    }
    out = args.out or (workspace / "build" / "timings.json")
    history = []
    if out.is_file():
        try:
            existing = json.loads(out.read_text())
            history = existing if isinstance(existing, list) else [existing]
        except (OSError, ValueError):
            history = []
    history.append(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\n[bench] total {payload['total_seconds']}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
