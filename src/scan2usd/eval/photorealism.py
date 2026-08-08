"""Held-out render metrics for real-vs-reconstructed appearance."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Compare at the resolution the scene was rendered at, not the capture's.
#
# Renders used to be upscaled to the 4K reference, which is both slower — six 4K
# GaussianBlurs and a 3840x2160 tensor through AlexNet, per view — and unfair:
# upsampling blurs the render and charges it for detail it was never asked to
# produce. Downscaling the reference instead is the standard novel-view-synthesis
# comparison and costs a quarter of the pixels.
#
# It does change the absolute numbers, so every report records which mode
# produced it and nothing compares across the two silently.
EVAL_AT_RENDER_RESOLUTION = True


@dataclass(frozen=True)
class ImageMetrics:
    image: str
    render: str
    psnr: float
    ssim: float
    lpips: float | None


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def psnr(reference: np.ndarray, rendered: np.ndarray) -> float:
    mse = float(np.mean((reference - rendered) ** 2))
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def ssim(reference: np.ndarray, rendered: np.ndarray) -> float:
    """Windowed RGB SSIM with the standard 11×11 Gaussian kernel."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(rendered, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - mu_x**2
    sigma_y = cv2.GaussianBlur(rendered * rendered, (11, 11), 1.5) - mu_y**2
    sigma_xy = cv2.GaussianBlur(reference * rendered, (11, 11), 1.5) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(value))


@lru_cache(maxsize=1)
def _lpips_metric():
    """
    Build the LPIPS network once, on the GPU when there is room for it.

    Reused across images and across tuner trials. The card is shared with
    training and rendering, so this only claims the GPU when a comfortable
    margin is free — AlexNet at this resolution needs a few hundred MB, and
    losing a training run to reclaim a metric would be a poor trade.
    """
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        return None
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free > 2_000_000_000:
                return metric.cuda()
    except Exception:  # noqa: BLE001 — CPU is always a valid answer
        pass
    return metric


def _lpips(reference: np.ndarray, rendered: np.ndarray) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    metric = _lpips_metric()
    if metric is None:
        return None
    device = next(metric.parameters()).device
    ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).to(device)
    out = torch.from_numpy(rendered).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(metric(ref, out).item())


def _find_render(render_dir: Path, source_name: str) -> Path | None:
    stem = Path(source_name).stem
    candidates = (
        render_dir / source_name,
        render_dir / f"{stem}.png",
        render_dir / f"{stem}.jpg",
        render_dir / f"{stem}.jpeg",
    )
    return next((path for path in candidates if path.is_file()), None)


def align(
    reference: np.ndarray,
    rendered: np.ndarray,
    *,
    at_render_resolution: bool = EVAL_AT_RENDER_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Bring both images to a common size — see EVAL_AT_RENDER_RESOLUTION."""
    if rendered.shape == reference.shape:
        return reference, rendered
    if at_render_resolution:
        reference = cv2.resize(
            reference,
            (rendered.shape[1], rendered.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        rendered = cv2.resize(
            rendered,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return reference, rendered


def _score_pair(job: tuple[str, str, str, bool, bool]) -> dict:
    """One view, in a worker process. LPIPS stays out of workers (see below)."""
    source_name, reference_path, render_path, compute_lpips, at_render_res = job
    reference, rendered = align(
        _rgb(Path(reference_path)),
        _rgb(Path(render_path)),
        at_render_resolution=at_render_res,
    )
    return {
        "image": source_name,
        "render": render_path,
        "psnr": psnr(reference, rendered),
        "ssim": ssim(reference, rendered),
        "lpips": _lpips(reference, rendered) if compute_lpips else None,
    }


def evaluate_held_out_renders(
    held_out_manifest: Path,
    reference_images_dir: Path,
    render_dir: Path,
    *,
    output_path: Path,
    compute_lpips: bool = False,
    at_render_resolution: bool = EVAL_AT_RENDER_RESOLUTION,
    workers: int | None = None,
) -> dict:
    specification = json.loads(held_out_manifest.read_text(encoding="utf-8"))
    jobs: list[tuple[str, str, str, bool, bool]] = []
    missing: list[str] = []
    for item in specification.get("images", []):
        source_name = str(item["file"])
        reference_path = reference_images_dir / source_name
        render_path = _find_render(render_dir, source_name)
        if not reference_path.is_file() or render_path is None:
            missing.append(source_name)
            continue
        jobs.append(
            (
                source_name,
                str(reference_path),
                str(render_path.resolve()),
                False,  # LPIPS is scored in this process, not the workers
                at_render_resolution,
            )
        )

    # PSNR and SSIM parallelise cleanly; LPIPS does not, because every worker
    # would load its own copy of AlexNet and, if CUDA is in use, contend for the
    # same card. So the pool does the pixel maths and LPIPS runs here in one
    # batch against a single cached network.
    if workers is None:
        workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scored = list(pool.map(_score_pair, jobs, chunksize=1))
    else:
        scored = [_score_pair(job) for job in jobs]

    if compute_lpips:
        for entry, job in zip(scored, jobs, strict=True):
            reference, rendered = align(
                _rgb(Path(job[1])),
                _rgb(Path(job[2])),
                at_render_resolution=at_render_resolution,
            )
            entry["lpips"] = _lpips(reference, rendered)

    metrics = [ImageMetrics(**entry) for entry in scored]
    finite_psnr = [item.psnr for item in metrics if np.isfinite(item.psnr)]
    lpips_values = [item.lpips for item in metrics if item.lpips is not None]
    report = {
        "evaluated": len(metrics),
        "expected": len(specification.get("images", [])),
        "missing": missing,
        # Which convention produced these numbers. Scores from the two modes are
        # not comparable, and the difference is large enough to look like a
        # quality change if the mode is not recorded alongside them.
        "eval_resolution": "render" if at_render_resolution else "reference",
        "mean_psnr": float(np.mean(finite_psnr)) if finite_psnr else None,
        "mean_ssim": float(np.mean([item.ssim for item in metrics])) if metrics else None,
        "mean_lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "images": [asdict(item) for item in metrics],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
