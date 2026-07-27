"""Held-out render metrics for real-vs-reconstructed appearance."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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
    """Build the LPIPS network once — it is reused across images and tuner trials."""
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        return None
    return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)


def _lpips(reference: np.ndarray, rendered: np.ndarray) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    metric = _lpips_metric()
    if metric is None:
        return None
    ref = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0)
    out = torch.from_numpy(rendered).permute(2, 0, 1).unsqueeze(0)
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


def evaluate_held_out_renders(
    held_out_manifest: Path,
    reference_images_dir: Path,
    render_dir: Path,
    *,
    output_path: Path,
    compute_lpips: bool = False,
) -> dict:
    specification = json.loads(held_out_manifest.read_text(encoding="utf-8"))
    metrics: list[ImageMetrics] = []
    missing: list[str] = []
    for item in specification.get("images", []):
        source_name = str(item["file"])
        reference_path = reference_images_dir / source_name
        render_path = _find_render(render_dir, source_name)
        if not reference_path.is_file() or render_path is None:
            missing.append(source_name)
            continue
        reference = _rgb(reference_path)
        rendered = _rgb(render_path)
        if rendered.shape != reference.shape:
            rendered = cv2.resize(
                rendered,
                (reference.shape[1], reference.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        metrics.append(
            ImageMetrics(
                image=source_name,
                render=str(render_path.resolve()),
                psnr=psnr(reference, rendered),
                ssim=ssim(reference, rendered),
                lpips=_lpips(reference, rendered) if compute_lpips else None,
            )
        )
    finite_psnr = [item.psnr for item in metrics if np.isfinite(item.psnr)]
    lpips_values = [item.lpips for item in metrics if item.lpips is not None]
    report = {
        "evaluated": len(metrics),
        "expected": len(specification.get("images", [])),
        "missing": missing,
        "mean_psnr": float(np.mean(finite_psnr)) if finite_psnr else None,
        "mean_ssim": float(np.mean([item.ssim for item in metrics])) if metrics else None,
        "mean_lpips": float(np.mean(lpips_values)) if lpips_values else None,
        "images": [asdict(item) for item in metrics],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
