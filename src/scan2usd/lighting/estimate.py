"""Estimate a conservative dome light and exposure from captured images."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import SceneManifest


@dataclass
class LightingEstimate:
    dome_texture: str
    exposure_ev: float
    intensity: float
    white_balance: list[float]
    source: str
    confidence: float
    approved: bool = False


def _sample_capture(images_dir: Path, *, max_images: int = 64) -> tuple[np.ndarray, float]:
    paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise RuntimeError(f"No capture images under {images_dir}")
    step = max(1, len(paths) // max_images)
    samples: list[np.ndarray] = []
    for path in paths[::step][:max_images]:
        with Image.open(path) as raw:
            image = np.asarray(raw.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        unsaturated = np.all((image > 0.02) & (image < 0.98), axis=-1)
        values = image[unsaturated]
        if len(values):
            samples.append(values)
    if not samples:
        raise RuntimeError("Capture contains no usable unsaturated pixels for lighting estimation")
    srgb = np.concatenate(samples, axis=0)
    linear = np.power(srgb, 2.2)
    mean = np.mean(linear, axis=0)
    luminance = float(mean @ np.array([0.2126, 0.7152, 0.0722]))
    return mean, luminance


def _write_constant_dome(path: Path, linear_rgb: np.ndarray) -> None:
    srgb = np.power(np.clip(linear_rgb, 0.0, 1.0), 1.0 / 2.2)
    pixels = np.zeros((128, 256, 3), dtype=np.uint8)
    pixels[:] = np.round(srgb * 255.0).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def write_lighting_usda(path: Path, estimate: LightingEstimate) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dome_rel = Path(estimate.dome_texture).resolve().relative_to(path.parent.resolve()).as_posix()
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "Lighting"
)

def Scope "Lighting"
{{
    def DomeLight "CaptureDome"
    {{
        asset inputs:texture:file = @{dome_rel}@
        float inputs:intensity = {estimate.intensity:.8g}
        float inputs:exposure = {estimate.exposure_ev:.8g}
        token inputs:texture:format = "latlong"
    }}
}}
''',
        encoding="utf-8",
    )
    return path


def estimate_scene_lighting(
    cfg: SceneConfig,
    manifest: SceneManifest,
) -> tuple[LightingEstimate, Path]:
    output_dir = cfg.workspace_dir / "build" / "lighting"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.materials.hdr_path is not None:
        source = cfg.materials.hdr_path
        dome = output_dir / source.name
        if source.resolve() != dome.resolve():
            shutil.copy2(source, dome)
        estimate = LightingEstimate(
            dome_texture=str(dome.resolve()),
            exposure_ev=0.0,
            intensity=1.0,
            white_balance=[1.0, 1.0, 1.0],
            source="captured_hdr",
            confidence=0.9,
        )
    else:
        mean, luminance = _sample_capture(cfg.nerfstudio_data_dir / "images")
        target = 0.18
        exposure = math.log2(target / max(luminance, 1e-6))
        gray = float(np.mean(mean))
        white_balance = np.clip(gray / np.maximum(mean, 1e-5), 0.5, 2.0)
        dome = output_dir / "estimated_dome.png"
        balanced = np.clip(mean * white_balance, 0.0, 1.0)
        _write_constant_dome(dome, balanced)
        estimate = LightingEstimate(
            dome_texture=str(dome.resolve()),
            exposure_ev=exposure,
            intensity=1.0,
            white_balance=[float(value) for value in white_balance],
            source="ldr_capture_average",
            confidence=0.35,
        )
        manifest.warnings.append(
            "Lighting is estimated from LDR averages; review or provide materials.hdr_path"
        )
    usda = write_lighting_usda(output_dir / "lighting.usda", estimate)
    report = output_dir / "lighting_report.json"
    report.write_text(json.dumps(asdict(estimate), indent=2) + "\n", encoding="utf-8")
    manifest.register_artifact(
        artifact_id="scene_lighting",
        kind="usd_lighting",
        path=usda,
        producer="scan2usd.lighting",
        metadata=asdict(estimate),
    )
    return estimate, usda


def approve_lighting(
    manifest: SceneManifest,
    *,
    reviewer: str,
    notes: str = "",
) -> None:
    artifact = manifest.artifact("scene_lighting")
    if artifact is None:
        raise RuntimeError("No scene-lighting artifact to approve")
    manifest.approve("lighting", reviewer=reviewer, notes=notes)
    artifact.metadata["approved"] = True
