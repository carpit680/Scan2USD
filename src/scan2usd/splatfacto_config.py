"""YAML ``splatfacto`` / ``process_data`` blocks → Nerfstudio CLI flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProcessDataConfig:
    """Passed to ``ns-process-data images``."""

    num_downscales: int = 3
    """Image pyramid depth (2**n); lower keeps more resolution for COLMAP / training."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProcessDataConfig:
        if not data:
            return cls()
        return cls(num_downscales=int(data.get("num_downscales", 3)))


@dataclass
class SplatfactoConfig:
    """Passed to ``ns-train splatfacto`` (see Nerfstudio SplatfactoModelConfig)."""

    max_num_iterations: int = 30_000
    experiment_name: str = "splatfacto"
    steps_per_log: int = 50
    use_bilateral_grid: bool = False
    background_color: str = "random"
    rasterize_mode: str = "classic"
    cull_alpha_thresh: float = 0.1
    ssim_lambda: float = 0.2
    camera_res_scale_factor: float | None = None
    """``pipeline.datamanager.camera-res-scale-factor`` (1.0 = full; lower if OOM)."""
    model_num_downscales: int | None = None
    """``pipeline.model.num-downscales`` (training multi-res schedule; default 2)."""
    camera_optimizer_mode: str = "off"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SplatfactoConfig:
        if not data:
            return cls()
        base = cls()
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {name: getattr(base, name) for name in field_names}
        for key, val in data.items():
            if key in field_names:
                kwargs[key] = val
        return cls(**kwargs)

    def ns_train_extra_args(self) -> list[str]:
        """Nerfstudio dotted CLI flags for ``ns-train splatfacto``."""
        out: list[str] = [
            "--pipeline.model.use-bilateral-grid",
            _bool_str(self.use_bilateral_grid),
            "--pipeline.model.background-color",
            self.background_color,
            "--pipeline.model.rasterize-mode",
            self.rasterize_mode,
            "--pipeline.model.cull-alpha-thresh",
            str(self.cull_alpha_thresh),
            "--pipeline.model.ssim-lambda",
            str(self.ssim_lambda),
        ]
        if self.camera_res_scale_factor is not None:
            out.extend(
                [
                    "--pipeline.datamanager.camera-res-scale-factor",
                    str(self.camera_res_scale_factor),
                ]
            )
        if self.model_num_downscales is not None:
            out.extend(
                [
                    "--pipeline.model.num-downscales",
                    str(self.model_num_downscales),
                ]
            )
        if self.camera_optimizer_mode != "off":
            out.extend(
                [
                    "--pipeline.model.camera-optimizer.mode",
                    self.camera_optimizer_mode,
                ]
            )
        return out


def _bool_str(v: bool) -> str:
    return "True" if v else "False"
