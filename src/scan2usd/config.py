from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scan2usd.splatfacto_config import ProcessDataConfig, SplatfactoConfig


# YAML keys whose values are filesystem paths (resolved against paths base).
# Other strings (e.g. ``external`` command names, class names) must NOT be coerced to Path.
_PATH_VALUE_KEYS: frozenset[str] = frozenset(
    {
        "video_path",
        "frames_dir",
        "workspace_dir",
        "colmap_txt_dir",
        "nerfstudio_data_dir",
        "splat_config_path",
        "renders_dir",
        "dataset_dir",
    }
)

# Standard layout under ``workspace_dir``. Omit these keys from YAML to inherit.
# Nested keys use dotted paths into the raw/config dict.
WORKSPACE_PATH_LAYOUT: dict[str, str] = {
    "frames_dir": "frames",
    "colmap_txt_dir": "colmap_txt",
    "nerfstudio_data_dir": "ns_data",
    "renders_dir": "renders",
    "dataset_dir": "dataset",
    "segmentation.masks_dir": "masks",
    "usd.output_dir": "usd",
}


def _coerce_path_field(val: Any, base: Path) -> Any:
    if val is None:
        return None
    if isinstance(val, Path):
        return val.expanduser().resolve()
    if not isinstance(val, str) or not val or val.startswith("${"):
        return val
    p = Path(val).expanduser()
    if not p.is_absolute():
        return (base / p).resolve()
    return p.resolve()


def _apply_path_fields(data: dict[str, Any], base: Path) -> dict[str, Any]:
    out = dict(data)
    for key in _PATH_VALUE_KEYS:
        if key not in out:
            continue
        out[key] = _coerce_path_field(out[key], base)
    return out


def _paths_base(config_path: Path, raw: dict[str, Any]) -> Path:
    """
    Base directory for resolving relative paths in YAML.

    Default: current working directory (so ``workspace/`` in a config under
    ``configs/`` resolves to ``<repo>/workspace``, not ``configs/workspace``).

    Set ``paths_relative_to: config`` to resolve relative to the YAML file's directory.
    Set ``paths_relative_to: /abs/path`` (or ``~/...``) to use an explicit base.
    """
    mode = raw.get("paths_relative_to")
    if mode == "config":
        return config_path.parent.resolve()
    if isinstance(mode, str) and mode not in ("", "cwd"):
        return Path(mode).expanduser().resolve()
    return Path.cwd().resolve()


def _nested_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    resolved = _coerce_path_field(value, base)
    return Path(resolved) if resolved is not None else None


def _raw_get(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _raw_has(data: dict[str, Any], dotted: str) -> bool:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _raw_del(data: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cur: Any = data
    stack: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        stack.append((cur, part))
        cur = cur[part]
    if not isinstance(cur, dict) or parts[-1] not in cur:
        return
    del cur[parts[-1]]
    # Drop empty nested dicts left behind (e.g. empty segmentation: {})
    for parent, key in reversed(stack):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break


def workspace_layout_paths(workspace_dir: Path | str) -> dict[str, str]:
    """Map dotted config keys → path strings under ``workspace_dir``."""
    root = Path(workspace_dir)
    return {key: str(root / leaf) for key, leaf in WORKSPACE_PATH_LAYOUT.items()}


def strip_workspace_derived_paths(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Remove standard layout path keys so they re-derive from ``workspace_dir`` on load.

    Use when the workspace folder changes, or when saving the form UI so YAML only
    needs ``workspace_dir`` set once. Explicit overrides can still be added in raw YAML.
    """
    out = dict(raw)
    for key in WORKSPACE_PATH_LAYOUT:
        _raw_del(out, key)
    return out


def sync_workspace_paths(
    raw: dict[str, Any],
    *,
    previous_workspace_dir: str | None = None,
) -> dict[str, Any]:
    """
    Keep workspace-relative paths coherent with ``workspace_dir``.

    - If ``workspace_dir`` changed vs ``previous_workspace_dir``, drop derived keys.
    - Drop derived keys whose values already match the default under the current workspace
      (redundant with inheritance).
    - Leave true overrides (non-default paths) intact.
    """
    out = dict(raw)
    ws = out.get("workspace_dir")
    if ws in (None, ""):
        ws = "workspace"
        out["workspace_dir"] = ws
    ws_str = str(ws)
    if previous_workspace_dir is not None and str(previous_workspace_dir) != ws_str:
        return strip_workspace_derived_paths(out)

    defaults = workspace_layout_paths(ws_str)
    for key, expected in defaults.items():
        if not _raw_has(out, key):
            continue
        current = _raw_get(out, key)
        if current in (None, ""):
            _raw_del(out, key)
            continue
        cur_s = str(current).rstrip("/\\")
        exp_s = str(expected).rstrip("/\\")
        if cur_s == exp_s:
            _raw_del(out, key)
            continue
        try:
            if Path(cur_s).expanduser().resolve() == Path(exp_s).expanduser().resolve():
                _raw_del(out, key)
        except OSError:
            pass
    return out


def _path_unset(data: dict[str, Any], key: str) -> bool:
    if key not in data:
        return True
    val = data[key]
    return val is None or val == ""


@dataclass
class CaptureConfig:
    """Input contract for a room/workcell capture."""

    modality: str = "rgb"
    depth_dir: Path | None = None
    lidar_path: Path | None = None
    calibration_path: Path | None = None
    clean_plate_dir: Path | None = None
    object_capture_dirs: dict[str, Path] = field(default_factory=dict)
    scale_anchor_m: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, base: Path) -> CaptureConfig:
        raw = data or {}
        object_dirs = {
            str(name): Path(_coerce_path_field(value, base))
            for name, value in dict(raw.get("object_capture_dirs") or {}).items()
        }
        return cls(
            modality=str(raw.get("modality", "rgb")).lower(),
            depth_dir=_nested_path(raw.get("depth_dir"), base),
            lidar_path=_nested_path(raw.get("lidar_path"), base),
            calibration_path=_nested_path(raw.get("calibration_path"), base),
            clean_plate_dir=_nested_path(raw.get("clean_plate_dir"), base),
            object_capture_dirs=object_dirs,
            scale_anchor_m=(
                None if raw.get("scale_anchor_m") is None else float(raw["scale_anchor_m"])
            ),
        )


@dataclass
class SplatCleanupConfig:
    """Post-export ParticleField stray-Gaussian cleanup."""

    enabled: bool = True
    # Measured on the kitchen scan (50k-iteration model, 940k Gaussians): 4.0 cut
    # 56% of them and tore dark holes in floors/walls, scoring 66.7; 8.0 kept the
    # geometry and scored 74.1 (PSNR 18.0 -> 23.3). The spatial filter is a single
    # global median-MAD sphere, so a fixed sigma gets more destructive as a model
    # densifies. Over-pruning wrecks large low-texture areas; under-pruning only
    # leaves a few floaters — so default to the safe side and tune per scene.
    outlier_std: float = 8.0
    min_opacity: float = 0.01
    max_scale: float | None = None
    # Targeted anti-floater filters. Held-out PSNR/SSIM cannot see these problems
    # (every held-out camera sits inside the capture path, where floaters hide
    # behind or beside the view), so they are governed by defaults rather than by
    # the tuner: giant Gaussians and the halo shell are never wanted.
    max_scale_frac: float | None = 0.08
    crop_margin: float | None = 0.5
    min_neighbors: int = 0
    neighbor_radius_frac: float = 0.01
    max_needle_ratio: float | None = None
    needle_min_length_frac: float = 0.005
    min_view_count: int = 0
    min_keep_fraction: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SplatCleanupConfig:
        raw = data or {}
        max_scale = raw.get("max_scale", None)
        scale_frac = raw.get("max_scale_frac", 0.08)
        crop = raw.get("crop_margin", 0.5)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            outlier_std=float(raw.get("outlier_std", 8.0)),
            min_opacity=float(raw.get("min_opacity", 0.01)),
            max_scale=None if max_scale is None else float(max_scale),
            max_scale_frac=None if scale_frac is None else float(scale_frac),
            crop_margin=None if crop is None else float(crop),
            min_neighbors=int(raw.get("min_neighbors", 0)),
            neighbor_radius_frac=float(raw.get("neighbor_radius_frac", 0.01)),
            max_needle_ratio=(None if raw.get("max_needle_ratio") is None else float(raw["max_needle_ratio"])),
            needle_min_length_frac=float(raw.get("needle_min_length_frac", 0.005)),
            min_view_count=int(raw.get("min_view_count", 0)),
            min_keep_fraction=float(raw.get("min_keep_fraction", 0.05)),
        )

    def to_params(self):
        from scan2usd.reconstruction.splat_cleanup import SplatCleanupParams

        return SplatCleanupParams(
            enabled=self.enabled,
            outlier_std=self.outlier_std,
            min_opacity=self.min_opacity,
            max_scale=self.max_scale,
            max_scale_frac=self.max_scale_frac,
            crop_margin=self.crop_margin,
            min_neighbors=self.min_neighbors,
            neighbor_radius_frac=self.neighbor_radius_frac,
            max_needle_ratio=self.max_needle_ratio,
            needle_min_length_frac=self.needle_min_length_frac,
            min_view_count=self.min_view_count,
            min_keep_fraction=self.min_keep_fraction,
        )


@dataclass
class ReconstructionConfig:
    """Visual and geometric reconstruction backend selection."""

    visual_backend: str = "3dgrut"
    rgbd_geometry_backend: str = "nvblox"
    rgb_geometry_backend: str = "openmvs"
    # Divide staged training images by this factor before 3DGRUT sees them.
    # Cost per iteration scales with pixels: a 4K frame is ~9x a 720p one, so 4K
    # at 50k iterations runs for many hours. 2 is a good balance on 8 GB.
    grut_downscale: int = 1
    # "standard" = UsdVol ParticleField3DGaussianSplat, which exposes per-Gaussian
    # positions/opacities/scales as USD attributes. Splat cleanup and the
    # splat-derived collision mesh both read those, so this must stay the default.
    # "nurec" is Omniverse's neural-volume container: it carries omni:nurec:crop
    # bounds and Isaac consumes it natively, but it hides the Gaussians inside
    # opaque OmniNuRecFieldAsset blobs and ALWAYS writes a USDZ regardless of the
    # requested extension. Use it for final delivery, not for processing.
    usd_splat_format: str = "standard"
    grut_config: str = "apps/colmap_3dgut.yaml"
    grut_max_iterations: int = 30000
    held_out_ratio: float = 0.1
    preview_allow_unobserved_background: bool = True
    # Fail `reconstruct` when COLMAP registers fewer than this fraction of the input
    # frames. A 2-of-113 reconstruction still produces artifacts that look "built"
    # but are fit to a couple of views. Set 0 to disable the gate.
    min_registration_rate: float = 0.5
    # Laplacian-variance floor for keeping a video frame. Resolution-dependent —
    # the same sharp frame scores ~20 at 720p and ~3.5 at 4K — so prefer
    # min_frame_features on high-resolution or mixed captures. 0 disables.
    blur_threshold: float = 50.0
    # Minimum SIFT keypoints (measured at a fixed 1280px) for a frame to be kept.
    # Rejects sharp-but-textureless frames (blank walls, ceilings) that cannot
    # register in COLMAP. Resolution-independent. 0 disables.
    min_frame_features: int = 0
    # Downscale extracted frames to this longest edge (0 = native). The distro
    # COLMAP has no CUDA, so SIFT is CPU-bound and cost scales with pixels;
    # COLMAP also clamps its own working size to 3200px. Extracting 4K at 1920
    # cut estimated feature extraction from 2.4 h to 0.9 h with no loss, since
    # training runs at that resolution anyway.
    frame_max_dim: int = 0
    splat_cleanup: SplatCleanupConfig = field(default_factory=SplatCleanupConfig)
    # Extra Hydra ``key=value`` overrides appended to 3DGRUT train.py (quality knobs).
    grut_overrides: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ReconstructionConfig:
        raw = data or {}
        overrides_raw = raw.get("grut_overrides") or []
        if isinstance(overrides_raw, str):
            overrides = [overrides_raw]
        else:
            overrides = [str(item) for item in overrides_raw]
        return cls(
            visual_backend=str(raw.get("visual_backend", "3dgrut")).lower(),
            rgbd_geometry_backend=str(raw.get("rgbd_geometry_backend", "nvblox")).lower(),
            rgb_geometry_backend=str(raw.get("rgb_geometry_backend", "openmvs")).lower(),
            grut_downscale=int(raw.get("grut_downscale", 1)),
            usd_splat_format=str(raw.get("usd_splat_format", "standard")).lower(),
            grut_config=str(raw.get("grut_config", "apps/colmap_3dgut.yaml")),
            grut_max_iterations=int(raw.get("grut_max_iterations", 30000)),
            held_out_ratio=float(raw.get("held_out_ratio", 0.1)),
            preview_allow_unobserved_background=bool(
                raw.get("preview_allow_unobserved_background", True)
            ),
            min_registration_rate=float(raw.get("min_registration_rate", 0.5)),
            blur_threshold=float(raw.get("blur_threshold", 50.0)),
            min_frame_features=int(raw.get("min_frame_features", 0)),
            frame_max_dim=int(raw.get("frame_max_dim", 0)),
            splat_cleanup=SplatCleanupConfig.from_dict(raw.get("splat_cleanup")),
            grut_overrides=overrides,
        )


@dataclass
class SegmentationConfig:
    proposal_model: str = "grounding-dino"
    mask_model: str = "sam2"
    masks_dir: Path | None = None
    min_views_per_object: int = 3
    review_required: bool = True
    # Allow environment-only builds (zero approved movables). Max-photorealism mode:
    # nothing is masked out of the splat and no placeholder object meshes are packaged.
    allow_no_objects: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, base: Path) -> SegmentationConfig:
        raw = data or {}
        return cls(
            proposal_model=str(raw.get("proposal_model", "grounding-dino")),
            mask_model=str(raw.get("mask_model", "sam2")),
            masks_dir=_nested_path(raw.get("masks_dir"), base),
            min_views_per_object=int(raw.get("min_views_per_object", 3)),
            review_required=bool(raw.get("review_required", True)),
            allow_no_objects=bool(raw.get("allow_no_objects", False)),
        )


@dataclass
class GeometryConfig:
    voxel_size_m: float = 0.02
    target_static_triangles: int = 500_000
    target_object_triangles: int = 100_000
    simplify_ratio: float = 0.35
    rgb_only_requires_scale: bool = True
    # Floor-plane sanity. The load-bearing check is max_points_below_floor: a real
    # floor has the scene resting on it. Inlier ratio is only a degeneracy guard —
    # a valid room floor is often just 5-10% of the points, so gating on it rejects
    # good scans (measured: good kitchen 6.1% vs broken desk 3.9% — indistinguishable,
    # while points-below separates them 2% vs 34%).
    max_points_below_floor: float = 0.10
    min_floor_inlier_ratio: float = 0.02
    # Used when reconstruction.rgb_geometry_backend == "splat": surface is fit to
    # the Gaussian centres, so it shares the splat's frame with no registration
    # step. Higher min_opacity keeps only Gaussians that sit on real surfaces.
    splat_mesh_depth: int = 10
    splat_mesh_min_opacity: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GeometryConfig:
        raw = data or {}
        return cls(
            voxel_size_m=float(raw.get("voxel_size_m", 0.02)),
            target_static_triangles=int(raw.get("target_static_triangles", 500_000)),
            target_object_triangles=int(raw.get("target_object_triangles", 100_000)),
            simplify_ratio=float(raw.get("simplify_ratio", 0.35)),
            rgb_only_requires_scale=bool(raw.get("rgb_only_requires_scale", True)),
            max_points_below_floor=float(raw.get("max_points_below_floor", 0.10)),
            min_floor_inlier_ratio=float(raw.get("min_floor_inlier_ratio", 0.02)),
            splat_mesh_depth=int(raw.get("splat_mesh_depth", 10)),
            splat_mesh_min_opacity=float(raw.get("splat_mesh_min_opacity", 0.3)),
        )


@dataclass
class MaterialConfig:
    texture_resolution: int = 4096
    author_baked_variant: bool = True
    author_pbr_variant: bool = True
    estimate_lighting: bool = True
    hdr_path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, base: Path) -> MaterialConfig:
        raw = data or {}
        return cls(
            texture_resolution=int(raw.get("texture_resolution", 4096)),
            author_baked_variant=bool(raw.get("author_baked_variant", True)),
            author_pbr_variant=bool(raw.get("author_pbr_variant", True)),
            estimate_lighting=bool(raw.get("estimate_lighting", True)),
            hdr_path=_nested_path(raw.get("hdr_path"), base),
        )


@dataclass
class PhysicsConfig:
    default_density_kg_m3: float = 700.0
    default_friction: float = 0.5
    default_restitution: float = 0.05
    dynamic_collider: str = "convexDecomposition"
    vhacd_max_hulls: int = 32
    vhacd_resolution: int = 100_000

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PhysicsConfig:
        raw = data or {}
        return cls(
            default_density_kg_m3=float(raw.get("default_density_kg_m3", 700.0)),
            default_friction=float(raw.get("default_friction", 0.5)),
            default_restitution=float(raw.get("default_restitution", 0.05)),
            dynamic_collider=str(raw.get("dynamic_collider", "convexDecomposition")),
            vhacd_max_hulls=int(raw.get("vhacd_max_hulls", 32)),
            vhacd_resolution=int(raw.get("vhacd_resolution", 100_000)),
        )


@dataclass
class UsdConfig:
    output_dir: Path | None = None
    root_filename: str = "scene.usd"
    isaac_version: str = "6.0"
    meters_per_unit: float = 1.0
    up_axis: str = "Z"
    default_look: str = "pbr"
    render_mode: str = "hybrid"
    binary_mesh_layers: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, base: Path) -> UsdConfig:
        raw = data or {}
        return cls(
            output_dir=_nested_path(raw.get("output_dir"), base),
            root_filename=str(raw.get("root_filename", "scene.usd")),
            isaac_version=str(raw.get("isaac_version", "6.0")),
            meters_per_unit=float(raw.get("meters_per_unit", 1.0)),
            up_axis=str(raw.get("up_axis", "Z")).upper(),
            default_look=str(raw.get("default_look", "pbr")).lower(),
            render_mode=str(raw.get("render_mode", "hybrid")).lower(),
            binary_mesh_layers=bool(raw.get("binary_mesh_layers", True)),
        )


@dataclass
class TuningConfig:
    """Auto-tuner budgets and parameter spaces (see scan2usd tune)."""

    max_cheap_trials: int = 12
    # Retrain trials re-run 3DGRUT training (hours each on GPU); opt-in.
    max_retrain_trials: int = 0
    lpips: bool = True
    # Optional {param: [values, …]} overrides of the built-in search spaces.
    cheap_params: dict[str, list[Any]] = field(default_factory=dict)
    retrain_params: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TuningConfig:
        raw = data or {}
        return cls(
            max_cheap_trials=int(raw.get("max_cheap_trials", 12)),
            max_retrain_trials=int(raw.get("max_retrain_trials", 0)),
            lpips=bool(raw.get("lpips", True)),
            cheap_params={
                str(k): list(v) for k, v in dict(raw.get("cheap_params") or {}).items()
            },
            retrain_params={
                str(k): list(v) for k, v in dict(raw.get("retrain_params") or {}).items()
            },
        )


@dataclass
class QAConfig:
    required: bool = True
    min_background_coverage: float = 0.9
    allow_background_holes: bool = False
    max_registration_error_m: float = 0.03
    max_depth_error_m: float = 0.05
    min_texture_coverage: float = 0.9
    max_dynamic_collider_faces: int = 20_000
    require_held_out_renders: bool = True
    require_isaac_validation: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> QAConfig:
        raw = data or {}
        return cls(
            required=bool(raw.get("required", True)),
            min_background_coverage=float(raw.get("min_background_coverage", 0.9)),
            allow_background_holes=bool(raw.get("allow_background_holes", False)),
            max_registration_error_m=float(raw.get("max_registration_error_m", 0.03)),
            max_depth_error_m=float(raw.get("max_depth_error_m", 0.05)),
            min_texture_coverage=float(raw.get("min_texture_coverage", 0.9)),
            max_dynamic_collider_faces=int(raw.get("max_dynamic_collider_faces", 20_000)),
            require_held_out_renders=bool(raw.get("require_held_out_renders", True)),
            require_isaac_validation=bool(raw.get("require_isaac_validation", True)),
        )


@dataclass
class SceneConfig:
    """Per-scene pipeline configuration (YAML)."""

    name: str = "default_scene"
    classes: list[str] = field(
        default_factory=lambda: [
            "chair",
            "table",
            "door",
            "couch",
            "cabinet",
            "obstacle",
            "box",
        ]
    )
    video_path: Path | None = None
    frames_dir: Path = field(default_factory=lambda: Path("workspace/frames"))
    workspace_dir: Path = field(default_factory=lambda: Path("workspace"))
    colmap_txt_dir: Path = field(default_factory=lambda: Path("workspace/colmap_txt"))
    nerfstudio_data_dir: Path = field(default_factory=lambda: Path("workspace/ns_data"))
    splat_config_path: Path | None = None
    renders_dir: Path = field(default_factory=lambda: Path("workspace/renders"))
    dataset_dir: Path = field(default_factory=lambda: Path("workspace/dataset"))
    yolo_model: str = "yolov8n.pt"
    train_epochs: int = 50
    train_imgsz: int = 640
    train_batch: int = 8
    seed: int = 42
    split: dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "session",
            "val_sessions": [],
            "val_ratio": 0.2,
        }
    )
    pose_sampling: dict[str, Any] = field(
        default_factory=lambda: {
            "num_poses": 200,
            "position_jitter_m": 0.05,
            "height_jitter_m": 0.02,
            "max_rotation_deg": 8.0,
            "interpolation_keyframes": 8,
        }
    )
    lift: dict[str, Any] = field(
        default_factory=lambda: {
            "min_points_in_box": 8,
            "merge_center_dist_m": 0.35,
        }
    )
    external: dict[str, str] = field(
        default_factory=lambda: {
            "colmap": "colmap",
            "ns_process_data": "ns-process-data",
            "ns_train": "ns-train",
            "ns_render": "ns-render",
            "grut_python": "python",
            "grut_root": "",
            "nvblox": "nvblox",
            "openmvs_interface": "InterfaceCOLMAP",
            "openmvs_densify": "DensifyPointCloud",
            "openmvs_reconstruct": "ReconstructMesh",
            "openmvs_refine": "RefineMesh",
            "openmvs_texture": "TextureMesh",
            "segmentation_python": "python",
            "sam2_runner": "",
            "object_reconstruction_runner": "",
            "isaac_python": "python.sh",
        }
    )
    process_data: ProcessDataConfig = field(default_factory=ProcessDataConfig)
    splatfacto: SplatfactoConfig = field(default_factory=SplatfactoConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    usd: UsdConfig = field(default_factory=UsdConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    @classmethod
    def load(cls, path: Path) -> SceneConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        data = dict(raw)
        data.pop("paths_relative_to", None)  # consumed by _paths_base, not a SceneConfig field
        base = _paths_base(path, raw)
        data = _apply_path_fields(data, base)
        return cls._from_dict(data, base=base)

    @classmethod
    def _from_dict(cls, data: dict[str, Any], *, base: Path | None = None) -> SceneConfig:
        base = (base or Path.cwd()).resolve()

        def p(key: str, default: Any) -> Any:
            return data[key] if key in data else default

        workspace_dir = Path(p("workspace_dir", Path("workspace")))
        if not workspace_dir.is_absolute():
            workspace_dir = (base / workspace_dir).resolve()

        def layout_path(key: str, leaf: str) -> Path:
            if _path_unset(data, key):
                return workspace_dir / leaf
            return Path(data[key])

        cfg = cls(
            name=str(p("name", "default_scene")),
            classes=list(p("classes", SceneConfig().classes)),
            video_path=p("video_path", None),
            frames_dir=layout_path("frames_dir", WORKSPACE_PATH_LAYOUT["frames_dir"]),
            workspace_dir=workspace_dir,
            colmap_txt_dir=layout_path(
                "colmap_txt_dir", WORKSPACE_PATH_LAYOUT["colmap_txt_dir"]
            ),
            nerfstudio_data_dir=layout_path(
                "nerfstudio_data_dir", WORKSPACE_PATH_LAYOUT["nerfstudio_data_dir"]
            ),
            splat_config_path=p("splat_config_path", None),
            renders_dir=layout_path("renders_dir", WORKSPACE_PATH_LAYOUT["renders_dir"]),
            dataset_dir=layout_path("dataset_dir", WORKSPACE_PATH_LAYOUT["dataset_dir"]),
            yolo_model=str(p("yolo_model", "yolov8n.pt")),
            train_epochs=int(p("train_epochs", 50)),
            train_imgsz=int(p("train_imgsz", 640)),
            train_batch=int(p("train_batch", 8)),
            seed=int(p("seed", 42)),
            split=dict(p("split", {})),
            pose_sampling=dict(p("pose_sampling", {})),
            lift=dict(p("lift", {})),
            external={**SceneConfig().external, **dict(p("external", {}))},
            process_data=ProcessDataConfig.from_dict(p("process_data", None)),
            splatfacto=SplatfactoConfig.from_dict(p("splatfacto", None)),
            capture=CaptureConfig.from_dict(p("capture", None), base),
            reconstruction=ReconstructionConfig.from_dict(p("reconstruction", None)),
            segmentation=SegmentationConfig.from_dict(p("segmentation", None), base),
            geometry=GeometryConfig.from_dict(p("geometry", None)),
            materials=MaterialConfig.from_dict(p("materials", None), base),
            physics=PhysicsConfig.from_dict(p("physics", None)),
            usd=UsdConfig.from_dict(p("usd", None), base),
            qa=QAConfig.from_dict(p("qa", None)),
            tuning=TuningConfig.from_dict(p("tuning", None)),
        )
        if cfg.usd.output_dir is None:
            cfg.usd.output_dir = (
                workspace_dir / WORKSPACE_PATH_LAYOUT["usd.output_dir"]
            )
        if cfg.segmentation.masks_dir is None:
            cfg.segmentation.masks_dir = (
                workspace_dir / WORKSPACE_PATH_LAYOUT["segmentation.masks_dir"]
            )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Fail early on contracts that would produce ambiguous or invalid assets."""
        if self.capture.modality not in {"rgb", "rgbd", "lidar"}:
            raise ValueError("capture.modality must be rgb, rgbd, or lidar")
        if not 0.0 <= self.reconstruction.held_out_ratio < 1.0:
            raise ValueError("reconstruction.held_out_ratio must be in [0, 1)")
        if self.segmentation.min_views_per_object < 1:
            raise ValueError("segmentation.min_views_per_object must be >= 1")
        if self.geometry.voxel_size_m <= 0:
            raise ValueError("geometry.voxel_size_m must be positive")
        if not 0.0 < self.geometry.simplify_ratio <= 1.0:
            raise ValueError("geometry.simplify_ratio must be in (0, 1]")
        if self.physics.default_density_kg_m3 <= 0:
            raise ValueError("physics.default_density_kg_m3 must be positive")
        if self.usd.up_axis not in {"Y", "Z"}:
            raise ValueError("usd.up_axis must be Y or Z")
        if self.usd.meters_per_unit <= 0:
            raise ValueError("usd.meters_per_unit must be positive")
        if self.usd.default_look not in {"baked", "pbr"}:
            raise ValueError("usd.default_look must be baked or pbr")
        for key in (
            "min_background_coverage",
            "min_texture_coverage",
        ):
            value = float(getattr(self.qa, key))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"qa.{key} must be in [0, 1]")
