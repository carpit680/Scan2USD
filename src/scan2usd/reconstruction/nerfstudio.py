from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from scan2usd.config import SceneConfig
from scan2usd.reconstruction.external_cli import resolve_colmap, resolve_nerfstudio_cli


def _python_for_subprocess_shims() -> str:
    """Interpreter that owns site-packages (venv ``python``, not resolved ``/usr/bin/...``)."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        cand = Path(venv) / "bin" / "python"
        if cand.is_file():
            return str(cand)
    return sys.executable


def _ninja_shim_dir() -> Path:
    """
    Directory containing a ``ninja`` shell script that forwards to ``python -m ninja``.

    PyTorch's extension builder expects a ``ninja`` executable on ``PATH``; the pip ``ninja``
    package does not always place one in ``venv/bin``.
    """
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "scan2usd"
    base.mkdir(parents=True, exist_ok=True)
    d = base / "ninja-shim"
    d.mkdir(parents=True, exist_ok=True)
    shim = d / "ninja"
    py = _python_for_subprocess_shims()
    body = f'#!/bin/sh\nexec "{py}" -m ninja "$@"\n'
    if not shim.is_file() or shim.read_text() != body:
        shim.write_text(body)
        shim.chmod(0o755)
    return d


def _apply_nerfstudio_runtime_env(env: dict[str, str]) -> None:
    """PyTorch 2.6+ defaults weights_only=True on torch.load; Nerfstudio needs the legacy unpickler."""
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _augment_subprocess_path(env: dict[str, str]) -> None:
    """Prepend common CUDA ``bin`` and pip-ninja shim so gsplat JIT can build."""
    path = env.get("PATH", os.environ.get("PATH", ""))
    prefixes: list[str] = []
    for cuda in ("/usr/local/cuda/bin", "/usr/local/cuda-13.0/bin", "/usr/local/cuda-13/bin"):
        if (Path(cuda) / "nvcc").is_file():
            prefixes.append(cuda)
            break
    if not shutil.which("ninja", path=path):
        prefixes.append(str(_ninja_shim_dir()))
    if prefixes:
        env["PATH"] = os.pathsep.join([*prefixes, path])


def run_cmd(cmd: list[str | Path], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    argv = [str(x) for x in cmd]
    print("+", " ".join(argv))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    _apply_nerfstudio_runtime_env(merged)
    _augment_subprocess_path(merged)
    subprocess.run(argv, check=True, cwd=str(cwd) if cwd else None, env=merged)


def ns_process_data_images(
    cfg: SceneConfig,
    images_dir: Path,
    output_dir: Path,
) -> None:
    """
    Run Nerfstudio ``ns-process-data images`` (runs COLMAP internally).
    """
    argv0 = resolve_nerfstudio_cli(cfg, "ns_process_data", default_name="ns-process-data")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    pd = cfg.process_data
    cmd: list[str | Path] = [
        *argv0,
        "images",
        "--data",
        str(images_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--num-downscales",
        str(pd.num_downscales),
    ]
    run_cmd(cmd)


def ns_train_splatfacto(
    cfg: SceneConfig,
    data_dir: Path,
    *,
    max_num_iterations: int | None = None,
    experiment_name: str | None = None,
    enable_viewer: bool | None = None,
) -> Path:
    """Run ``ns-train splatfacto``. Returns path to config.yml under outputs."""
    sf = cfg.splatfacto
    iters = max_num_iterations if max_num_iterations is not None else sf.max_num_iterations
    exp_name = experiment_name if experiment_name is not None else sf.experiment_name
    use_viewer = enable_viewer if enable_viewer is not None else False

    argv0 = resolve_nerfstudio_cli(cfg, "ns_train", default_name="ns-train")
    out_root = cfg.workspace_dir / "ns_outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    cmd: list[str | Path] = [
        *argv0,
        "splatfacto",
        "--data",
        str(data_dir.resolve()),
        "--output-dir",
        str(out_root.resolve()),
        "--max-num-iterations",
        str(iters),
        "--experiment-name",
        exp_name,
        # Nerfstudio's default max_log_size=10 redraws a rolling table with ANSI cursor-up.
        # That often prints hundreds of bare headers when run as a subprocess (viewer banner,
        # terminal size, etc.). max_log_size=0 emits one line per log step instead.
        "--logging.local-writer.max-log-size",
        "0",
        "--logging.steps-per-log",
        str(sf.steps_per_log),
        *sf.ns_train_extra_args(),
    ]
    if use_viewer:
        cmd.extend(["--vis", "viewer", "--viewer.quit-on-train-completion", "True"])
    else:
        # No live viser server; tensorboard events still land under the run log dir.
        cmd.extend(["--vis", "tensorboard"])
    run_cmd(cmd)
    # Nerfstudio writes to output_dir / experiment_name / splatfacto / <timestamp> / config.yml
    candidates = sorted(out_root.glob(f"{exp_name}/splatfacto/*/config.yml"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No config.yml found under {out_root}")
    return candidates[-1]


def ns_render_camera_path(
    cfg: SceneConfig,
    load_config: Path,
    camera_path_json: Path,
    output_dir: Path,
) -> None:
    exe = resolve_nerfstudio_cli(cfg, "ns_render", default_name="ns-render")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *exe,
        "camera-path",
        "--load-config",
        str(load_config.resolve()),
        "--camera-path-filename",
        str(camera_path_json.resolve()),
        "--output-path",
        str(output_dir.resolve()),
        "--output-format",
        "images",
    ]
    run_cmd(cmd)


def export_colmap_txt_from_sparse(cfg: SceneConfig, sparse_dir: Path, out_dir: Path) -> None:
    from scan2usd.reconstruction.colmap_io import export_colmap_to_txt

    colmap = resolve_colmap(cfg)
    export_colmap_to_txt(sparse_dir, out_dir, colmap_bin=colmap)


def find_latest_splat_config(cfg: SceneConfig) -> Path | None:
    """``splat_config_path`` from YAML, else newest ``ns_outputs/.../config.yml``."""
    if cfg.splat_config_path is not None:
        p = Path(cfg.splat_config_path)
        if p.is_file():
            return p
    out_root = cfg.workspace_dir / "ns_outputs"
    exp = cfg.splatfacto.experiment_name
    candidates = sorted(
        out_root.glob(f"{exp}/splatfacto/*/config.yml"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def ns_viewer(cfg: SceneConfig, load_config: Path) -> None:
    """Run ``ns-viewer`` for an existing Splatfacto ``config.yml``."""
    argv0 = resolve_nerfstudio_cli(cfg, "ns_viewer", default_name="ns-viewer")
    run_cmd([*argv0, "--load-config", str(load_config.resolve())])


def ns_viewer_with_boxes(
    cfg: SceneConfig,
    load_config: Path,
    objects_npz: Path,
) -> None:
    """Open the splat in Nerfstudio's viewer with lifted 3D AABB wireframes overlaid."""
    from scan2usd.viz.viewer import run_splat_viewer_with_boxes

    run_splat_viewer_with_boxes(
        load_config.resolve(),
        objects_npz.resolve(),
        list(cfg.classes),
        ns_data_dir=cfg.nerfstudio_data_dir,
    )


def find_ns_colmap_sparse(ns_data_dir: Path) -> Path | None:
    """Typical path after ns-process-data: colmap/sparse/0."""
    p = ns_data_dir / "colmap" / "sparse" / "0"
    if p.is_dir():
        return p
    return None
