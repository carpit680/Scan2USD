"""Dependency checks for ``scan2usd doctor`` (binaries, Python imports, Linux apt hints)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import typer

from scan2usd.config import SceneConfig
from scan2usd.reconstruction.external_cli import resolve_colmap, resolve_nerfstudio_cli


def _linux() -> bool:
    return sys.platform.startswith("linux")


@dataclass
class ItemResult:
    """One line under External tools or Python environment."""

    label: str
    ok: bool
    detail: str
    apt_if_missing: tuple[str, ...] = ()
    required: bool = True
    pip_hint: str | None = None


def _apt_install_line(packages: tuple[str, ...]) -> str:
    pkgs = " ".join(sorted(set(packages)))
    return f"sudo apt update && sudo apt install -y {pkgs}"


def _check_binary(which_name: str, *, apt: tuple[str, ...], label: str | None = None) -> ItemResult:
    path = shutil.which(which_name)
    lab = label or which_name
    if path:
        return ItemResult(lab, True, path, (), True)
    return ItemResult(lab, False, "MISSING", apt, True)


def _check_import(
    module: str,
    *,
    label: str,
    version_attr: str = "__version__",
    apt_on_failure: tuple[str, ...] = (),
    pip_hint: str | None = None,
    required: bool = True,
) -> ItemResult:
    try:
        m = __import__(module, fromlist=["_"])
    except Exception as e:  # noqa: BLE001 — surface any import error to the user
        return ItemResult(label, False, f"import failed: {e}", apt_on_failure, required, pip_hint)
    ver = getattr(m, version_attr, None)
    detail = str(ver) if ver is not None else "import OK"
    return ItemResult(label, True, detail, (), required, None)


def _check_opencv() -> ItemResult:
    """
    OpenCV wheels often need system libs on minimal Debian/Ubuntu images.

    See also: https://github.com/opencv/opencv-python
    """
    label = "opencv (cv2)"
    pip_hint = "pip install opencv-python-headless (already a scan2usd dependency)"
    apt_libs = (
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
    )
    try:
        import cv2  # noqa: PLC0415

        return ItemResult(label, True, cv2.__version__, (), True, None)
    except Exception as e:  # noqa: BLE001
        return ItemResult(label, False, f"import failed: {e}", apt_libs, True, pip_hint)


def _colmap_item(cfg: SceneConfig) -> ItemResult:
    try:
        c = resolve_colmap(cfg)
        return ItemResult("colmap", True, c, (), True)
    except FileNotFoundError:
        return ItemResult("colmap", False, "MISSING", ("colmap",), True)


def _nerfstudio_cli_items(cfg: SceneConfig) -> list[ItemResult]:
    specs = [
        ("ns_process_data", "ns-process-data"),
        ("ns_train", "ns-train"),
        ("ns_render", "ns-render"),
    ]
    out: list[ItemResult] = []
    for key, default in specs:
        try:
            argv = resolve_nerfstudio_cli(cfg, key, default_name=default)
            joined = " ".join(argv)
            out.append(ItemResult(key, True, f"{joined} …", (), True))
        except FileNotFoundError:
            out.append(
                ItemResult(
                    key,
                    False,
                    "MISSING",
                    (),
                    True,
                    "pip install nerfstudio (this venv) or set external.* paths in YAML",
                )
            )
    return out


def _optional_binaries() -> list[ItemResult]:
    """Used by gdown, git-based deps, generic HTTP downloads."""
    optional: list[tuple[str, str, tuple[str, ...]]] = [
        ("git", "git", ("git",)),
        ("curl", "curl", ("curl",)),
        ("wget", "wget", ("wget",)),
    ]
    out: list[ItemResult] = []
    for label, exe, apt in optional:
        path = shutil.which(exe)
        if path:
            out.append(ItemResult(label, True, path, (), False))
        else:
            out.append(ItemResult(label, False, "MISSING (optional)", apt, False))
    return out


def _nvidia_smi_item() -> ItemResult:
    path = shutil.which("nvidia-smi")
    if path:
        return ItemResult("nvidia-smi", True, path, (), False)
    return ItemResult(
        "nvidia-smi",
        False,
        "MISSING (optional — GPU training)",
        (),
        False,
    )


def _external_config_item(
    cfg: SceneConfig,
    key: str,
    *,
    label: str,
    required: bool = True,
) -> ItemResult:
    value = str((cfg.external or {}).get(key, "")).strip()
    if not value:
        return ItemResult(label, False, f"external.{key} is not configured", (), required)
    first = value.split()[0]
    path = Path(first).expanduser()
    resolved = str(path.resolve()) if path.is_file() else shutil.which(first)
    if resolved:
        return ItemResult(label, True, resolved, (), required)
    return ItemResult(label, False, f"not found: {value}", (), required)


def _hybrid_usd_tools(cfg: SceneConfig) -> list[ItemResult]:
    grut_root = Path(str((cfg.external or {}).get("grut_root", ""))).expanduser()
    grut = ItemResult(
        "3DGRUT v1.1+",
        (grut_root / "train.py").is_file(),
        str(grut_root.resolve()) if (grut_root / "train.py").is_file() else "external.grut_root missing",
        (),
        True,
    )
    items = [
        grut,
        _external_config_item(cfg, "sam2_runner", label="SAM2 runner"),
        _external_config_item(
            cfg,
            "object_reconstruction_runner",
            label="object reconstruction runner",
        ),
        _external_config_item(cfg, "isaac_python", label="Isaac Sim python.sh"),
    ]
    if cfg.capture.modality in {"rgbd", "lidar"}:
        items.append(_external_config_item(cfg, "nvblox", label="nvblox mesh runner"))
    else:
        for key, label in (
            ("openmvs_interface", "OpenMVS InterfaceCOLMAP"),
            ("openmvs_densify", "OpenMVS DensifyPointCloud"),
            ("openmvs_reconstruct", "OpenMVS ReconstructMesh"),
            ("openmvs_refine", "OpenMVS RefineMesh"),
        ):
            items.append(_external_config_item(cfg, key, label=label))
    return items


GSPLAT_SPLATFACTO_LABEL = "gsplat CUDA (Splatfacto)"


def _find_nvcc() -> str | None:
    """
    Return an ``nvcc`` path if one exists.

    Many installs put the toolkit under ``/usr/local/cuda/bin`` without adding it to the
    interactive shell ``PATH``; ``run_cmd`` in ``nerfstudio.py`` prepends the same dirs
    for subprocesses, so doctor should not report a false negative when ``nvcc`` exists
    there.
    """
    w = shutil.which("nvcc")
    if w:
        return w
    bin_dirs: list[Path] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(key)
        if v:
            bin_dirs.append(Path(v) / "bin")
    # Keep in sync with ``_augment_subprocess_path`` in ``reconstruction/nerfstudio.py``.
    bin_dirs.extend(
        Path(p)
        for p in (
            "/usr/local/cuda/bin",
            "/usr/local/cuda-13.0/bin",
            "/usr/local/cuda-13/bin",
        )
    )
    seen: set[str] = set()
    for d in bin_dirs:
        try:
            key = str(d.resolve())
        except OSError:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        cand = d / "nvcc"
        if cand.is_file():
            return str(cand)
    return None


def _check_torch_cuda_splatfacto() -> ItemResult:
    """Splatfacto trains on GPU; CPU-only torch will fail later."""
    label = "torch.cuda (Splatfacto)"
    try:
        import torch  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return ItemResult(
            label,
            False,
            f"import failed: {e}",
            (),
            True,
            "pip install torch (see https://pytorch.org/get-started/locally/)",
        )
    if not torch.cuda.is_available():
        return ItemResult(
            label,
            False,
            "torch.cuda.is_available() is False (CPU-only build or no driver)",
            (),
            True,
            "Install a CUDA PyTorch wheel matching your NVIDIA driver from https://pytorch.org/get-started/locally/",
        )
    try:
        dev = torch.cuda.get_device_name(0)
    except Exception:
        dev = "CUDA device"
    return ItemResult(label, True, dev, (), True, None)


def _check_gsplat_splatfacto() -> ItemResult:
    """
    gsplat either ships a precompiled ``csrc`` extension or JIT-builds via ``nvcc``.

    If neither is available, ``_C`` is None and ``ns-train splatfacto`` crashes in rasterization.
    """
    if importlib.util.find_spec("gsplat") is None:
        return ItemResult(
            GSPLAT_SPLATFACTO_LABEL,
            False,
            "gsplat package not found",
            (),
            True,
            "pip install gsplat (Nerfstudio pins a version; reinstall nerfstudio if needed)",
        )
    try:
        from gsplat import csrc  # noqa: PLC0415, F401

        return ItemResult(
            GSPLAT_SPLATFACTO_LABEL,
            True,
            "prebuilt CUDA extension (gsplat.csrc)",
            (),
            True,
            None,
        )
    except ImportError:
        pass

    nvcc = _find_nvcc()
    if nvcc:
        on_path = shutil.which("nvcc") is not None
        hint = (
            ""
            if on_path
            else " (not on PATH; scan2usd prepends CUDA bin for ns-train / ns-render)"
        )
        return ItemResult(
            GSPLAT_SPLATFACTO_LABEL,
            True,
            f"no prebuilt csrc; nvcc for JIT — {nvcc}{hint}",
            (),
            True,
            None,
        )

    return ItemResult(
        GSPLAT_SPLATFACTO_LABEL,
        False,
        "no prebuilt gsplat.csrc and no nvcc found (PATH, CUDA_HOME, /usr/local/cuda*/bin)",
        ("nvidia-cuda-toolkit",),
        True,
        "Install NVIDIA CUDA Toolkit so nvcc is on PATH (version should match your PyTorch CUDA, e.g. cu12). "
        "https://developer.nvidia.com/cuda-downloads — on Ubuntu you can try: sudo apt install nvidia-cuda-toolkit",
    )


def _check_nerfstudio_import() -> ItemResult:
    spec = importlib.util.find_spec("nerfstudio")
    if spec is None:
        return ItemResult(
            "nerfstudio (import)",
            False,
            "MISSING",
            (),
            True,
            "pip install nerfstudio",
        )
    try:
        import nerfstudio as ns  # noqa: PLC0415

        ver = getattr(ns, "__version__", None)
        if not ver:
            try:
                from importlib.metadata import version  # noqa: PLC0415

                ver = version("nerfstudio")
            except Exception:
                ver = "import OK"
        return ItemResult("nerfstudio (import)", True, str(ver), (), True, None)
    except Exception as e:  # noqa: BLE001
        return ItemResult(
            "nerfstudio (import)",
            False,
            f"import failed: {e}",
            (),
            True,
            "pip install nerfstudio",
        )


def _numpy_note(echo: Callable[[str], None]) -> None:
    try:
        import numpy as np  # noqa: PLC0415

        if int(np.__version__.split(".", 1)[0]) < 2:
            echo("")
            echo(
                "Note: NumPy < 2. Ultralytics' opencv-python expects NumPy 2.x after some Nerfstudio installs. "
                "Run: pip install \"numpy>=2,<3\""
            )
    except Exception:
        pass


def _reconstruct_ready(
    colmap: ItemResult,
    ns_items: list[ItemResult],
    ffmpeg: ItemResult,
    ffprobe: ItemResult,
) -> bool:
    if not colmap.ok:
        return False
    if not ffmpeg.ok or not ffprobe.ok:
        return False
    return all(x.ok for x in ns_items)


@dataclass
class DoctorReport:
    """Structured doctor results for CLI and GUI consumers."""

    groups: dict[str, list[ItemResult]]
    apt_packages: tuple[str, ...]
    apt_install_line: str | None
    reconstruct_ready: bool


def collect_doctor_results(cfg: SceneConfig) -> DoctorReport:
    """Run all doctor checks and return structured results (no I/O)."""
    colmap = _colmap_item(cfg)
    ns_items = _nerfstudio_cli_items(cfg)
    ffmpeg = _check_binary("ffmpeg", apt=("ffmpeg",), label="ffmpeg")
    ffprobe = _check_binary("ffprobe", apt=("ffmpeg",), label="ffprobe")
    optional_bins = _optional_binaries()
    nvidia = _nvidia_smi_item()
    hybrid_tools = _hybrid_usd_tools(cfg)
    py_items: list[ItemResult] = [
        _check_import("numpy", label="numpy", pip_hint='pip install "numpy>=1.24"'),
        _check_opencv(),
        _check_import(
            "torch",
            label="torch",
            pip_hint="pip install torch (see https://pytorch.org/get-started/locally/)",
        ),
        _check_torch_cuda_splatfacto(),
        _check_gsplat_splatfacto(),
        _check_import("ultralytics", label="ultralytics", pip_hint="pip install ultralytics"),
        _check_nerfstudio_import(),
        _check_import(
            "trimesh",
            label="trimesh (USD geometry)",
            pip_hint='pip install -e ".[geometry]"',
        ),
        _check_import(
            "scipy",
            label="scipy (registration QA)",
            pip_hint='pip install -e ".[geometry]"',
        ),
        _check_import(
            "gradio",
            label="gradio (review UI)",
            pip_hint='pip install -e ".[review]"',
        ),
    ]

    groups: dict[str, list[ItemResult]] = {
        "external": [colmap, *ns_items, ffmpeg, ffprobe],
        "hybrid": list(hybrid_tools),
        "optional": [*optional_bins, nvidia],
        "python": list(py_items),
    }

    apt_pkgs: list[str] = []
    for items in groups.values():
        for it in items:
            if not it.ok and it.apt_if_missing:
                apt_pkgs.extend(it.apt_if_missing)
    apt_unique = tuple(sorted(set(apt_pkgs)))
    apt_line = _apt_install_line(apt_unique) if _linux() and apt_unique else None

    return DoctorReport(
        groups=groups,
        apt_packages=apt_unique,
        apt_install_line=apt_line,
        reconstruct_ready=_reconstruct_ready(colmap, ns_items, ffmpeg, ffprobe)
        and all(x.ok for x in py_items if x.required),
    )


def print_doctor_report(cfg: SceneConfig) -> None:
    """Print full doctor output to stdout."""
    echo = typer.echo
    report = collect_doctor_results(cfg)

    colmap = report.groups["external"][0]
    # external: colmap, ns..., ffmpeg, ffprobe — ns items are between colmap and last two
    external = report.groups["external"]
    ns_items = external[1:-2]
    ffmpeg = external[-2]
    ffprobe = external[-1]
    optional_bins = report.groups["optional"][:-1]
    nvidia = report.groups["optional"][-1]
    hybrid_tools = report.groups["hybrid"]
    py_items = report.groups["python"]

    echo("External tools:")
    _print_item(echo, colmap)
    if all(not x.ok for x in ns_items):
        echo(
            "  nerfstudio (ns-process-data, ns-train, ns-render): MISSING — "
            "not on PATH and ``import nerfstudio`` failed",
        )
    else:
        for it in ns_items:
            _print_item(echo, it)
    _print_item(echo, ffmpeg)
    _print_item(echo, ffprobe)

    echo("")
    echo("Hybrid USD / Isaac production tools:")
    for it in hybrid_tools:
        _print_item(echo, it)

    echo("")
    echo("Optional / tooling (recommended):")
    for it in optional_bins:
        _print_item(echo, it)
    _print_item(echo, nvidia)

    echo("")
    echo("Python environment:")
    for it in py_items:
        _print_item(echo, it)

    echo("")
    echo("Next steps:")
    _emit_next_steps(echo, colmap, ns_items, ffmpeg, ffprobe, optional_bins, py_items)

    if report.apt_install_line:
        echo("")
        echo("Suggested install (Debian / Ubuntu) for missing system packages:")
        echo(f"  {report.apt_install_line}")
    elif report.apt_packages and not _linux():
        echo("")
        echo("Some fixes need OS packages (apt hints omitted: not Linux).")

    _numpy_note(echo)


def _print_item(echo: Callable[[str], None], it: ItemResult) -> None:
    status = "OK" if it.ok else "MISSING"
    suffix = "" if it.required else " (optional)"
    if it.ok:
        echo(f"  {it.label}{suffix}: {status} — {it.detail}")
    else:
        echo(f"  {it.label}{suffix}: {status} — {it.detail}")


def _emit_next_steps(
    echo: Callable[[str], None],
    colmap: ItemResult,
    ns_items: list[ItemResult],
    ffmpeg: ItemResult,
    ffprobe: ItemResult,
    optional_bins: list[ItemResult],
    py_items: list[ItemResult],
) -> None:
    if not colmap.ok:
        echo(
            "  • COLMAP: ``sudo apt install colmap`` on Ubuntu/Debian, or "
            "https://colmap.github.io/ — or set ``external.colmap`` in YAML.",
        )
    if any(not x.ok for x in ns_items):
        echo(
            f"  • Nerfstudio: ``pip install nerfstudio`` using this interpreter ({sys.executable}), "
            "or set ``external.ns_process_data``, ``external.ns_train``, ``external.ns_render``. "
            "Console scripts missing → ``python -m nerfstudio.scripts.*`` is used when importable.",
        )
    if not ffmpeg.ok or not ffprobe.ok:
        echo(
            "  • ffmpeg (includes ffprobe): ``sudo apt install ffmpeg`` — required by ``ns-process-data``.",
        )

    for it in optional_bins:
        if not it.ok and it.apt_if_missing:
            echo(f"  • {it.label}: ``sudo apt install {it.apt_if_missing[0]}`` (optional but useful).")

    for it in py_items:
        if not it.ok and it.pip_hint:
            echo(f"  • {it.label}: {it.pip_hint}")

    gsplat_item = next((it for it in py_items if it.label == GSPLAT_SPLATFACTO_LABEL), None)
    if gsplat_item is not None and not gsplat_item.ok:
        echo(
            "  • Splatfacto: the ``gsplat`` wheel often has no prebuilt ``csrc``; without ``nvcc``, "
            "training dies in ``rasterization`` (``_C`` is None). Install a CUDA toolkit with ``nvcc`` "
            "(``sudo apt install nvidia-cuda-toolkit`` on Ubuntu is a starting point; match CUDA major to "
            "your PyTorch build, e.g. cu12, or use NVIDIA’s installer).",
        )

    if not shutil.which("nvidia-smi"):
        echo(
            "  • GPU: ``nvidia-smi`` not found — install NVIDIA drivers / CUDA for GPU training "
            "(not a single apt meta-package; use Ubuntu \"Additional Drivers\" or NVIDIA’s CUDA guide).",
        )

    if _reconstruct_ready(colmap, ns_items, ffmpeg, ffprobe) and all(
        x.ok for x in py_items if x.required
    ):
        echo("  • Core toolchain looks ready; you can try ``scan2usd reconstruct``.")

