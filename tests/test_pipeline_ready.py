"""Pipeline stage skip / ready checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scan2usd.pipeline.orchestrator import PipelineOrchestrator


def _orch(tmp_path: Path, *, masks_dir: Path | None = None) -> PipelineOrchestrator:
    cfg = MagicMock()
    cfg.workspace_dir = tmp_path
    cfg.segmentation.masks_dir = masks_dir or (tmp_path / "masks")
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.cfg = cfg
    orch.manifest = MagicMock()
    orch.build_root = tmp_path / "build"
    orch.state_path = tmp_path / "build" / "pipeline_state.json"
    orch.manifest_path = tmp_path / "scene_manifest.json"
    orch.state = {"schema_version": "1.0", "stages": {}}
    return orch


def test_instance_masks_ready_false_without_pngs(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    report = tmp_path / "build" / "segmentation" / "mask_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    art = MagicMock()
    art.path = str(report)
    orch.manifest.artifact.return_value = art
    (tmp_path / "masks").mkdir()
    assert orch._instance_masks_ready() is False


def test_instance_masks_ready_true_with_png(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    report = tmp_path / "build" / "segmentation" / "mask_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    art = MagicMock()
    art.path = str(report)
    orch.manifest.artifact.return_value = art
    mask = tmp_path / "masks" / "laptop_001" / "frame_00001.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"x")
    assert orch._instance_masks_ready() is True


def test_run_stage_reruns_when_ready_false(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    orch.state_path.parent.mkdir(parents=True)
    orch.manifest_path.write_text("{}")
    orch.manifest.save = MagicMock()
    orch.state["stages"]["mask_propagation"] = {"status": "completed"}
    calls = {"n": 0}

    def op():
        calls["n"] += 1
        return "ok"

    result = orch.run_stage("mask_propagation", op, force=False, ready=lambda: False)
    assert result == "ok"
    assert calls["n"] == 1
