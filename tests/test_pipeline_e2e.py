"""End-to-end CLI smoke: lift → synthesize → export → benchmark (no ``reconstruct``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scan2usd.cli import app
from tests.fixtures_e2e import build_e2e_workspace, write_minimal_scene_yaml


@pytest.mark.e2e
def test_pipeline_lift_synthesize_export_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_minimal_scene_yaml(tmp_path / "scene.yaml")
    build_e2e_workspace(tmp_path)
    cfg = tmp_path / "scene.yaml"
    runner = CliRunner()

    r = runner.invoke(app, ["lift", str(cfg)])
    assert r.exit_code == 0, r.output + (r.exception and str(r.exception) or "")
    assert (tmp_path / "e2e_workspace" / "objects_3d.npz").is_file()

    r = runner.invoke(app, ["synthesize", str(cfg), "--skip-render"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "e2e_workspace" / "camera_path.json").is_file()

    r = runner.invoke(app, ["export-dataset", str(cfg), "--mode", "synthetic"])
    assert r.exit_code == 0, r.output
    syn_yaml = tmp_path / "e2e_workspace" / "dataset_synthetic" / "data.yaml"
    assert syn_yaml.is_file()

    r = runner.invoke(app, ["benchmark", str(cfg), "--experiment", "B"])
    assert r.exit_code == 0, r.output
    rep = tmp_path / "e2e_workspace" / "reports" / "report_B.json"
    assert rep.is_file()


def test_label_key_colmap_matches_frame(tmp_path: Path) -> None:
    from scan2usd.dataset.split import frame_label_key, label_key_from_colmap_image_name

    d = tmp_path / "f" / "sub"
    d.mkdir(parents=True)
    p = d / "a.jpg"
    p.write_bytes(b"")
    k = frame_label_key(tmp_path / "f", p)
    assert k == "sub_a"
    assert label_key_from_colmap_image_name("sub/a.jpg") == "sub_a"
