"""Schema completeness tests."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, get_args

from scan2usd.config import (
    CaptureConfig,
    GeometryConfig,
    MaterialConfig,
    PhysicsConfig,
    QAConfig,
    ReconstructionConfig,
    SegmentationConfig,
    UsdConfig,
)
from scan2usd_gui.schema import (
    COMMAND_DEFS,
    CONFIG_PARAMS,
    EXPECTED_CLI_COMMANDS,
    TOOL_DEFS,
    get_schema,
)


def _typer_command_id(cmd: Any) -> str:
    if cmd.name:
        return str(cmd.name)
    assert cmd.callback is not None
    return str(cmd.callback.__name__).replace("_", "-")


def test_all_cli_commands_in_schema():
    ids = {c["id"] for c in COMMAND_DEFS}
    missing = EXPECTED_CLI_COMMANDS - ids
    extra = ids - EXPECTED_CLI_COMMANDS
    assert not missing, f"Missing commands: {sorted(missing)}"
    assert not extra, f"Unexpected commands: {sorted(extra)}"


def test_typer_commands_subset_of_schema():
    from scan2usd.cli import app

    typer_names = {_typer_command_id(cmd) for cmd in app.registered_commands}
    missing = typer_names - {c["id"] for c in COMMAND_DEFS}
    assert not missing, f"Typer commands missing from COMMAND_DEFS: {sorted(missing)}"


def test_apply_metric_scale_present():
    cmd = next(c for c in COMMAND_DEFS if c["id"] == "apply-metric-scale")
    opt_names = {o["config_path"] for o in cmd["options"]}
    assert "reviewer" in opt_names
    assert "meters_per_unit" in opt_names
    assert "known_length_m" in opt_names
    assert "source_length" in opt_names
    for o in cmd["options"]:
        if o["type"] == "path":
            assert o.get("widget") == "path", o["id"]


def test_config_params_have_tooltips():
    for p in CONFIG_PARAMS:
        assert p["tooltip"], f"{p['id']} missing tooltip"
        assert p["config_path"], f"{p['id']} missing config_path"
        assert p["label"], f"{p['id']} missing label"
        assert p.get("widget"), f"{p['id']} missing widget"
        assert p.get("help_level") in {"essential", "advanced"}


def test_path_params_have_path_widget():
    for p in CONFIG_PARAMS:
        if p["type"] == "path":
            assert p["widget"] == "path", p["id"]
            assert p.get("path_kind") in {"file", "dir", "any"}


def test_essential_sliders_have_bounds():
    for p in CONFIG_PARAMS:
        if p.get("help_level") == "essential" and p.get("widget") == "slider":
            assert p["min"] is not None and p["max"] is not None, p["id"]


def test_command_options_have_tooltips():
    for cmd in COMMAND_DEFS + TOOL_DEFS:
        for opt in cmd.get("options", []):
            assert opt["tooltip"], f"{cmd['id']}/{opt['id']} missing tooltip"
            assert opt.get("widget"), f"{cmd['id']}/{opt['id']} missing widget"


def test_get_schema_shape():
    s = get_schema()
    assert "config_params" in s
    assert "commands" in s
    assert "pipeline_stages" in s
    assert "tools" in s
    assert len([c for c in s["commands"] if c.get("job_kind") != "tool"]) == len(
        EXPECTED_CLI_COMMANDS
    )
    assert any(p.get("help_level") == "essential" for p in s["config_params"])
    assert any(c["id"] == "apply-metric-scale" for c in s["pipeline_stages"])
    assert any(t["id"].startswith("tool-") for t in s["tools"])


def test_grut_overrides_and_ns_viewer_in_config():
    paths = {p["config_path"] for p in CONFIG_PARAMS}
    assert "reconstruction.grut_overrides" in paths
    assert "external.ns_viewer" in paths


def _dataclass_leaf_paths(cls: type, prefix: str) -> set[str]:
    """Collect dotted paths for leaf fields (expand nested dataclasses)."""
    from typing import get_type_hints

    hints = get_type_hints(cls)
    paths: set[str] = set()
    for f in dataclasses.fields(cls):
        path = f"{prefix}.{f.name}"
        ann = hints.get(f.name, f.type)
        nested: type | None = None
        if isinstance(ann, type) and dataclasses.is_dataclass(ann):
            nested = ann
        else:
            for arg in get_args(ann):
                if isinstance(arg, type) and dataclasses.is_dataclass(arg):
                    nested = arg
                    break
        if nested is not None:
            paths |= _dataclass_leaf_paths(nested, path)
        else:
            paths.add(path)
    return paths


def test_reconstruction_and_nested_fields_have_config_paths():
    schema_paths = {p["config_path"] for p in CONFIG_PARAMS}
    required: set[str] = set()
    for cls, prefix in [
        (ReconstructionConfig, "reconstruction"),
        (SegmentationConfig, "segmentation"),
        (GeometryConfig, "geometry"),
        (MaterialConfig, "materials"),
        (PhysicsConfig, "physics"),
        (UsdConfig, "usd"),
        (QAConfig, "qa"),
        (CaptureConfig, "capture"),
    ]:
        required |= _dataclass_leaf_paths(cls, prefix)
    missing = required - schema_paths
    assert not missing, f"SceneConfig fields missing from CONFIG_PARAMS: {sorted(missing)}"


def _cli_option_key(pname: str, default: Any) -> str:
    """Map Typer param to schema option key (flag stem, e.g. show_boxes → boxes)."""
    from typer.models import OptionInfo

    if isinstance(default, OptionInfo) and default.param_decls:
        decl = default.param_decls[0]
        # '--boxes/--no-boxes' or '--load-config'
        primary = decl.split("/")[0].lstrip("-")
        return primary.replace("-", "_")
    return pname


def test_cli_options_covered_by_schema():
    """Every Typer option/argument (except config) appears in that command's schema."""
    from typer.models import ArgumentInfo, OptionInfo

    from scan2usd.cli import app

    # Rare intentional omissions (config is always injected by the job runner)
    omit: dict[str, set[str]] = {}

    by_id = {_typer_command_id(c): c for c in app.registered_commands}

    for cmd_def in COMMAND_DEFS:
        cmd_id = cmd_def["id"]
        typer_cmd = by_id.get(cmd_id)
        assert typer_cmd is not None, cmd_id
        callback = typer_cmd.callback
        assert callback is not None
        sig = inspect.signature(callback)
        schema_keys: set[str] = set()
        for opt in cmd_def.get("options", []):
            key = opt["config_path"] or ""
            if key.startswith("_pos_"):
                schema_keys.add(key[len("_pos_") :])
            else:
                schema_keys.add(key)
        for pname, param in sig.parameters.items():
            if pname in {"config", "self", "ctx"}:
                continue
            if pname in omit.get(cmd_id, set()):
                continue
            default = param.default
            if default is inspect.Parameter.empty or isinstance(
                default, (OptionInfo, ArgumentInfo)
            ):
                key = _cli_option_key(pname, default)
                assert key in schema_keys, (
                    f"{cmd_id}: CLI param {pname!r} (key {key!r}) missing from schema"
                )


def test_path_typed_command_options_use_path_widget():
    for cmd in COMMAND_DEFS + TOOL_DEFS:
        for opt in cmd.get("options", []):
            if opt["type"] == "path":
                assert opt.get("widget") == "path", f"{cmd['id']}/{opt['id']}"


def test_tools_scripts_are_under_tools():
    for t in TOOL_DEFS:
        assert t["script"].startswith("tools/"), t["id"]
        assert t.get("job_kind") == "tool"
        assert ".." not in t["script"]
