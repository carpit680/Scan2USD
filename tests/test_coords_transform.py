import json
from pathlib import Path

import numpy as np

from scan2usd.synthetic.transforms_io import (
    applied_transform_to_4x4,
    transform_aabb_colmap_to_nerfstudio,
    transform_aabb_to_dataparser,
    transform_points_colmap_to_nerfstudio,
)


def test_applied_transform_matches_ns_process_default():
    meta = {"applied_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]}
    t = applied_transform_to_4x4(meta)
    p = np.array([1.0, 2.0, 3.0])
    out = transform_points_colmap_to_nerfstudio(p[None], t)[0]
    np.testing.assert_allclose(out, [1.0, 3.0, -2.0])


def test_transform_aabb_corners():
    meta = {"applied_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]}
    t = applied_transform_to_4x4(meta)
    bbox = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    out = transform_aabb_colmap_to_nerfstudio(bbox, t)
    np.testing.assert_allclose(out[0], [0.0, 0.0, -2.0])
    np.testing.assert_allclose(out[1], [1.0, 3.0, 0.0])


def test_orient_transform_strips_applied(tmp_path: Path) -> None:
    meta = {"applied_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]]}
    tjson = tmp_path / "transforms.json"
    tjson.write_text(__import__("json").dumps(meta))
    from scan2usd.synthetic.transforms_io import (
        applied_transform_to_4x4,
        orient_transform_for_saved_coords,
    )

    t_app = applied_transform_to_4x4(meta)
    t_full = np.eye(4)
    t_full[:3, :3] = np.diag([2.0, 2.0, 2.0])
    t_orient = orient_transform_for_saved_coords(t_full[:3, :4], tmp_path)
    np.testing.assert_allclose(t_orient, t_full @ np.linalg.inv(t_app), rtol=1e-5)


def test_dataparser_transform_scale():
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = np.diag([2.0, 2.0, 2.0])
    bbox = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = transform_aabb_to_dataparser(bbox, t[:3, :4], 0.5)
    np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out[1], [1.0, 1.0, 1.0])


def test_workspace_transforms_if_present():
    tjson = Path("workspace/ns_data/transforms.json")
    if not tjson.is_file():
        return
    meta = json.loads(tjson.read_text())
    t = applied_transform_to_4x4(meta)
    assert t is not None
