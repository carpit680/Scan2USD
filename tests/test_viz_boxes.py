import numpy as np

from scan2usd.viz.boxes import load_objects_3d


def test_load_objects_3d_with_obb_fields(tmp_path):
    np.savez(
        tmp_path / "objects_3d.npz",
        class_id=np.array([0], dtype=np.int32),
        bbox_min=np.array([[0.0, 0.0, 0.0]]),
        bbox_max=np.array([[2.0, 1.0, 3.0]]),
        obb_center=np.array([[1.0, 0.5, 1.5]]),
        obb_half=np.array([[1.0, 0.5, 1.5]]),
        obb_rotation=np.tile(np.eye(3), (1, 1, 1)),
        coord_frame=np.array("nerfstudio"),
    )
    cid, centers, rots, halves, bmin, bmax = load_objects_3d(tmp_path / "objects_3d.npz")
    assert len(cid) == 1
    np.testing.assert_allclose(centers[0], [1.0, 0.5, 1.5])
    np.testing.assert_allclose(halves[0], [1.0, 0.5, 1.5])
