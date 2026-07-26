import numpy as np

from scan2usd.reconstruction.colmap_io import bbox3d_from_points_in_frustum


def test_bbox3d_depth_trim_tightens_extent():
  """Background points far along the ray should not inflate the AABB."""
  n = 200
  uv = np.zeros((n, 2))
  depth = np.concatenate([np.full(20, 2.0), np.full(180, 50.0)])
  xyz = np.zeros((n, 3))
  xyz[:20] = np.random.randn(20, 3) * 0.2 + np.array([0.0, 0.0, 2.0])
  xyz[20:] = np.random.randn(180, 3) * 0.2 + np.array([0.0, 0.0, 50.0])
  bb = bbox3d_from_points_in_frustum(uv, depth, xyz, -10, -10, 10, 10, min_points=8)
  assert bb is not None
  assert bb[1, 2] - bb[0, 2] < 5.0
