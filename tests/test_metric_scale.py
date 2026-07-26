import numpy as np
import pytest

from scan2usd.geometry.frames import uniform_scale, validate_similarity
from scan2usd.geometry.metric_scale import (
    apply_uniform_metric_scale,
    meters_per_unit_from_lengths,
)


def test_meters_per_unit_from_lengths():
    assert meters_per_unit_from_lengths(known_length_m=0.9, source_length=3.0) == pytest.approx(0.3)


def test_apply_uniform_metric_scale_scales_translation_and_linear():
    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = [1.0, 2.0, 3.0]
    # 90° about Z then translate — unit scale
    base[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    out = apply_uniform_metric_scale(base, 0.5)
    assert uniform_scale(out) == pytest.approx(0.5)
    np.testing.assert_allclose(out[:3, 3], [0.5, 1.0, 1.5])
    # Rotation preserved up to scale
    r = out[:3, :3] / 0.5
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-8)
    assert np.linalg.det(r) == pytest.approx(1.0)
    validate_similarity(out)


def test_apply_uniform_metric_scale_rejects_non_positive():
    with pytest.raises(ValueError):
        apply_uniform_metric_scale(np.eye(4), 0.0)
