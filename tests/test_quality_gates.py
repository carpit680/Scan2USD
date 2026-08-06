"""
Gates are pinned against the artifacts that actually got through.

Every fixture here is a real report from this project, not an invented one. A
gate that would not have caught the thing it exists to catch is worse than no
gate, because it makes the next silent failure look reviewed.
"""

from __future__ import annotations

from scan2usd.eval.gates import (
    check_floor,
    check_fog,
    check_mesh_sanity,
    check_training_health,
    summarize,
)

# The bedroom's first 50k run: opacity resets outlived pruning.
BEDROOM_V1 = {
    "input_count": 2_904_033,
    "kept_count": 979_803,
    "removed_opacity": 1_892_372,
    "fog_metrics": {"transmittance_across_room": 0.095, "fog_inside_hull": 20_609},
}

# The same scene with lambda_opacity=0.01: the model never grew.
COLLAPSED = {
    "input_count": 8_576,
    "kept_count": 1_081,
    "removed_opacity": 6_688,
    "fog_metrics": {"transmittance_across_room": 0.9989, "fog_inside_hull": 2},
}

HEALTHY = {
    "input_count": 1_100_000,
    "kept_count": 950_000,
    "removed_opacity": 120_000,
    "fog_metrics": {"transmittance_across_room": 0.92, "fog_inside_hull": 900},
}


def test_dead_gaussian_pileup_is_caught_and_named():
    result = check_training_health(BEDROOM_V1)
    assert result.status == "warn"
    assert result.metrics["dead_opacity_fraction"] == 0.6516
    assert "grut_schedule_autofix" in result.recommendation


def test_collapsed_training_fails_outright():
    result = check_training_health(COLLAPSED)
    assert result.status == "fail"
    assert "lambda_opacity" in result.recommendation


def test_an_empty_scene_is_never_reported_as_clear():
    """
    The trap this gate exists for.

    The collapsed run measured 99.9% transmittance and 2 haze Gaussians, which
    is a perfect clarity score for a scene containing nothing. Passing that
    through as green would launder a destroyed model into a good result.
    """
    result = check_fog(COLLAPSED)
    assert result.status == "unknown"
    assert "meaningless" in result.summary
    # And the raw number, which looks excellent, must not read as a pass.
    assert result.metrics["transmittance_across_room"] > 0.99


def test_hazy_scene_warns_with_the_filters_that_help():
    result = check_fog(BEDROOM_V1)
    assert result.status == "warn"
    assert "free_space_votes" in result.recommendation
    # And warns off the rule that ate the walls.
    assert "air_min_neighbors" in result.recommendation


def test_healthy_scene_passes_both():
    assert check_training_health(HEALTHY).status == "pass"
    assert check_fog(HEALTHY).status == "pass"


def test_the_twenty_vertex_hull_fails():
    """The kitchen shipped this as collision geometry with usable: true."""
    result = check_mesh_sanity({"vertices": 20, "faces": 36})
    assert result.status == "fail"
    assert "hull, not a room" in result.summary
    assert "rgb_geometry_backend" in result.recommendation


def test_mesh_inflated_by_the_halo_warns():
    result = check_mesh_sanity(
        {"vertices": 280_000, "faces": 278_814, "extents": [16.3, 21.1, 9.4]},
        observed_extents=[7.9, 10.4, 5.5],
    )
    assert result.status == "warn"
    assert "crop_margin" in result.recommendation


def test_floor_gate_uses_below_plane_not_inlier_ratio():
    """
    Inlier ratio cannot separate these two; the below-plane fraction can.

    The kitchen fit was good at 6.1% inliers and the desk fit unusable at 3.9% —
    indistinguishable. Below-floor was 2.4% against 34.4%.
    """
    kitchen = check_floor({"inlier_ratio": 0.061, "points_below_floor_fraction": 0.024})
    desk = check_floor({"inlier_ratio": 0.039, "points_below_floor_fraction": 0.344})
    assert kitchen.status == "pass"
    assert desk.status == "fail"


def test_missing_reports_are_unknown_not_pass():
    for result in (
        check_training_health(None),
        check_fog(None),
        check_mesh_sanity(None),
        check_floor(None),
    ):
        assert result.status == "unknown"
    # "unknown" must not block a build, but must not claim success either.
    assert summarize([check_fog(None)])["ok"] is True
    assert summarize([check_fog(None)])["gates"][0]["status"] == "unknown"


def test_summary_separates_failures_from_warnings():
    payload = summarize(
        [check_training_health(COLLAPSED), check_fog(BEDROOM_V1), check_floor(None)]
    )
    assert payload["failed"] == ["training_health"]
    assert payload["warned"] == ["fog"]
    assert payload["ok"] is False


def test_fog_gate_admits_when_its_own_number_is_circular():
    """
    hull_air deletes exactly the population transmittance measures, so it reads
    100% afterwards no matter what the room looks like. A gate that presented
    that as evidence would be the same trap as scoring a collapsed model.
    """
    report = dict(HEALTHY)
    report["free_space_breakdown"] = {"removed_hull_air": 14_542}
    report["fog_metrics"] = {"transmittance_across_room": 1.0, "fog_inside_hull": 0}
    result = check_fog(report)
    assert result.status == "pass"
    assert "circular" in result.summary
    assert "LPIPS" in result.recommendation
