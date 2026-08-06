"""
Stage-boundary quality gates: catch a bad artifact where it is made.

Every ``ready()`` predicate in the orchestrator asks whether a file exists. None
asks whether what is in it is any good, so a bad artifact travels the whole graph
and is only questioned at ``validate-usd`` — where, in preview mode, every check
is non-required and the report says ``usable: true`` anyway. That is how a
20-vertex collision hull shipped as a room, and how a training run that produced
8,576 Gaussians instead of 2.9 million completed without complaint.

Each gate is a pure function over an artifact's own report, returning a status
and — the part that matters — a recommendation naming the knob to turn. A gate
that only says "bad" makes someone re-derive the diagnosis that was already done
here.

One rule is load-bearing and easy to get wrong: **a metric computed on a
degenerate model is not a good score, it is a meaningless one**. The collapsed
run above reported 99.9% transmittance and 2 haze Gaussians — a perfect clarity
result for a scene with nothing in it. Clarity is therefore gated behind a
population check, and reports ``unknown`` rather than ``pass`` when there is no
scene to be clear about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GateResult:
    name: str
    status: str  # "pass" | "warn" | "fail" | "unknown"
    summary: str
    recommendation: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"pass", "unknown"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


def check_training_health(
    cleanup_report: dict[str, Any] | None,
    *,
    max_dead_opacity_fraction: float = 0.4,
    min_gaussians: int = 50_000,
) -> GateResult:
    """
    Did training produce a usable model?

    Two failure modes, both seen on this project and both silent at the time.

    A run can end with most of its Gaussians at zero opacity — 65.2% on the
    bedroom's first 50k — when opacity resets outlive the pruning that would
    clear what they killed. Those primitives cost VRAM and iteration time and
    are the faint-sheet population that reads as haze.

    A run can also collapse outright. With ``lambda_opacity`` at 0.01 the
    bedroom never grew past ~8,000 Gaussians against 2.9 million, because the
    penalty drove opacities under the density-prune threshold as fast as
    densification created them — and the loss stayed healthy throughout, so
    nothing in the training output looked wrong.
    """
    if not cleanup_report:
        return GateResult("training_health", "unknown", "No cleanup report yet.")

    total = int(cleanup_report.get("input_count") or 0)
    kept = int(cleanup_report.get("kept_count") or 0)
    dead = int(cleanup_report.get("removed_opacity") or 0)
    if total <= 0:
        return GateResult("training_health", "unknown", "Cleanup report has no counts.")

    dead_fraction = dead / total
    metrics = {
        "trained_gaussians": total,
        "kept_gaussians": kept,
        "dead_opacity_fraction": round(dead_fraction, 4),
    }

    if kept < min_gaussians:
        return GateResult(
            "training_health",
            "fail",
            f"Only {kept:,} Gaussians survived cleanup, from {total:,} trained — "
            "the model is degenerate, not merely dirty.",
            "Training collapsed. If reconstruction.anti_fog is on, lower "
            "lambda_opacity (0.01 destroyed this scene; stay near 0.0005) or "
            "disable it. Every metric downstream of this is meaningless.",
            metrics,
        )
    if dead_fraction > max_dead_opacity_fraction:
        return GateResult(
            "training_health",
            "warn",
            f"{dead_fraction:.1%} of trained Gaussians were below the opacity "
            f"floor ({dead:,} of {total:,}).",
            "Opacity resets are outliving pruning. Set "
            "reconstruction.grut_schedule_autofix so densify, prune and reset "
            "end together, or align strategy.prune.end_iteration with "
            "strategy.densify.end_iteration by hand.",
            metrics,
        )
    return GateResult(
        "training_health",
        "pass",
        f"{kept:,} Gaussians kept, {dead_fraction:.1%} dead on arrival.",
        metrics=metrics,
    )


def check_fog(
    cleanup_report: dict[str, Any] | None,
    *,
    min_transmittance: float = 0.8,
    min_gaussians: int = 50_000,
) -> GateResult:
    """
    Can you see across the room?

    Held-out PSNR cannot answer this: haze between the camera and a wall renders
    roughly the pixels the wall would, so it is nearly free photometrically and
    ruinous to look at.

    Deliberately refuses to answer for a degenerate model. An empty scene is
    perfectly transparent, and reporting that as a pass is worse than reporting
    nothing — it launders a collapsed training run into a green tick.
    """
    if not cleanup_report:
        return GateResult("fog", "unknown", "No cleanup report yet.")
    fog = cleanup_report.get("fog_metrics") or {}
    if not fog:
        return GateResult(
            "fog",
            "unknown",
            "Cleanup ran without a COLMAP sparse model, so haze was not measured.",
            "Pass the COLMAP sparse/0 directory to cleanup to enable fog metrics.",
        )

    kept = int(cleanup_report.get("kept_count") or 0)
    transmittance = fog.get("transmittance_across_room")
    breakdown = cleanup_report.get("free_space_breakdown") or {}
    # hull_air removes exactly the population the transmittance metric counts,
    # so afterwards it reads 100% whatever the scene looks like. Say so rather
    # than presenting a tautology as evidence.
    tautological = int(breakdown.get("removed_hull_air") or 0) > 0
    metrics = {
        "transmittance_across_room": transmittance,
        "fog_inside_hull": fog.get("fog_inside_hull"),
        "kept_gaussians": kept,
    }
    if kept < min_gaussians:
        return GateResult(
            "fog",
            "unknown",
            f"Not measured: only {kept:,} Gaussians remain, so clarity is "
            "meaningless — an empty room is perfectly clear.",
            "Fix training health first; this number will lie until there is a "
            "scene to be clear about.",
            metrics,
        )
    if transmittance is None:
        return GateResult("fog", "unknown", "No transmittance in the report.", metrics=metrics)

    if transmittance < min_transmittance:
        return GateResult(
            "fog",
            "warn",
            f"Only {transmittance:.1%} of a view survives crossing the room.",
            "Enable reconstruction.splat_cleanup.free_space_votes (3 is "
            "measured-safe) and free_behind. If haze persists, hull_air clears "
            "the remainder — on the bedroom that cost 0.73 dB of PSNR with LPIPS "
            "unchanged. Do not use air_min_neighbors on a room scan: it removes "
            "textureless walls.",
            metrics,
        )
    if tautological:
        return GateResult(
            "fog",
            "pass",
            f"{transmittance:.1%} of a view survives crossing the room — but "
            "hull_air removed exactly the Gaussians this metric counts, so the "
            "number is circular.",
            "Judge this build by eye or by held-out LPIPS. On the bedroom, "
            "clearing the interior cost 0.73 dB of PSNR with LPIPS unchanged.",
            metrics,
        )
    return GateResult(
        "fog",
        "pass",
        f"{transmittance:.1%} of a view survives crossing the room.",
        metrics=metrics,
    )


def check_mesh_sanity(
    mesh_report: dict[str, Any] | None,
    *,
    observed_extents: list[float] | None = None,
    min_faces: int = 10_000,
    max_extent_ratio: float = 2.0,
) -> GateResult:
    """
    Is the collision mesh a room, or a box around one?

    OpenMVS collapses on white walls and falls back to the sparse points, which
    on the kitchen produced a 20-vertex, 36-face hull. It was packaged as
    collision geometry and reported ``usable: true``, and the 12.6-unit
    registration error it caused was recorded as a non-required check.
    """
    if not mesh_report:
        return GateResult("mesh_sanity", "unknown", "No static mesh report yet.")

    faces = int(mesh_report.get("faces") or 0)
    vertices = int(mesh_report.get("vertices") or 0)
    extents = mesh_report.get("extents_m") or mesh_report.get("extents")
    metrics = {"faces": faces, "vertices": vertices, "extents": extents}

    if faces < min_faces:
        return GateResult(
            "mesh_sanity",
            "fail",
            f"Collision mesh has {faces:,} faces from {vertices:,} vertices — "
            "that is a hull, not a room.",
            "The dense reconstruction collapsed. Set "
            "reconstruction.rgb_geometry_backend: splat to fit the surface to "
            "the Gaussians instead, which also removes the registration step "
            "between two independent reconstructions.",
            metrics,
        )

    if observed_extents and extents:
        ratios = [
            float(m) / float(o)
            for m, o in zip(extents, observed_extents, strict=False)
            if float(o) > 1e-9
        ]
        if ratios:
            worst = max(ratios)
            metrics["worst_extent_ratio"] = round(worst, 2)
            if worst > max_extent_ratio:
                return GateResult(
                    "mesh_sanity",
                    "warn",
                    f"Mesh spans {worst:.1f}x the observed volume on its worst axis.",
                    "The mesh is being fitted to Gaussians outside the room. "
                    "Tighten reconstruction.splat_cleanup.crop_margin so the "
                    "halo is gone before the surface is fitted.",
                    metrics,
                )
    return GateResult(
        "mesh_sanity", "pass", f"{faces:,} faces, {vertices:,} vertices.", metrics=metrics
    )


def check_floor(
    alignment: dict[str, Any] | None,
    *,
    max_points_below: float = 0.1,
) -> GateResult:
    """
    Is the floor plane the floor?

    Inlier ratio cannot tell: the good kitchen fit scored 6.1% and the unusable
    desk fit 3.9%. The fraction of points that end up *below* the plane
    separates them cleanly — 2.4% against 34.4% — because a plane fitted to the
    wrong surface buries the scene.
    """
    if not alignment:
        return GateResult("floor", "unknown", "No floor alignment record yet.")
    below = alignment.get("points_below_floor_fraction")
    metrics = {
        "points_below_floor_fraction": below,
        "inlier_ratio": alignment.get("inlier_ratio"),
    }
    if below is None:
        return GateResult("floor", "unknown", "Alignment has no below-floor fraction.", metrics=metrics)
    if float(below) > max_points_below:
        return GateResult(
            "floor",
            "fail",
            f"{float(below):.1%} of the scene sits below the fitted floor.",
            "The plane was fitted to the wrong surface — a table top or a wall. "
            "Re-run align-floor, and if it persists the capture likely lacks "
            "floor coverage.",
            metrics,
        )
    return GateResult(
        "floor", "pass", f"{float(below):.1%} of points below the floor plane.", metrics=metrics
    )


def summarize(results: list[GateResult]) -> dict[str, Any]:
    """Roll gate results into one payload for the manifest and the GUI."""
    return {
        "gates": [result.to_dict() for result in results],
        "failed": [r.name for r in results if r.status == "fail"],
        "warned": [r.name for r in results if r.status == "warn"],
        "ok": all(r.ok for r in results),
    }
