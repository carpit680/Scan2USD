"""Physical-property estimation for reconstructed rigid objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scan2usd.config import PhysicsConfig


PHYSICS_TEMPLATES: dict[str, dict[str, float]] = {
    "generic": {"density_kg_m3": 700.0, "friction": 0.5, "restitution": 0.05},
    "cardboard": {"density_kg_m3": 200.0, "friction": 0.55, "restitution": 0.02},
    "plastic": {"density_kg_m3": 950.0, "friction": 0.4, "restitution": 0.1},
    "wood": {"density_kg_m3": 650.0, "friction": 0.55, "restitution": 0.05},
    "metal": {"density_kg_m3": 7_800.0, "friction": 0.35, "restitution": 0.08},
}


@dataclass(frozen=True)
class PhysicalProperties:
    mass_kg: float
    density_kg_m3: float
    center_of_mass_m: list[float]
    diagonal_inertia_kg_m2: list[float]
    principal_axes: list[list[float]]
    friction: float
    restitution: float
    collider: str
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _template(name: str, cfg: PhysicsConfig) -> dict[str, float]:
    values = dict(PHYSICS_TEMPLATES.get(name, PHYSICS_TEMPLATES["generic"]))
    if name == "generic":
        values.update(
            {
                "density_kg_m3": cfg.default_density_kg_m3,
                "friction": cfg.default_friction,
                "restitution": cfg.default_restitution,
            }
        )
    return values


def estimate_physical_properties(
    mesh,
    cfg: PhysicsConfig,
    *,
    template: str = "generic",
) -> PhysicalProperties:
    values = _template(template, cfg)
    measurement_mesh = mesh if mesh.is_watertight else mesh.convex_hull
    confidence = 0.9 if mesh.is_watertight else 0.45
    source = "watertight_mesh" if mesh.is_watertight else "convex_hull_fallback"
    measurement_mesh = measurement_mesh.copy()
    measurement_mesh.density = values["density_kg_m3"]
    mass = max(float(measurement_mesh.mass), 1e-4)
    center = np.asarray(measurement_mesh.center_mass, dtype=np.float64)
    inertia = np.asarray(measurement_mesh.moment_inertia, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1.0
    return PhysicalProperties(
        mass_kg=mass,
        density_kg_m3=float(values["density_kg_m3"]),
        center_of_mass_m=[float(value) for value in center],
        diagonal_inertia_kg_m2=[float(value) for value in eigenvalues],
        principal_axes=eigenvectors.tolist(),
        friction=float(values["friction"]),
        restitution=float(values["restitution"]),
        collider=cfg.dynamic_collider,
        confidence=confidence,
        source=source,
    )
