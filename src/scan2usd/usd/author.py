"""Dependency-light USDA authoring; Isaac's bundled Python converts to USDC."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from scan2usd.config import SceneConfig
from scan2usd.geometry.mesh_ops import load_mesh
from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_external_command
from scan2usd.synthetic.transforms_io import rotmat_to_quat_wxyz


def usd_identifier(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not clean:
        clean = "Asset"
    if clean[0].isdigit():
        clean = f"_{clean}"
    return clean


def relative_asset(path: Path, layer_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), layer_path.parent.resolve())).as_posix()


def _float(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.9g}"


def _vectors(values: np.ndarray, *, width: int = 3) -> str:
    array = np.asarray(values)
    return ",\n            ".join(
        "(" + ", ".join(_float(item) for item in row[:width]) + ")" for row in array
    )


def _integers(values: np.ndarray) -> str:
    return ", ".join(str(int(value)) for value in np.asarray(values).reshape(-1))


def matrix4d_text(matrix: list[list[float]] | np.ndarray) -> str:
    """Serialize a Scan2USD column-vector 4x4 as a USD row-vector matrix4d.

    Internal math uses ``p' = M @ p`` (translation in the last column). OpenUSD
    ``xformOp:transform`` uses row vectors (``p' = p * M``, translation in the
    last row), so we write ``M.T``.
    """
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("USD transform must be 4x4")
    usd = value.T
    return "(\n            " + ",\n            ".join(
        "(" + ", ".join(_float(item) for item in row) + ")" for row in usd
    ) + "\n        )"


def write_mesh_usda(
    mesh_path: Path,
    output_path: Path,
    *,
    default_prim: str = "Geometry",
    mesh_prim: str = "Mesh",
    meters_per_unit: float = 1.0,
) -> Path:
    mesh = load_mesh(mesh_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _vectors(np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    counts = ", ".join("3" for _ in range(len(faces)))
    indices = _integers(faces)
    normals_block = ""
    if len(mesh.vertex_normals) == len(mesh.vertices):
        normals_block = (
            "\n        normal3f[] normals = [\n            "
            + _vectors(np.asarray(mesh.vertex_normals))
            + '\n        ] (\n            interpolation = "vertex"\n        )'
        )
    uv_block = ""
    uv = getattr(mesh.visual, "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        uv_block = (
            "\n        texCoord2f[] primvars:st = [\n            "
            + _vectors(np.asarray(uv), width=2)
            + '\n        ] (\n            interpolation = "vertex"\n        )'
        )
    output_path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "{usd_identifier(default_prim)}"
    metersPerUnit = {meters_per_unit:.9g}
    upAxis = "Z"
)

def Xform "{usd_identifier(default_prim)}"
{{
    def Mesh "{usd_identifier(mesh_prim)}"
    {{
        point3f[] points = [
            {points}
        ]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]{normals_block}{uv_block}
        uniform token subdivisionScheme = "none"
    }}
}}
''',
        encoding="utf-8",
    )
    return output_path


def write_object_physics_usda(
    path: Path,
    *,
    physics: dict,
    collision_approximation: str,
) -> Path:
    inertia = physics.get("diagonal_inertia_kg_m2", [1e-3, 1e-3, 1e-3])
    axes = np.asarray(physics.get("principal_axes", np.eye(3)), dtype=np.float64)
    quat = rotmat_to_quat_wxyz(axes)
    path.write_text(
        f'''#usda 1.0

over "Object" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    bool physics:rigidBodyEnabled = true
    float physics:mass = {_float(physics.get("mass_kg", 1.0))}
    point3f physics:centerOfMass = (0, 0, 0)
    float3 physics:diagonalInertia = ({", ".join(_float(v) for v in inertia)})
    quatf physics:principalAxes = ({_float(quat[0])}, ({_float(quat[1])}, {_float(quat[2])}, {_float(quat[3])}))

    over "Collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        bool physics:collisionEnabled = true
        uniform token physics:approximation = "{collision_approximation}"
        token visibility = "invisible"
        rel material:binding:physics = </Object/PhysicsMaterial>
    }}

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {_float(physics.get("friction", 0.5))}
        float physics:dynamicFriction = {_float(physics.get("friction", 0.5))}
        float physics:restitution = {_float(physics.get("restitution", 0.05))}
    }}
}}
''',
        encoding="utf-8",
    )
    return path


def write_object_asset_usda(
    path: Path,
    *,
    instance_id: str,
    class_name: str,
    geometry_layer: Path,
    collision_layer: Path,
    materials_layer: Path,
    physics_layer: Path,
    default_look: str,
) -> Path:
    geometry = relative_asset(geometry_layer, path)
    collision = relative_asset(collision_layer, path)
    materials = relative_asset(materials_layer, path)
    physics = relative_asset(physics_layer, path)
    path.write_text(
        f'''#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        @{materials}@,
        @{physics}@
    ]
)

def Xform "Object" (
    prepend variantSets = "look"
    variants = {{
        string look = "{default_look}"
    }}
)
{{
    custom string semantic:class = "{class_name}"
    custom string semantic:instanceId = "{instance_id}"

    def Xform "Visual" (
        prepend references = @{geometry}@</Geometry>
    )
    {{
    }}

    def Xform "Collision" (
        prepend references = @{collision}@</Geometry>
    )
    {{
        token visibility = "invisible"
    }}

    variantSet "look" = {{
        "baked" {{
            over "Visual" {{
                over "Mesh" {{
                    rel material:binding = </Materials/Baked>
                }}
            }}
        }}
        "pbr" {{
            over "Visual" {{
                over "Mesh" {{
                    rel material:binding = </Materials/PBR>
                }}
            }}
        }}
    }}
}}
''',
        encoding="utf-8",
    )
    return path


def convert_layer_with_isaac(
    cfg: SceneConfig,
    source: Path,
    target: Path,
    *,
    required: bool = True,
) -> Path:
    prefix = resolve_external_command(
        cfg,
        "isaac_python",
        default="python.sh",
        required=required,
    )
    if prefix is None:
        return source
    script = Path(__file__).resolve().parents[3] / "tools" / "isaac" / "convert_stage.py"
    adapter = ExternalToolAdapter("isaac_python", prefix)
    adapter.run(str(script), "--input", str(source.resolve()), "--output", str(target.resolve()))
    if not target.is_file():
        raise FileNotFoundError(f"Isaac conversion did not create {target}")
    return target
