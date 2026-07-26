"""Baked and relightable PBR material variants for rigid object meshes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest


PBR_DEFAULTS = {
    "generic": (0.6, 0.0),
    "cardboard": (0.8, 0.0),
    "plastic": (0.45, 0.0),
    "wood": (0.65, 0.0),
    "metal": (0.3, 0.9),
}


def find_baked_texture(mesh_path: Path) -> Path | None:
    if mesh_path.suffix.lower() != ".obj":
        return None
    text = mesh_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^mtllib\s+(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    mtl = mesh_path.parent / match.group(1).strip()
    if not mtl.is_file():
        return None
    mtl_text = mtl.read_text(encoding="utf-8", errors="ignore")
    texture_match = re.search(r"^map_Kd\s+(.+)$", mtl_text, flags=re.MULTILINE)
    if not texture_match:
        return None
    texture = mtl.parent / texture_match.group(1).strip()
    return texture if texture.is_file() else None


def _delight(image: np.ndarray) -> np.ndarray:
    """Conservative single-image de-lighting for an approximate PBR base color."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    linear = np.power(np.clip(rgb, 0.0, 1.0), 2.2)
    luminance = np.maximum(
        linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722,
        1e-4,
    )
    sigma = max(3.0, min(image.shape[:2]) / 32.0)
    illumination = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = linear / np.maximum(illumination[..., None], 0.05)
    scale = 0.5 / max(float(np.median(normalized)), 1e-3)
    normalized = np.clip(normalized * scale, 0.0, 1.0)
    return np.round(np.power(normalized, 1.0 / 2.2) * 255.0).astype(np.uint8)


def _constant_map(size: tuple[int, int], value: int | tuple[int, int, int]) -> np.ndarray:
    width, height = size
    if isinstance(value, tuple):
        array = np.zeros((height, width, 3), dtype=np.uint8)
        array[:] = value
        return array
    return np.full((height, width), value, dtype=np.uint8)


def generate_material_textures(
    baked_texture: Path,
    output_dir: Path,
    *,
    template: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baked_out = output_dir / "baked_basecolor.png"
    with Image.open(baked_texture) as raw:
        image = np.asarray(raw.convert("RGB"))
    Image.fromarray(image).save(baked_out)
    basecolor = output_dir / "pbr_basecolor.png"
    Image.fromarray(_delight(image)).save(basecolor)
    height, width = image.shape[:2]
    roughness, metallic = PBR_DEFAULTS.get(template, PBR_DEFAULTS["generic"])
    normal_path = output_dir / "pbr_normal.png"
    roughness_path = output_dir / "pbr_roughness.png"
    metallic_path = output_dir / "pbr_metallic.png"
    Image.fromarray(_constant_map((width, height), (128, 128, 255))).save(normal_path)
    Image.fromarray(_constant_map((width, height), round(roughness * 255))).save(
        roughness_path
    )
    Image.fromarray(_constant_map((width, height), round(metallic * 255))).save(
        metallic_path
    )
    return {
        "baked": baked_out,
        "basecolor": basecolor,
        "normal": normal_path,
        "roughness": roughness_path,
        "metallic": metallic_path,
    }


def _asset(path: Path, relative_to: Path) -> str:
    return Path(path).resolve().relative_to(relative_to.resolve()).as_posix()


def write_materials_usda(
    output_path: Path,
    textures: dict[str, Path],
) -> Path:
    """Write two portable UsdPreviewSurface materials for later Isaac composition."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative = output_path.parent
    baked = _asset(textures["baked"], relative)
    base = _asset(textures["basecolor"], relative)
    normal = _asset(textures["normal"], relative)
    roughness = _asset(textures["roughness"], relative)
    metallic = _asset(textures["metallic"], relative)
    text = f'''#usda 1.0
(
    defaultPrim = "Materials"
)

def Scope "Materials"
{{
    def Material "Baked"
    {{
        token outputs:surface.connect = </Materials/Baked/Surface.outputs:surface>
        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0, 0, 0)
            color3f inputs:emissiveColor.connect = </Materials/Baked/Texture.outputs:rgb>
            token outputs:surface
        }}
        def Shader "Texture"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{baked}@
            float2 inputs:st.connect = </Materials/Baked/Primvar.outputs:result>
            token inputs:sourceColorSpace = "sRGB"
            float3 outputs:rgb
        }}
        def Shader "Primvar"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
    }}

    def Material "PBR"
    {{
        token outputs:surface.connect = </Materials/PBR/Surface.outputs:surface>
        def Shader "Surface"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Materials/PBR/BaseColor.outputs:rgb>
            float inputs:roughness.connect = </Materials/PBR/Roughness.outputs:r>
            float inputs:metallic.connect = </Materials/PBR/Metallic.outputs:r>
            normal3f inputs:normal.connect = </Materials/PBR/Normal.outputs:rgb>
            token outputs:surface
        }}
        def Shader "BaseColor"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{base}@
            float2 inputs:st.connect = </Materials/PBR/Primvar.outputs:result>
            token inputs:sourceColorSpace = "sRGB"
            float3 outputs:rgb
        }}
        def Shader "Roughness"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{roughness}@
            float2 inputs:st.connect = </Materials/PBR/Primvar.outputs:result>
            token inputs:sourceColorSpace = "raw"
            float outputs:r
        }}
        def Shader "Metallic"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{metallic}@
            float2 inputs:st.connect = </Materials/PBR/Primvar.outputs:result>
            token inputs:sourceColorSpace = "raw"
            float outputs:r
        }}
        def Shader "Normal"
        {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{normal}@
            float2 inputs:st.connect = </Materials/PBR/Primvar.outputs:result>
            token inputs:sourceColorSpace = "raw"
            float3 outputs:rgb
        }}
        def Shader "Primvar"
        {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
    }}
}}
'''
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_object_materials(
    cfg: SceneConfig,
    manifest: SceneManifest,
    obj: ObjectRecord,
    *,
    baked_texture: Path | None = None,
) -> dict[str, Path]:
    if not obj.render_mesh:
        raise RuntimeError(f"{obj.instance_id} has no render mesh")
    mesh_path = Path(obj.render_mesh)
    baked_texture = baked_texture or find_baked_texture(mesh_path)
    if baked_texture is None:
        raise RuntimeError(
            f"No texture atlas found for {obj.instance_id}; object reconstruction must emit one"
        )
    output_dir = mesh_path.parent / "textures"
    template = str(obj.physics.get("template", "generic"))
    textures = generate_material_textures(baked_texture, output_dir, template=template)
    materials_path = write_materials_usda(mesh_path.parent / "materials.usda", textures)
    obj.baked_texture = str(textures["baked"].resolve())
    obj.pbr_textures = {
        key: str(path.resolve())
        for key, path in textures.items()
        if key != "baked"
    }
    obj.physics["materials_approved"] = False
    manifest.upsert_object(obj)
    manifest.register_artifact(
        artifact_id=f"object_{obj.instance_id}_materials",
        kind="usd_material_variants",
        path=materials_path,
        producer="scan2usd.materials",
        metadata={
            "variants": ["baked", "pbr"],
            "delighting": "single-image low-frequency illumination normalization",
            "warning": "PBR maps are estimates and require review",
        },
    )
    report = {
        "instance_id": obj.instance_id,
        "materials": str(materials_path.resolve()),
        "textures": {key: str(path.resolve()) for key, path in textures.items()},
    }
    (mesh_path.parent / "material_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"materials": materials_path, **textures}
