"""Hybrid Scan-to-USD status, pending work, and quality levers.

Overall goal
------------
Build a hybrid Isaac Sim scene from an RGB scan: Gaussian ParticleField for
appearance plus meshes for collision/objects, with the scene Z-up, floor at
Z=0, and (eventually) metric scale for physics.

Current deliverable: preview-mode ``workspace/usd/scene.usd`` for the kitchen
scan (``configs/example_scene.yaml`` / ``configs/high_quality_scene.yaml``).

Goal themes
-----------
- Run hybrid USD pipeline in Isaac
- Floor alignment (Z-up, floor at Z=0)
- Configurable stray-splat cleanup
- Longer 3DGRUT training for sharper env visuals
- Dedicated high-quality config for final builds

Done (as of latest workspace)
-----------------------------
- Isaac Sim 6 + viewer (``tools/isaac/view_scene.py``)
- Floor RANSAC + ``scan2usd align-floor``
- USD row-vector matrix authoring (``matrix4d_text`` writes ``M.T``)
- Packaging: splat gets COLMAP→USD ``T``; baked StaticCollision/Proxy stay identity
- ``splat_cleanup`` / ``cleanup-splat`` (``outlier_std`` primary knob)
- 3DGRUT quality path with ``grut_overrides``; 50k-iter train on this scene
- ``configs/high_quality_scene.yaml``
- Preview object ``bottle_001`` (masked out of env splat by design)

Pending
-------
- **True metric scale** — use ``scan2usd apply-metric-scale`` after measuring a
  known length (see below). Physics sizes stay wrong until this is approved.
- **Production dense static mesh** — OpenMVS/CUDA COLMAP dense path incomplete
- **Clean plate** — ``capture.clean_plate_dir`` still null (holes behind movables)
- **Object detail** — bottle is a preview hull; set ``object_capture_dirs``
- **More objects** — only one approved movable so far
- **HQ full rebuild** — last 50k used ``example_scene.yaml`` (``num_downscales: 2``);
  full HQ needs ``high_quality_scene.yaml`` + preferably re-process at
  ``num_downscales: 1``
- **Production ``build_mode``** — needs metric scale + stronger QA gates

Improving USD accuracy (geometry / registration / physics)
----------------------------------------------------------
1. Measure a known length in the room; run ``scan2usd apply-metric-scale``.
2. Keep floor alignment; re-run ``align-floor`` if COLMAP changes.
3. Rebuild static collision with a real dense mesh after metric ``T`` is set.
4. Clean plate + object detail captures for manipulables.
5. ``scan2usd validate-usd`` once metric.

Improving sharpness (ParticleField)
-----------------------------------
1. Rebuild with ``configs/high_quality_scene.yaml`` (SH ramp + full-res COLMAP).
2. Capture quality (less blur, more overlap); optional re-preprocess at
   ``num_downscales: 1``.
3. Tune ``splat_cleanup.outlier_std`` (higher keeps fine detail, more floaters).
4. Movables stay out of the env splat — improve object meshes, not env train length.

Metric scale quick start
------------------------
After ``align-floor``, pick an edge you can measure in meters (counter width,
door width). Estimate the same edge length in current scene/COLMAP units
(viewer measure or point distance), then::

    scan2usd apply-metric-scale configs/example_scene.yaml \\
      --known-length-m 0.91 --source-length 2.45 --reviewer YOUR_NAME

Or if you already know meters per COLMAP unit::

    scan2usd apply-metric-scale configs/example_scene.yaml \\
      --meters-per-unit 0.37 --reviewer YOUR_NAME

Then rebuild baked geometry and package::

    scan2usd build-object-usd configs/example_scene.yaml bottle_001
    scan2usd package-usd configs/example_scene.yaml
