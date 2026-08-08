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

Quality/tuning additions (2026-07-26)
-------------------------------------
- ``scan2usd render-heldout`` — headless Isaac renders at held-out COLMAP
  cameras → PSNR/SSIM/LPIPS vs real frames → ``usd/scene_quality.json``.
- ``scan2usd tune`` + GUI **Tuning** page — cheap splat-cleanup sweeps and
  opt-in 3DGRUT retrain trials optimizing ``quality_score``; winner promoted
  to ``<config>_tuned.yaml`` (a full snapshot, so it goes stale as its source
  moves on — the header records the source hash). ``configs/bedroom_scene.yaml``
  is the most measured profile; there is no golden one yet.
- Correctness: baked meshes now carry ``transform_hash`` (auto re-bake when
  floor/metric transform changes — the desk scan's 0.23 m registration error
  came from a stale bake); floor RANSAC rejects planes below
  ``geometry.min_floor_inlier_ratio`` (desk run had accepted 3.9% inliers);
  ``segmentation.allow_no_objects`` enables environment-only builds.
- Capture gate: ``reconstruct`` now reports the COLMAP registration rate and
  fails below ``reconstruction.min_registration_rate``; frame extraction reports
  how many frames it dropped as blurry (``reconstruction.blur_threshold``).

KITCHEN scan is the working scene (2026-07-26)
-----------------------------------------------
``configs/kitchen_scene.yaml`` (50k) / ``configs/kitchen_preview.yaml`` (7k fast
validation), workspace ``workspace_kitchen``, source ``~/Desktop/scan2.MOV``.

- COLMAP: **787/787 frames registered (100%)**, 115,288 points, clean intrinsics
  (fx 583.54 / fy 583.73, principal point centred). Frame extraction kept
  787/851 (only 7.5% dropped as blurry).
- Floor alignment passes: 1.6% of points below the plane, 100% of cameras above,
  aligned Z 5th percentile 0.00.
- Full hybrid chain runs end-to-end **environment-only** (no approved objects,
  via ``segmentation.allow_no_objects``): 3DGRUT → cleanup (262,616 → 236,203
  Gaussians) → OpenMVS static geometry → lighting → ``usd/kitchen.usd``.
- **Measured quality** (Isaac RTX renders vs real frames, all 79 held-out views):

  =========================  =============  =====  =====  =====  ==========
  build                      quality_score   PSNR   SSIM  LPIPS  gaussians
  =========================  =============  =====  =====  =====  ==========
  7k iters, outlier_std 4          68.90    19.95  0.777  0.316    236,203
  50k iters, outlier_std 4         66.74    18.03  0.756  0.323    414,776
  50k iters, outlier_std 8         74.12    23.30  0.816  0.264    459,816
  **50k iters, outlier_std 20**    **74.13**  23.35  0.816  0.265    462,103
  =========================  =============  =====  =====  =====  ==========

  Those are **appearance** scores. After ``validate-usd`` populates the
  registration error from the placeholder collision mesh, the winner's total
  becomes 64.13 = 74.13 appearance − 10.0 registration penalty. ``scene_quality
  .json`` now carries a ``score_breakdown`` so the two are never confused;
  ``appearance_score`` is the tuner-comparable number.

  Baselines in ``workspace_kitchen/baselines/``; trials in
  ``workspace_kitchen/tuning/trials.json``; winner promoted to
  ``configs/kitchen_scene_tuned.yaml``.

- **The cleanup default was destroying quality.** At ``outlier_std: 4.0`` the
  50k model scored *worse* than the 7k one. The spatial filter is one global
  median-MAD sphere, so a fixed sigma prunes far harder as a model densifies:
  it deleted 56% of the Gaussians and tore dark holes through floors and walls
  (large low-texture areas dominate PSNR/SSIM, so object detail could not
  compensate). Raising it to 8.0 recovered +7.4 points / +5.3 dB PSNR. The curve
  is flat from 8 to 20, so 8.0 is the knee and is now the default in
  ``SplatCleanupConfig``, ``SplatCleanupParams``, the GUI schema, and the configs.
- Isaac renders align object-for-object with ground-truth frames, validating the
  COLMAP→USD transform, the OpenCV→USD camera flip, intrinsics→aperture mapping,
  and USD's row-vector matrix convention.

- **Static collision geometry is still a placeholder.** The OpenMVS wrappers fall
  back to sparse points, so ``static_proxy.ply`` is a **20-vertex, 36-face hull**
  spanning the whole (outlier-inflated) point cloud. That is why
  ``splat_proxy_registration`` reports a ~12.6-unit median error. It does not
  affect the splat's appearance, but the scene is NOT physics-ready until the
  dense path works. Preview builds do not require this check, so
  ``build_report.json`` still says ``usable: true`` — that is preview semantics,
  not a production pass.

Floater investigation (2026-07-26) — measured, not guessed
----------------------------------------------------------
Reported symptom: from outside the scene, stray splats obstruct the view; some
objects have floating splats. Held-out PSNR/SSIM **cannot see this** — every
held-out camera sits inside the capture path, so exterior obstruction is
invisible to it. ``tools/isaac/render_orbit.py`` was added to look from outside.

Interior PSNR (baseline with no floater filters = 23.35 dB), exterior judged
from orbit renders:

  ===========================================  ==========  =====================
  cleanup configuration                        interior     exterior
  ===========================================  ==========  =====================
  none (opacity + outlier_std only)              23.35      halo + haze
  **crop 0.5 + max_scale_frac 0.08**           **22.62**    far halo gone
  + needle 8.0 + density 2                       17.49      noticeably cleaner
  aggressive (scale 0.02, crop 0.25, density 3)  15.46      cleanest
  ===========================================  ==========  =====================

Conclusions:

- Deleting the **far halo** (~12k Gaussians beyond the observed volume) and the
  ~1k scene-sized blobs costs **0.73 dB** — effectively free. These are now the
  only cleanup filters on by default.
- Deleting the **haze** (spikes, isolated Gaussians near scene edges) costs
  3-6 dB. Those Gaussians genuinely contribute to observed views, so this is a
  real trade, not a free win. ``min_neighbors``, ``max_needle_ratio`` and
  ``min_view_count`` therefore default to OFF, documented with their cost.
- Root cause of the remaining haze is representational, not a cleanup bug: a
  3DGS model is only valid **inside the volume it was observed from**. Rendering
  an inside-out room capture from outside asks for view directions that have no
  training data, where view-dependent SH produces garbage. Fix it at capture
  (wider coverage, 360 camera) or accept exterior views are out of domain.
- Frustum-visibility trimming (the cheap approximation of TrimGS) does **not**
  discriminate here: the median Gaussian is seen by 208 of 787 cameras, and only
  0.3% by two or fewer, because room-interior frusta radiate outward over the
  halo and no occlusion is resolved. 3DGRUT ships the real thing —
  ``threedgrut/export/scripts/filter_visibility.py`` accumulates per-Gaussian
  ``mog_visibility`` from actual renders — which is the correct next step.

Fog measurement and free-space carving (2026-08-05)
----------------------------------------------------
The conclusion above — "the remaining haze is representational" — held for
*exterior* views but was never tested against the interior, because there was no
interior haze metric. ``tools/geometry/analyze_splat.py`` supplies one, and
``reconstruction/free_space.py`` turns it into a filter.

Method: every (camera, SfM point) pair is a ray whose interior must be empty,
since the camera saw that point. Voxels along those rays are carved free. A
Gaussian in carved-free space with no surface within a fixed radius is an
artifact by construction rather than by threshold. Both conditions are required:
a featureless white wall has few SfM points, but nothing behind it to aim rays
at either, so it never reads as haze.

Two measurement bugs found and fixed before any number here was trusted:

- **Grid sized to the Gaussians** made every result incomparable between models.
  The bedroom's raw export spans 574 units because of a few strays (1.36-unit
  voxels, 57-voxel camera hull); the cleaned model spans 47 (0.15-unit voxels,
  48,832). Comparing them measured the grid, and made cleanup look like it was
  *creating* fog. The grid now spans camera positions unioned with the 5-95
  percentile of SfM points — camera centres have no outlier tail and match the
  5-95 point extent within a few percent, while the 1-99 percentile still spans
  69.7 units.
- **"On a surface" defined as shared voxel occupancy** tracked the resolution:
  82% of the model read as surface at 0.35-unit voxels, 71% at 0.06-unit. It is
  now a fixed metric radius.

Bedroom, one consistent definition (crop 0.5 -> 0.25, free_space_votes 3):

  ==========================================  =========  =========
  metric                                       before     after
  ==========================================  =========  =========
  fog inside the camera path                   34,653     20,609
  fog as fraction of what is in there          28.1%      18.8%
  transmittance across the room                6.7%       10.7%
  blocking mass outside the observed volume    40.3%      10.8%
  held-out quality score                       65.97      65.76
  ==========================================  =========  =========

Conclusions:

- The carve removes exactly what it can prove (16,621 Gaussians) at no cost to
  surfaces, and the tighter crop cuts exterior blocking mass by four. LPIPS
  *improved* (0.401 -> 0.390) while PSNR fell 0.75 dB, which is the expected
  signature of deleting semi-transparent haze: it was contributing a small
  correct-on-average signal that PSNR rewards and perception does not.
- **Post-hoc carving cannot finish the job.** 20,609 Gaussians remain in the
  walked volume with no surface near them, in voxels no ray reaches — rays only
  travel toward SfM points, so volume in front of textureless surfaces is never
  carved. Transmittance 10.7% is still far from clear.
- That residue is a *training* problem, which is what the anti-fog work targets:
  3DGRUT already ships ``loss.use_opacity`` and ``strategy.prune_weight``
  (accumulated-evidence pruning), both disabled by default. The literature is
  explicit that photometric loss alone cannot remove floaters, because their
  opacity gradients vanish once the blended colour reaches equilibrium.

Two bugs found by running this for real, both fixed:

1. ``tools/isaac/render_heldout.py`` produced zero images — ``app.update()`` does
   not trigger Replicator annotators; a synchronous ``rep.orchestrator.step()``
   per view is required. ``--limit N`` added for fast smoke tests.
2. The floor gate used inlier ratio, which cannot distinguish a good floor from a
   broken one (kitchen 6.1% vs desk 3.9%). Replaced by
   ``geometry.max_points_below_floor`` (kitchen 2.4% vs desk 34.4%).

ROOT CAUSE of the (unusable) desk build (2026-07-26)
-----------------------------------------------------
The desk scan is **not tunable — it must be recaptured**. COLMAP registered
**2 of 113 frames**; the whole workspace is built on those two views:

- ``IMG_0056.mov``: 219 s, 6571 frames → 113 extracted (the rest dropped as
  blurry); surviving frames have median Laplacian variance ~67 (sharp ≈ 300+),
  so the footage is soft throughout, and 113 frames over 219 s leaves ~1.9 s
  between views — far too little overlap for feature matching.
- Consequences: 1640-point sparse cloud; degenerate intrinsics (fx 6763 vs
  fy 4192 on 1920×1080); floor plane fit to 64 points (3.9% inliers) that sits
  mid-cloud, not on any surface; 0.226 m splat-vs-proxy registration error.
- Recapture guidance: slow continuous motion, locked exposure/focus, plenty of
  overlap, good light; then ``--video-stride 8`` (~800 candidate frames) and
  confirm ``reconstruct`` reports a high registration rate before building.

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
