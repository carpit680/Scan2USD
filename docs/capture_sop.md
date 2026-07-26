# Capture SOP for Hybrid USD

## Goals

- Enough overlap for COLMAP / SfM (roughly **>60%** forward overlap between adjacent usable frames).
- **Stable** motion: avoid motion blur; pause briefly on direction changes if needed.
- **Coverage**: loop around every movable rigid object from multiple heights and angles; capture its sides, top, support surface, and contact region.
- Prefer synchronized RGB-D or LiDAR for metric collision geometry. RGB-only production capture needs a visible fiducial or measured scale reference.

## Camera / recording

- Prefer fixed exposure or mild auto-exposure (large exposure shifts hurt multiview consistency).
- Resolution: 720p–1080p is a good default for home-scale scenes on a **12GB+** GPU; downscale if training runs OOM.
- Lock focus, white balance, and exposure when possible. Record calibration and retain original timestamps.

## Production passes

1. **Scene pass:** capture the room/workcell as found.
2. **Clean plate:** move approved manipulable objects away and recapture the newly exposed floor/walls from matching viewpoints.
3. **Object detail:** capture incomplete objects separately against a feature-rich background; include the complete underside/contact geometry when push/pick/place is required.
4. **Optional HDR:** capture a bracketed panorama or measured environment light for reviewed PBR relighting.

One-pass scans can produce previews, but cannot reveal occluded background or unseen object surfaces without guessing. Production validation intentionally rejects those gaps.

## What to avoid

- Whip pans, rolling shutter artifacts, and long stretches pointing at textureless walls.
- Dominating dynamic objects (people moving through most frames) for the first reconstruction pass.

## File naming (recommended)

Use `session__frame` style folder or filename prefixes so `scan2usd` can split by **session** without leakage.

Example directory layout:

```text
frames/
  morning_lr__000001.jpg
  morning_lr__000002.jpg
  evening_kitchen__000001.jpg
```

## ROS2 / RGB-D

Record synchronized color, registered depth/point cloud, `CameraInfo`, and poses/TF. Preserve the metric sensor frame so the COLMAP→USD registration can be audited.
