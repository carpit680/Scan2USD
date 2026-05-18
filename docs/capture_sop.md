# Capture SOP (MVP)

## Goals

- Enough overlap for COLMAP / SfM (roughly **>60%** forward overlap between adjacent usable frames).
- **Stable** motion: avoid motion blur; pause briefly on direction changes if needed.
- **Coverage**: walk loops that observe target objects (chairs, tables, doors, etc.) from multiple heights and angles.

## Camera / recording

- Prefer fixed exposure or mild auto-exposure (large exposure shifts hurt multiview consistency).
- Resolution: 720p–1080p is a good default for home-scale scenes on a **12GB+** GPU; downscale if training runs OOM.

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

## ROS2 (later)

For now, export MP4 or image sequences offline. When integrating ROS2, record synchronized `sensor_msgs/Image` plus optional calibration topics into a bag, then export frames offline.
