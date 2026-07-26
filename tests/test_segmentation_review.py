from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import SceneManifest
from scan2usd.review.app import ReviewSession
from scan2usd.segmentation.propagate import import_masks_into_manifest
from scan2usd.segmentation.propose import FrameDetection, associate_detections


def test_adjacent_same_class_detections_become_one_instance():
    detections = [
        FrameDetection("frame_000.jpg", "chair", 0.9, (10, 10, 30, 30)),
        FrameDetection("frame_001.jpg", "chair", 0.8, (11, 10, 31, 30)),
        FrameDetection("frame_001.jpg", "chair", 0.7, (60, 10, 80, 30)),
    ]
    proposals = associate_detections(detections)
    assert len(proposals) == 2
    assert len(proposals[0].detections) == 2
    assert proposals[0].instance_id == "chair_001"


def test_mask_import_and_review_gate(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.segmentation.masks_dir = tmp_path / "masks"
    cfg.segmentation.min_views_per_object = 3
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    proposals = associate_detections(
        [
            FrameDetection(f"frame_{i:03d}.jpg", "box", 0.9, (2, 2, 10, 10))
            for i in range(3)
        ]
    )
    mask_dir = cfg.segmentation.masks_dir / proposals[0].instance_id
    mask_dir.mkdir(parents=True)
    for i in range(3):
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 2:10] = 255
        Image.fromarray(mask).save(mask_dir / f"frame_{i:03d}.png")
    report = import_masks_into_manifest(
        cfg,
        manifest,
        proposals,
        cfg.segmentation.masks_dir,
    )
    assert report["box_001"]["meets_min_views"]

    path = manifest.save(tmp_path / "scene_manifest.json")
    review = ReviewSession(cfg, path)
    review.update_object(
        "box_001",
        class_name="box",
        movable=True,
        review_state="approved",
        observed_background_coverage=0.95,
    )
    review.approve_segmentation(reviewer="tester")
    loaded = SceneManifest.load(path)
    assert loaded.approvals["segmentation"]["state"] == "approved"


def test_review_rejects_unapproved_instance(tmp_path):
    cfg = SceneConfig()
    cfg.segmentation.min_views_per_object = 1
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    mask_dir = tmp_path / "masks" / "obj_001"
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(mask_dir / "frame.png")
    manifest.objects.append(
        ObjectRecord(
            instance_id="obj_001",
            display_name="Object",
            mask_dir=str(mask_dir),
        )
    )
    path = manifest.save(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError, match="state=pending"):
        ReviewSession(cfg, path).approve_segmentation(reviewer="tester")


def test_approve_segmentation_allows_holes_when_configured(tmp_path):
    cfg = SceneConfig()
    cfg.segmentation.min_views_per_object = 1
    cfg.qa.min_background_coverage = 0.9
    cfg.qa.allow_background_holes = True
    cfg.capture.clean_plate_dir = None
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="production",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    mask_dir = tmp_path / "masks" / "obj_001"
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(mask_dir / "frame.png")
    manifest.objects.append(
        ObjectRecord(
            instance_id="obj_001",
            display_name="Obj",
            mask_dir=str(mask_dir),
            review_state="approved",
            movable=True,
            observed_background_coverage=0.0,
        )
    )
    path = manifest.save(tmp_path / "manifest.json")
    ReviewSession(cfg, path).approve_segmentation(reviewer="dev")
    loaded = SceneManifest.load(path)
    assert loaded.approvals["segmentation"]["state"] == "approved"


def test_approve_segmentation_ignores_maskless_and_rejected(tmp_path):
    cfg = SceneConfig()
    cfg.segmentation.min_views_per_object = 1
    cfg.qa.min_background_coverage = 0.0
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    good_dir = tmp_path / "masks" / "good_001"
    good_dir.mkdir(parents=True)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(good_dir / "frame.png")
    bad_dir = tmp_path / "masks" / "bad_001"
    bad_dir.mkdir(parents=True)
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(bad_dir / "frame.png")
    manifest.objects.extend(
        [
            ObjectRecord(
                instance_id="good_001",
                display_name="Good",
                mask_dir=str(good_dir),
                review_state="approved",
                observed_background_coverage=1.0,
            ),
            ObjectRecord(
                instance_id="bad_001",
                display_name="Bad",
                mask_dir=str(bad_dir),
                review_state="rejected",
            ),
            ObjectRecord(
                instance_id="empty_001",
                display_name="Empty",
                mask_dir=str(tmp_path / "missing"),
                review_state="pending",
            ),
        ]
    )
    path = manifest.save(tmp_path / "manifest.json")
    ReviewSession(cfg, path).approve_segmentation(reviewer="tester")
    loaded = SceneManifest.load(path)
    assert loaded.approvals["segmentation"]["state"] == "approved"
    assert loaded.get_object("good_001").review_state == "approved"
    assert loaded.get_object("bad_001").review_state == "rejected"
    assert loaded.get_object("empty_001").review_state == "pending"


def test_merge_instances_unions_masks_and_rejects_sources(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.segmentation.masks_dir = tmp_path / "masks"
    cfg.workspace_dir.mkdir(parents=True)
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    primary_dir = cfg.segmentation.masks_dir / "speaker_001"
    other_dir = cfg.segmentation.masks_dir / "speaker_002"
    primary_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    # Primary has frame_000 with small blob; source has denser same frame + extra frame
    small = np.zeros((16, 16), dtype=np.uint8)
    small[4:8, 4:8] = 255
    Image.fromarray(small).save(primary_dir / "frame_000.png")
    large = np.zeros((16, 16), dtype=np.uint8)
    large[2:14, 2:14] = 255
    Image.fromarray(large).save(other_dir / "frame_000.png")
    Image.fromarray(large).save(other_dir / "frame_001.png")
    manifest.objects.extend(
        [
            ObjectRecord(
                instance_id="speaker_001",
                display_name="Speaker 1",
                class_name="speaker",
                mask_dir=str(primary_dir),
                review_state="approved",
                observed_background_coverage=0.5,
            ),
            ObjectRecord(
                instance_id="speaker_002",
                display_name="Speaker 2",
                class_name="speaker",
                mask_dir=str(other_dir),
                review_state="pending",
                observed_background_coverage=0.9,
            ),
        ]
    )
    path = manifest.save(tmp_path / "manifest.json")
    session = ReviewSession(cfg, path)
    result = session.merge_instances("speaker_001", ["speaker_002"])
    assert result["frames_added"] == 1
    assert result["frames_replaced"] == 1
    assert result["mask_count_before"] == 1
    assert result["mask_count"] == 2
    loaded = SceneManifest.load(path)
    primary = loaded.get_object("speaker_001")
    source = loaded.get_object("speaker_002")
    assert primary.review_state == "approved"
    assert primary.physics["merged_from"] == ["speaker_002"]
    assert primary.observed_background_coverage == 0.9
    assert source.review_state == "rejected"
    assert source.physics["merged_into"] == "speaker_001"
    assert (primary_dir / "frame_001.png").is_file()
    assert ReviewSession._mask_foreground_count(primary_dir / "frame_000.png") > 16

    # Revert the merge
    undo = session.unmerge_instances("speaker_001", ["speaker_002"])
    assert undo["restored"] == ["speaker_002"]
    assert undo["frames_removed"] == 1
    assert undo["frames_restored"] == 1
    assert undo["mask_count_before"] == 2
    assert undo["mask_count"] == 1
    loaded2 = SceneManifest.load(path)
    primary2 = loaded2.get_object("speaker_001")
    source2 = loaded2.get_object("speaker_002")
    assert "merged_from" not in primary2.physics
    assert primary2.observed_background_coverage == 0.5
    assert source2.review_state == "pending"
    assert "merged_into" not in source2.physics
    assert not (primary_dir / "frame_001.png").is_file()
    assert ReviewSession._mask_foreground_count(primary_dir / "frame_000.png") == int(
        (small > 127).sum()
    )


def test_merge_or_unions_complementary_regions(tmp_path):
    """Same frame, different regions: merge should OR (not denser-wins replace)."""
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.workspace_dir.mkdir()
    cfg.segmentation.masks_dir = tmp_path / "masks"
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    (cfg.nerfstudio_data_dir / "images").mkdir(parents=True)
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    primary_dir = cfg.segmentation.masks_dir / "obj_001"
    other_dir = cfg.segmentation.masks_dir / "obj_002"
    primary_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    left = np.zeros((16, 16), dtype=np.uint8)
    left[2:8, 2:6] = 255
    right = np.zeros((16, 16), dtype=np.uint8)
    right[2:8, 10:14] = 255
    Image.fromarray(left).save(primary_dir / "frame_000.png")
    Image.fromarray(right).save(other_dir / "frame_000.png")
    # Source also has an exclusive frame so mask_count increases
    Image.fromarray(right).save(other_dir / "frame_001.png")
    manifest.objects.extend(
        [
            ObjectRecord(
                instance_id="obj_001",
                display_name="Obj 1",
                class_name="obj",
                mask_dir=str(primary_dir),
                review_state="approved",
            ),
            ObjectRecord(
                instance_id="obj_002",
                display_name="Obj 2",
                class_name="obj",
                mask_dir=str(other_dir),
                review_state="pending",
            ),
        ]
    )
    path = manifest.save(tmp_path / "manifest.json")
    session = ReviewSession(cfg, path)
    result = session.merge_instances("obj_001", ["obj_002"])
    assert result["mask_count_before"] == 1
    assert result["frames_added"] == 1
    assert result["frames_replaced"] == 1
    assert result["mask_count"] == 2
    union_fg = ReviewSession._mask_foreground_count(primary_dir / "frame_000.png")
    assert union_fg == int((left > 127).sum()) + int((right > 127).sum())

    undo = session.unmerge_instances("obj_001", ["obj_002"])
    assert undo["mask_count"] == 1
    assert undo["frames_removed"] == 1
    assert ReviewSession._mask_foreground_count(primary_dir / "frame_000.png") == int(
        (left > 127).sum()
    )


def test_delete_mask_updates_count(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.workspace_dir.mkdir()
    cfg.segmentation.masks_dir = tmp_path / "masks"
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    images = cfg.nerfstudio_data_dir / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images / "frame_000.jpg")
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images / "frame_001.jpg")
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    mask_dir = cfg.segmentation.masks_dir / "obj_001"
    mask_dir.mkdir(parents=True)
    blob = np.zeros((8, 8), dtype=np.uint8)
    blob[2:6, 2:6] = 255
    Image.fromarray(blob).save(mask_dir / "frame_000.png")
    Image.fromarray(blob).save(mask_dir / "frame_001.png")
    manifest.objects.append(
        ObjectRecord(
            instance_id="obj_001",
            display_name="Obj",
            mask_dir=str(mask_dir),
        )
    )
    path = manifest.save(tmp_path / "manifest.json")
    session = ReviewSession(cfg, path)
    overlays = session.render_overlays("obj_001")
    assert len(overlays) == 2
    assert overlays[0]["mask_name"] == "frame_000.png"
    result = session.delete_mask("obj_001", "frame_000.png")
    assert result["mask_count"] == 1
    assert result["deleted"] == ["frame_000.png"]
    assert not (mask_dir / "frame_000.png").is_file()
    assert len(session.mask_files("obj_001")) == 1
    assert len(session.render_overlays("obj_001")) == 1

    # batch delete remaining
    Image.fromarray(blob).save(mask_dir / "frame_000.png")
    Image.fromarray(blob).save(mask_dir / "frame_002.png")
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images / "frame_002.jpg")
    batch = session.delete_masks("obj_001", ["frame_000.png", "frame_001.png", "frame_002.png"])
    assert sorted(batch["deleted"]) == ["frame_000.png", "frame_001.png", "frame_002.png"]
    assert batch["mask_count"] == 0
    assert session.mask_files("obj_001") == []


def test_reclassify_renames_instance_and_moves_masks(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.workspace_dir.mkdir()
    cfg.segmentation.masks_dir = tmp_path / "masks"
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    speaker_dir = cfg.segmentation.masks_dir / "speaker_001"
    speaker_dir.mkdir(parents=True)
    blob = np.zeros((8, 8), dtype=np.uint8)
    blob[2:6, 2:6] = 255
    Image.fromarray(blob).save(speaker_dir / "frame_000.png")
    kb_dir = cfg.segmentation.masks_dir / "keyboard_001"
    kb_dir.mkdir(parents=True)
    Image.fromarray(blob).save(kb_dir / "frame_000.png")
    manifest.objects.extend(
        [
            ObjectRecord(
                instance_id="speaker_001",
                display_name="Speaker 001",
                class_name="speaker",
                mask_dir=str(speaker_dir),
                review_state="approved",
            ),
            ObjectRecord(
                instance_id="keyboard_001",
                display_name="Keyboard 001",
                class_name="keyboard",
                mask_dir=str(kb_dir),
            ),
            ObjectRecord(
                instance_id="speaker_002",
                display_name="Speaker 002",
                class_name="speaker",
                physics={"merged_into": "speaker_001"},
            ),
        ]
    )
    path = manifest.save(tmp_path / "manifest.json")
    session = ReviewSession(cfg, path)

    preview = session.preview_reclassify("speaker_001", "keyboard")
    assert preview["will_rename"] is True
    assert preview["new_instance_id"] == "keyboard_002"

    updated = session.update_object(
        "speaker_001",
        class_name="keyboard",
        movable=True,
        review_state="approved",
        observed_background_coverage=0.5,
        reclassify=True,
    )
    assert updated.instance_id == "keyboard_002"
    assert updated.class_name == "keyboard"
    assert not speaker_dir.exists()
    assert (cfg.segmentation.masks_dir / "keyboard_002" / "frame_000.png").is_file()
    loaded = SceneManifest.load(path)
    assert loaded.get_object("keyboard_002").mask_dir.endswith("keyboard_002")
    assert loaded.get_object("speaker_002").physics["merged_into"] == "keyboard_002"
    with pytest.raises(KeyError):
        loaded.get_object("speaker_001")


def test_review_gradio_app_builds_without_starting_server(tmp_path, monkeypatch):
    gradio = pytest.importorskip("gradio")
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    images = cfg.nerfstudio_data_dir / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(80, 100, 120)).save(images / "frame.jpg")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(mask_dir / "frame.png")
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    from scan2usd.pipeline.manifest import ObjectRecord

    manifest.objects.append(
        ObjectRecord(
            instance_id="obj_001",
            display_name="Object",
            mask_dir=str(mask_dir),
        )
    )
    path = manifest.save(tmp_path / "manifest.json")
    monkeypatch.setattr(gradio.Blocks, "launch", lambda self, **kwargs: self)
    from scan2usd.review.app import launch_review_app

    app = launch_review_app(cfg, path)
    assert isinstance(app, gradio.Blocks)
