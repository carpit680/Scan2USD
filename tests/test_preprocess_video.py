from pathlib import Path

from scan2usd.preprocess.video import clear_frame_images, frames_dir_has_images


def test_clear_frame_images_removes_only_images(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.PNG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    assert frames_dir_has_images(tmp_path)
    assert clear_frame_images(tmp_path) == 2
    assert not frames_dir_has_images(tmp_path)
    assert (tmp_path / "notes.txt").is_file()


def test_list_frame_images_sorted_and_filtered(tmp_path):
    from scan2usd.preprocess.video import list_frame_images

    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_bytes(b"x")

    names = [p.name for p in list_frame_images(tmp_path)]
    assert names == ["a.png", "b.jpg"]
    assert list_frame_images(tmp_path / "missing") == []


def test_extract_frames_respects_max_dim(tmp_path):
    """Frames are downscaled on write so COLMAP's CPU SIFT sees fewer pixels."""
    import cv2
    import numpy as np
    from scan2usd.preprocess.video import extract_frames

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    rng = np.random.default_rng(0)
    for _ in range(4):
        writer.write(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    writer.release()
    if not video.is_file():
        import pytest

        pytest.skip("OpenCV cannot write mp4 here")

    out = tmp_path / "frames"
    paths = extract_frames(video, out, stride=1, blur_threshold=0, max_dim=640)
    assert paths
    img = cv2.imread(str(paths[0]))
    assert max(img.shape[:2]) == 640
