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
