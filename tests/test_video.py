from pathlib import Path

from scan2usd.preprocess.video import VIDEO_SUFFIXES, is_supported_video_suffix


def test_mov_suffix_supported() -> None:
    assert is_supported_video_suffix(Path("walkthrough.MOV"))
    assert is_supported_video_suffix(Path("clip.mov"))
    assert ".mov" in VIDEO_SUFFIXES


def test_mp4_supported() -> None:
    assert is_supported_video_suffix(Path("x.mp4"))
