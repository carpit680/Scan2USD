from scan2usd.doctor_deps import ItemResult, _apt_install_line, _reconstruct_ready


def test_apt_install_line_dedupes_and_sorts() -> None:
    line = _apt_install_line(("ffmpeg", "colmap", "ffmpeg", "git"))
    assert line == "sudo apt update && sudo apt install -y colmap ffmpeg git"


def test_reconstruct_ready_requires_colmap_ffmpeg_ns() -> None:
    ok_col = ItemResult("colmap", True, "/x", (), True)
    bad_col = ItemResult("colmap", False, "MISSING", ("colmap",), True)
    ns_ok = [
        ItemResult("ns_process_data", True, "argv", (), True),
        ItemResult("ns_train", True, "argv", (), True),
        ItemResult("ns_render", True, "argv", (), True),
    ]
    ns_bad = [ItemResult("ns_process_data", False, "MISSING", (), True)]
    ff_ok = ItemResult("ffmpeg", True, "/f", (), True)
    ff_bad = ItemResult("ffmpeg", False, "MISSING", ("ffmpeg",), True)
    fp_ok = ItemResult("ffprobe", True, "/p", (), True)
    fp_bad = ItemResult("ffprobe", False, "MISSING", ("ffmpeg",), True)

    assert _reconstruct_ready(ok_col, ns_ok, ff_ok, fp_ok) is True
    assert _reconstruct_ready(bad_col, ns_ok, ff_ok, fp_ok) is False
    assert _reconstruct_ready(ok_col, ns_bad, ff_ok, fp_ok) is False
    assert _reconstruct_ready(ok_col, ns_ok, ff_bad, fp_ok) is False
    assert _reconstruct_ready(ok_col, ns_ok, ff_ok, fp_bad) is False
