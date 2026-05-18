from pathlib import Path

from scan2usd.dataset.split import ImageRecord, assign_train_val, resolve_label_path


def test_session_split_deterministic(tmp_path: Path) -> None:
    recs = [
        ImageRecord(tmp_path / "a1.jpg", session="s1"),
        ImageRecord(tmp_path / "a2.jpg", session="s1"),
        ImageRecord(tmp_path / "b1.jpg", session="s2"),
    ]
    tr, va = assign_train_val(recs, strategy="session", val_sessions=["s2"], val_ratio=0.2, seed=0)
    assert {r.session for r in va} == {"s2"}
    assert {r.session for r in tr} == {"s1"}


def test_random_split_ratio(tmp_path: Path) -> None:
    recs = [ImageRecord(tmp_path / f"f{i}.jpg", session="s") for i in range(10)]
    tr, va = assign_train_val(recs, strategy="random_frame", val_ratio=0.3, seed=123)
    assert len(va) >= 1
    assert len(tr) + len(va) == 10


def test_single_session_flat_dir_gets_train_and_val(tmp_path: Path) -> None:
    recs = [ImageRecord(tmp_path / f"frame_{i:06d}.jpg", session="default") for i in range(20)]
    tr, va = assign_train_val(recs, strategy="session", val_ratio=0.2, seed=42)
    assert len(tr) > 0
    assert len(va) > 0
    assert len(tr) + len(va) == 20


def test_resolve_label_path_preprocess_to_colmap(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "frame_000058.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    p = resolve_label_path(labels, "frame_00059.jpg")
    assert p.name == "frame_000058.txt"
    assert p.is_file()
