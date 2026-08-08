import json

from PIL import Image

from scan2usd.eval.photorealism import evaluate_held_out_renders


def test_identical_heldout_render_has_perfect_metrics(tmp_path):
    refs = tmp_path / "refs"
    renders = tmp_path / "renders"
    refs.mkdir()
    renders.mkdir()
    image = Image.new("RGB", (32, 24), color=(80, 120, 160))
    image.save(refs / "frame.jpg")
    image.save(renders / "frame.jpg")
    heldout = tmp_path / "held_out.json"
    heldout.write_text(json.dumps({"images": [{"file": "frame.jpg"}]}), encoding="utf-8")

    report = evaluate_held_out_renders(
        heldout,
        refs,
        renders,
        output_path=tmp_path / "report.json",
    )
    assert report["evaluated"] == 1
    assert report["mean_ssim"] > 0.999
    # JPEG round-trips can still differ by less than metric precision.
    assert report["mean_psnr"] is None or report["mean_psnr"] > 60.0


def _checker(size, shift=0):
    import numpy as np

    y, x = np.mgrid[0 : size[0], 0 : size[1]]
    board = (((x + shift) // 8 + y // 8) % 2).astype(np.float32)
    return np.stack([board, board * 0.5, 1.0 - board], axis=-1)


def test_parallel_scoring_equals_serial(tmp_path):
    """
    Workers must not change the answer.

    Parallelism that shifts a metric is worse than no parallelism, because the
    score is what every tuning decision on this project is made from.
    """
    import json

    import numpy as np
    from PIL import Image

    from scan2usd.eval.photorealism import evaluate_held_out_renders

    refs = tmp_path / "refs"
    renders = tmp_path / "renders"
    refs.mkdir()
    renders.mkdir()
    names = []
    for index in range(6):
        name = f"frame_{index:03d}.jpg"
        names.append(name)
        Image.fromarray((_checker((64, 96)) * 255).astype("uint8")).save(refs / name)
        Image.fromarray(
            (_checker((64, 96), shift=index) * 255).astype("uint8")
        ).save(renders / f"frame_{index:03d}.png")
    manifest = tmp_path / "held_out.json"
    manifest.write_text(json.dumps({"images": [{"file": n} for n in names]}))

    common = dict(
        held_out_manifest=manifest,
        reference_images_dir=refs,
        render_dir=renders,
        compute_lpips=False,
    )
    serial = evaluate_held_out_renders(output_path=tmp_path / "s.json", workers=1, **common)
    parallel = evaluate_held_out_renders(output_path=tmp_path / "p.json", workers=4, **common)

    assert serial["evaluated"] == parallel["evaluated"] == 6
    for key in ("mean_psnr", "mean_ssim"):
        assert np.isclose(serial[key], parallel[key], rtol=0, atol=1e-12)


def test_eval_resolution_is_recorded_and_changes_the_comparison(tmp_path):
    """
    The two modes are not interchangeable, so every report says which it used.

    On the real bedroom renders, switching from reference to render resolution
    left PSNR at 18.87 -> 18.89 but moved SSIM 0.798 -> 0.741: upsampling the
    render blurred both images toward each other in exactly the flat regions
    SSIM weights most.
    """
    import json

    from PIL import Image

    from scan2usd.eval.photorealism import evaluate_held_out_renders

    refs = tmp_path / "refs"
    renders = tmp_path / "renders"
    refs.mkdir()
    renders.mkdir()
    Image.fromarray((_checker((128, 192)) * 255).astype("uint8")).save(refs / "a.jpg")
    # Half-size render, as Isaac produces against a larger capture.
    Image.fromarray((_checker((64, 96), shift=3) * 255).astype("uint8")).save(
        renders / "a.png"
    )
    manifest = tmp_path / "held_out.json"
    manifest.write_text(json.dumps({"images": [{"file": "a.jpg"}]}))

    common = dict(
        held_out_manifest=manifest,
        reference_images_dir=refs,
        render_dir=renders,
        compute_lpips=False,
        workers=1,
    )
    at_render = evaluate_held_out_renders(output_path=tmp_path / "r.json", **common)
    at_reference = evaluate_held_out_renders(
        output_path=tmp_path / "f.json", at_render_resolution=False, **common
    )
    assert at_render["eval_resolution"] == "render"
    assert at_reference["eval_resolution"] == "reference"
    assert at_render["mean_ssim"] != at_reference["mean_ssim"]
