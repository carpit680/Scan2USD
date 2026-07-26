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
