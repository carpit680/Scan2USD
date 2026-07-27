"""Prune a 3DGRUT model by measured render visibility, then export USD.

Runs under the 3DGRUT venv (``external.grut_python``), not the scan2usd venv.

Why this exists: geometric floater filters (size, position, neighbour count,
needle shape) all trade real interior fidelity for exterior tidiness, because
they cannot tell an artifact from a Gaussian that is genuinely carrying a
surface. 3DGRUT's tracer reports ``mog_visibility`` per Gaussian, so rendering
every training view and accumulating it gives the actual contribution of each
Gaussian — the criterion TrimGS uses. A Gaussian that contributes to nothing can
be removed with no cost to the observed views, which is exactly the property the
geometric filters lacked.

Prints the visibility distribution so a threshold can be chosen from data. Use
``--percentile`` to drop the least-contributing fraction, or ``--threshold`` for
an absolute cut. With neither, it only reports and exports nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("visibility_filter_export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Output .usdz/.usd")
    parser.add_argument("--dataset", default=None, help="Override dataset path from checkpoint")
    parser.add_argument("--format", default="nurec", choices=["nurec", "standard"])
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Drop this percentage of Gaussians with the lowest measured visibility",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Drop Gaussians below this visibility"
    )
    parser.add_argument("--no-transform", action="store_true", default=True)
    parser.add_argument(
        "--report", type=Path, default=None, help="Write the visibility distribution as JSON"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Running this file by absolute path puts tools/geometry on sys.path, not the
    # 3DGRUT checkout, so make the import work from GRUT_ROOT or the cwd.
    import os

    grut_root = os.environ.get("GRUT_ROOT") or os.getcwd()
    if grut_root not in sys.path:
        sys.path.insert(0, grut_root)

    from threedgrut.export import compute_average_visibility
    from threedgrut.export.scripts.export_usd import load_model_from_checkpoint
    from threedgrut.export.usd import NuRecExporter, USDExporter

    model, conf, background = load_model_from_checkpoint(str(args.checkpoint))
    total = int(model.get_positions().shape[0])
    logger.info("loaded %s Gaussians", f"{total:,}")

    import threedgrut.datasets as datasets

    if args.dataset:
        conf.path = args.dataset
    train_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)[0]
    logger.info("training views: %d", len(train_dataset))

    visibility = np.asarray(compute_average_visibility(model, train_dataset, conf))
    logger.info(
        "visibility: min %.3e  median %.3e  mean %.3e  max %.3e",
        visibility.min(),
        float(np.median(visibility)),
        visibility.mean(),
        visibility.max(),
    )
    distribution = {}
    for q in (0.1, 1, 5, 10, 25, 50, 75, 100):
        value = float(np.percentile(visibility, q))
        distribution[f"p{q}"] = value
        logger.info("  p%-5s %.6e   (<=: %s Gaussians)", q, value, f"{int((visibility <= value).sum()):,}")
    zero = int((visibility <= 0).sum())
    logger.info("contributing to nothing: %s (%.2f%%)", f"{zero:,}", 100.0 * zero / total)

    if args.report:
        import json

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "total_gaussians": total,
                    "zero_visibility": zero,
                    "percentiles": distribution,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.percentile is None and args.threshold is None:
        logger.info("no --percentile/--threshold given: reporting only, not exporting")
        return
    if args.output is None:
        logger.error("--output is required when filtering")
        sys.exit(1)

    cutoff = (
        float(np.percentile(visibility, args.percentile))
        if args.percentile is not None
        else float(args.threshold)
    )
    keep = visibility > cutoff
    logger.info(
        "cutoff %.6e -> keeping %s/%s (%.1f%%)",
        cutoff,
        f"{int(keep.sum()):,}",
        f"{total:,}",
        100.0 * keep.mean(),
    )

    import torch

    keep_idx = torch.from_numpy(np.flatnonzero(keep)).to(model.get_positions().device)
    for name in (
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    ):
        param = getattr(model, name, None)
        if param is None:
            continue
        with torch.no_grad():
            setattr(model, name, torch.nn.Parameter(param[keep_idx], requires_grad=False))
    logger.info("model now has %s Gaussians", f"{int(model.get_positions().shape[0]):,}")

    exporter = (
        NuRecExporter()
        if args.format == "nurec"
        else USDExporter(apply_normalizing_transform=not args.no_transform)
    )
    exporter.export(
        model=model,
        output_path=args.output,
        dataset=None,
        conf=conf,
        background=background,
    )
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
