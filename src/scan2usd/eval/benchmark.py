from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scan2usd.config import SceneConfig


def train_yolo(
    data_yaml: Path,
    cfg: SceneConfig,
    *,
    experiment: str,
    project: Path,
    name: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(cfg.yolo_model)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.train_epochs,
        imgsz=cfg.train_imgsz,
        batch=cfg.train_batch,
        seed=cfg.seed,
        project=str(project.resolve()),
        name=f"{name}_{experiment}",
        exist_ok=True,
        verbose=False,
        workers=0,
    )
    trainer = model.trainer
    best = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
    if best is None or not best.exists():
        root = project / f"{name}_{experiment}"
        cand = sorted(root.glob("**/weights/best.pt"))
        best = cand[-1] if cand else None
    metrics = {}
    if trainer is not None and hasattr(trainer, "metrics") and trainer.metrics is not None:
        metrics = dict(trainer.metrics.results_dict) if hasattr(trainer.metrics, "results_dict") else {}
    return {"metrics": metrics, "best_weights": str(best.resolve()) if best else None}


def val_yolo(weights: Path, data_yaml: Path, cfg: SceneConfig) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml.resolve()), imgsz=cfg.train_imgsz, split="val", verbose=False)
    out: dict[str, Any] = {}
    if metrics is not None:
        if hasattr(metrics, "results_dict"):
            out.update(metrics.results_dict)
        elif isinstance(metrics, dict):
            out.update(metrics)
    return out


def run_experiment(
    experiment: str,
    data_yaml: Path,
    cfg: SceneConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Train one experiment; return metrics + paths to best weights.

    Experiments differ only by ``data.yaml`` content / dataset root they point at.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_out = train_yolo(
        data_yaml,
        cfg,
        experiment=experiment,
        project=out_dir / f"runs_{experiment}",
        name="train",
    )
    weights = Path(train_out["best_weights"]) if train_out.get("best_weights") else None
    val_metrics: dict[str, Any] = {}
    if weights and weights.exists():
        val_metrics = val_yolo(weights, data_yaml, cfg)
    report = {
        "experiment": experiment,
        "data_yaml": str(data_yaml.resolve()),
        "train_metrics": train_out,
        "val_metrics": val_metrics,
        "weights": str(weights.resolve()) if weights and weights.exists() else None,
    }
    (out_dir / f"report_{experiment}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def compare_abc(reports_dir: Path) -> dict[str, Any]:
    def load(ex: str) -> dict[str, Any]:
        p = reports_dir / f"report_{ex}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    a, b, c = load("A"), load("B"), load("C")

    def mget(rep: dict[str, Any], *keys: str) -> float | None:
        vm = rep.get("val_metrics") or {}
        for k in keys:
            if k in vm:
                v = vm[k]
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    map_a = mget(a, "metrics/mAP50-95(B)", "mAP50-95", "maps", "map")
    map_c = mget(c, "metrics/mAP50-95(B)", "mAP50-95", "maps", "map")
    summary = {
        "mAP_A": map_a,
        "mAP_B": mget(b, "metrics/mAP50-95(B)", "mAP50-95", "maps", "map"),
        "mAP_C": map_c,
        "goal_C_gt_A": (map_c is not None and map_a is not None and map_c > map_a),
    }
    (reports_dir / "summary_ABC.json").write_text(json.dumps(summary, indent=2))
    return summary
