"""Open-vocabulary 2D proposals and lightweight temporal instance association."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from scan2usd.config import SceneConfig


@dataclass(frozen=True)
class FrameDetection:
    frame: str
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    source: str = "open-vocabulary"


@dataclass
class InstanceProposal:
    instance_id: str
    class_name: str
    detections: list[FrameDetection] = field(default_factory=list)
    confidence: float = 0.0


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def associate_detections(
    detections: Iterable[FrameDetection],
    *,
    min_iou: float = 0.25,
    max_frame_gap: int = 2,
) -> list[InstanceProposal]:
    """
    Associate adjacent-frame proposals by class and IoU.

    This generates review candidates only; approval and SAM-style propagation
    are required before a candidate becomes a production object.
    """
    by_frame: dict[str, list[FrameDetection]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    frames = sorted(by_frame)
    active: list[tuple[InstanceProposal, int, tuple[float, float, float, float]]] = []
    output: list[InstanceProposal] = []
    counters: dict[str, int] = {}
    for frame_index, frame in enumerate(frames):
        used: set[int] = set()
        for detection in sorted(by_frame[frame], key=lambda item: item.confidence, reverse=True):
            best_index = None
            best_iou = min_iou
            for index, (proposal, last_index, last_box) in enumerate(active):
                if index in used:
                    continue
                if proposal.class_name != detection.class_name:
                    continue
                if frame_index - last_index > max_frame_gap:
                    continue
                overlap = box_iou(last_box, detection.xyxy)
                if overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index is None:
                count = counters.get(detection.class_name, 0) + 1
                counters[detection.class_name] = count
                safe_class = "".join(
                    char if char.isalnum() else "_" for char in detection.class_name.lower()
                ).strip("_") or "object"
                proposal = InstanceProposal(
                    instance_id=f"{safe_class}_{count:03d}",
                    class_name=detection.class_name,
                    detections=[detection],
                    confidence=detection.confidence,
                )
                output.append(proposal)
                active.append((proposal, frame_index, detection.xyxy))
                used.add(len(active) - 1)
            else:
                proposal, _last_index, _last_box = active[best_index]
                proposal.detections.append(detection)
                proposal.confidence = sum(d.confidence for d in proposal.detections) / len(
                    proposal.detections
                )
                active[best_index] = (proposal, frame_index, detection.xyxy)
                used.add(best_index)
        active = [
            item for item in active if frame_index - item[1] <= max_frame_gap
        ]
    return output


def _detection_from_result(result, class_names: list[str]) -> list[FrameDetection]:
    detections: list[FrameDetection] = []
    boxes = result.boxes
    if boxes is None:
        return detections
    names = getattr(result, "names", {})
    frame = Path(result.path).name
    for cls, confidence, xyxy in zip(
        boxes.cls.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.xyxy.cpu().tolist(),
    ):
        class_index = int(cls)
        class_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else str(names.get(class_index, class_index))
        )
        detections.append(
            FrameDetection(
                frame=frame,
                class_name=class_name,
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in xyxy),
                source="yolo-world",
            )
        )
    return detections


def propose_with_yolo_world(
    cfg: SceneConfig,
    *,
    images_dir: Path,
    prompts: list[str],
    confidence: float = 0.2,
) -> list[InstanceProposal]:
    """Run Ultralytics YOLO-World, already compatible with the project runtime."""
    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise RuntimeError("Ultralytics YOLOWorld is required for proposal generation") from exc
    model_name = cfg.segmentation.proposal_model
    if model_name in {"grounding-dino", "yolo-world"}:
        model_name = "yolov8s-worldv2.pt"
    model = YOLOWorld(model_name)
    model.set_classes(prompts)
    results = model.predict(
        source=str(images_dir),
        conf=confidence,
        save=False,
        verbose=False,
        stream=True,
    )
    detections: list[FrameDetection] = []
    for result in results:
        detections.extend(_detection_from_result(result, prompts))
    return associate_detections(detections)


def save_proposals(proposals: list[InstanceProposal], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "instances": [
            {
                **asdict(proposal),
                "detections": [asdict(detection) for detection in proposal.detections],
            }
            for proposal in proposals
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_proposals(path: Path) -> list[InstanceProposal]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    proposals: list[InstanceProposal] = []
    for item in raw.get("instances", []):
        detections = [
            FrameDetection(
                frame=str(det["frame"]),
                class_name=str(det["class_name"]),
                confidence=float(det["confidence"]),
                xyxy=tuple(float(value) for value in det["xyxy"]),
                source=str(det.get("source", "open-vocabulary")),
            )
            for det in item.get("detections", [])
        ]
        proposals.append(
            InstanceProposal(
                instance_id=str(item["instance_id"]),
                class_name=str(item["class_name"]),
                detections=detections,
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return proposals
