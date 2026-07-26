"""SAM2.1 video-mask runner for Scan2USD's external propagation contract."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor


def _eligible_instances(raw: dict, *, maximum: int, min_views: int) -> list[dict]:
    instances = [
        item
        for item in raw.get("instances", [])
        if len(item.get("detections", [])) >= min_views
    ]
    instances.sort(
        key=lambda item: (
            float(item.get("confidence", 0.0)) * np.log2(len(item["detections"]) + 1),
            len(item["detections"]),
        ),
        reverse=True,
    )
    return instances[:maximum]


def _best_prompt(instance: dict, frame_to_index: dict[str, int]) -> tuple[int, np.ndarray]:
    detections = [
        detection
        for detection in instance["detections"]
        if detection["frame"] in frame_to_index
    ]
    if not detections:
        raise RuntimeError(f"No proposal frames exist for {instance['instance_id']}")
    detection = max(detections, key=lambda item: float(item["confidence"]))
    return frame_to_index[detection["frame"]], np.asarray(detection["xyxy"], dtype=np.float32)


def _save_outputs(
    outputs: dict[tuple[str, int], np.ndarray],
    frames: list[Path],
    output_root: Path,
    *,
    min_area: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (instance_id, frame_index), mask in outputs.items():
        binary = np.asarray(mask > 0, dtype=np.uint8) * 255
        if int((binary > 0).sum()) < min_area:
            continue
        target_dir = output_root / instance_id
        target_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(target_dir / f"{frames[frame_index].stem}.png")
        counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def _sam2_video_dir(frames: list[Path]) -> tuple[Path, dict[int, Path]]:
    """SAM2 expects integer-named frames; stage symlinks and map indices back."""
    staging = Path(tempfile.mkdtemp(prefix="scan2usd_sam2_"))
    index_to_frame: dict[int, Path] = {}
    for index, frame in enumerate(frames):
        link = staging / f"{index:05d}{frame.suffix.lower()}"
        try:
            link.symlink_to(frame.resolve())
        except OSError:
            shutil.copy2(frame, link)
        index_to_frame[index] = frame
    return staging, index_to_frame


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 0, path.name


def _propagate_batch(
    predictor: SAM2VideoPredictor,
    video_dir: Path,
    frames: list[Path],
    instances: list[dict],
    *,
    device: str,
) -> dict[tuple[str, int], np.ndarray]:
    frame_to_index = {path.name: index for index, path in enumerate(frames)}
    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )
    prompt_indices: list[int] = []
    for instance in instances:
        frame_index, box = _best_prompt(instance, frame_to_index)
        prompt_indices.append(frame_index)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=instance["instance_id"],
            box=box,
        )

    outputs: dict[tuple[str, int], np.ndarray] = {}

    def consume(generator) -> None:
        for frame_index, object_ids, mask_logits in generator:
            for index, object_id in enumerate(object_ids):
                outputs[(str(object_id), int(frame_index))] = (
                    mask_logits[index, 0].detach().float().cpu().numpy()
                )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        consume(
            predictor.propagate_in_video(
                state,
                start_frame_idx=min(prompt_indices),
                reverse=False,
            )
        )
        if max(prompt_indices) > 0:
            consume(
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=max(prompt_indices),
                    reverse=True,
                )
            )
    predictor.reset_state(state)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-format", default="foreground-white")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-mask-area", type=int, default=64)
    args = parser.parse_args()

    if args.mask_format != "foreground-white":
        raise ValueError("Only foreground-white masks are supported")
    frames = sorted(
        (
            path
            for path in args.images.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=_frame_sort_key,
    )
    if not frames:
        raise RuntimeError(f"No frames under {args.images}")
    video_dir, _index_to_frame = _sam2_video_dir(frames)
    raw = json.loads(args.proposals.read_text(encoding="utf-8"))
    instances = _eligible_instances(
        raw,
        maximum=args.max_objects,
        min_views=args.min_views,
    )
    if not instances:
        raise RuntimeError("No proposals meet SAM2 propagation thresholds")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(args.model, device=device)
    all_counts: dict[str, int] = {}
    try:
        for start in range(0, len(instances), args.batch_size):
            batch = instances[start : start + args.batch_size]
            outputs = _propagate_batch(
                predictor,
                video_dir,
                frames,
                batch,
                device=device,
            )
            counts = _save_outputs(
                outputs,
                frames,
                args.output,
                min_area=args.min_mask_area,
            )
            all_counts.update(counts)
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(video_dir, ignore_errors=True)

    report = {
        "model": args.model,
        "device": device,
        "candidate_instances": len(raw.get("instances", [])),
        "propagated_instances": len(instances),
        "mask_counts": all_counts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sam2_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
