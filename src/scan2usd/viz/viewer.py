"""Nerfstudio splat viewer with Scan2USD 3D box overlay."""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

from scan2usd.reconstruction.nerfstudio import _apply_nerfstudio_runtime_env
from scan2usd.viz.boxes import apply_dataparser_transform_obbs, load_objects_3d, overlay_obb_wireframes


def run_splat_viewer_with_boxes(
    load_config: Path,
    objects_npz: Path,
    class_names: list[str],
    *,
    ns_data_dir: Path | None = None,
) -> None:
    """
    Start ``ns-viewer``-equivalent session and draw lifted AABBs in the same viser scene as the splat.

    Requires ``nerfstudio`` and ``viser`` in the active environment.
    """
    _apply_nerfstudio_runtime_env(os.environ)

    # Import order matters: ``nerfstudio.utils.writer`` before ``eval_setup`` triggers a
    # writer ↔ base_config circular import (AttributeError on LocalWriter).
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.viewer.viewer import Viewer as ViewerState
    from nerfstudio.utils import writer

    class_ids, centers, rotations, halves, _bbox_min, _bbox_max = load_objects_3d(
        objects_npz, ns_data_dir=ns_data_dir
    )

    config, pipeline, _, step = eval_setup(
        load_config,
        eval_num_rays_per_chunk=None,
        test_mode="test",
    )
    num_rays_per_chunk = config.viewer.num_rays_per_chunk
    config.vis = "viewer"
    config.viewer.num_rays_per_chunk = num_rays_per_chunk

    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    viewer_callback_lock = Lock()
    viewer_state = ViewerState(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        share=config.viewer.make_share_url,
        train_lock=viewer_callback_lock,
    )
    banner_messages = viewer_state.viewer_info

    config.logging.local_writer.enable = False
    writer.setup_local_writer(
        config.logging,
        max_iter=config.max_num_iterations,
        banner_messages=banner_messages,
    )

    assert pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    viewer_state.update_scene(step=step)

    dp = pipeline.datamanager.train_dataparser_outputs
    if ns_data_dir is None:
        raise ValueError("ns_data_dir is required to place boxes in the splat viewer frame")
    centers, rotations, halves = apply_dataparser_transform_obbs(
        centers,
        rotations,
        halves,
        dp.dataparser_transform,
        float(dp.dataparser_scale),
        ns_data_dir=ns_data_dir,
    )

    from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO

    overlay_obb_wireframes(
        viewer_state.viser_server,
        class_ids,
        centers,
        rotations,
        halves,
        class_names=class_names,
        viser_scale_ratio=VISER_NERFSTUDIO_SCALE_RATIO,
    )

    for line in banner_messages:
        print(line)
    print(f"Overlaid {len(class_ids)} 3D boxes from {objects_npz}")

    while True:
        time.sleep(0.01)
