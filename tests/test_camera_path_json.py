import json
from pathlib import Path

import numpy as np

from scan2usd.synthetic.poses import write_nerfstudio_camera_path


def test_camera_path_seconds_is_scalar(tmp_path: Path) -> None:
    eye = np.eye(4)
    out = tmp_path / "camera_path.json"
    write_nerfstudio_camera_path([eye, eye], width=720, height=1280, meta={}, out_path=out)
    doc = json.loads(out.read_text())
    assert isinstance(doc["seconds"], (int, float))
    assert len(doc["camera_path"]) == 2
