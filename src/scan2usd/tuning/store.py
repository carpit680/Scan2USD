"""Persistent trial store for the auto-tuner (workspace/tuning/trials.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRIALS_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrialRecord:
    trial_id: str
    kind: str  # "cheap" (no retrain) | "retrain"
    params: dict[str, Any]
    status: str = "running"  # running | scored | failed
    quality_score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    render_dir: str | None = None
    artifacts_dir: str | None = None
    started_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None


class TrialStore:
    """Append/update trials in a JSON file; safe to reload for resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.trials: list[TrialRecord] = []
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.trials = [TrialRecord(**item) for item in raw.get("trials", [])]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRIALS_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "trials": [asdict(trial) for trial in self.trials],
        }
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def upsert(self, trial: TrialRecord) -> None:
        for index, existing in enumerate(self.trials):
            if existing.trial_id == trial.trial_id:
                self.trials[index] = trial
                self.save()
                return
        self.trials.append(trial)
        self.save()

    def scored_params(self, kind: str) -> list[dict[str, Any]]:
        return [t.params for t in self.trials if t.kind == kind and t.status == "scored"]

    def best(self) -> TrialRecord | None:
        scored = [
            t for t in self.trials if t.status == "scored" and t.quality_score is not None
        ]
        if not scored:
            return None
        return max(scored, key=lambda t: t.quality_score)

    def next_trial_id(self, kind: str) -> str:
        count = sum(1 for t in self.trials if t.kind == kind)
        return f"{kind}_{count + 1:03d}"
