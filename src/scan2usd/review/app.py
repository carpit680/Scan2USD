"""Review backend and optional Gradio UI for production segmentation approval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest


class ReviewSession:
    def __init__(self, cfg: SceneConfig, manifest_path: Path) -> None:
        self.cfg = cfg
        self.manifest_path = manifest_path
        self.manifest = SceneManifest.load(manifest_path)

    def reload(self) -> None:
        self.manifest = SceneManifest.load(self.manifest_path)

    def save(self) -> None:
        self.manifest.save(self.manifest_path)

    def instance_ids(self) -> list[str]:
        return [obj.instance_id for obj in self.manifest.objects]

    def object(self, instance_id: str) -> ObjectRecord:
        return self.manifest.get_object(instance_id)

    @staticmethod
    def class_slug(class_name: str) -> str:
        safe = "".join(
            char if char.isalnum() else "_" for char in (class_name or "").lower()
        ).strip("_")
        return safe or "object"

    def next_instance_id(self, class_name: str, *, exclude_id: str | None = None) -> str:
        """Allocate ``{class_slug}_{NNN}`` using the next free index for that class."""
        slug = self.class_slug(class_name)
        prefix = f"{slug}_"
        max_n = 0
        for obj in self.manifest.objects:
            if exclude_id and obj.instance_id == exclude_id:
                continue
            iid = obj.instance_id
            if not iid.startswith(prefix):
                continue
            suffix = iid[len(prefix) :]
            if suffix.isdigit():
                max_n = max(max_n, int(suffix))
        return f"{slug}_{max_n + 1:03d}"

    def preview_reclassify(self, instance_id: str, class_name: str) -> dict[str, Any]:
        obj = self.object(instance_id)
        new_class = (class_name or "").strip() or "unknown"
        new_slug = self.class_slug(new_class)
        current_slug = self.class_slug(obj.class_name)
        will_rename = new_slug != current_slug
        new_id = (
            self.next_instance_id(new_class, exclude_id=instance_id)
            if will_rename
            else instance_id
        )
        return {
            "instance_id": instance_id,
            "class_name": new_class,
            "current_class_name": obj.class_name,
            "will_rename": will_rename and new_id != instance_id,
            "new_instance_id": new_id if (will_rename and new_id != instance_id) else instance_id,
        }

    def _replace_instance_id_refs(self, old_id: str, new_id: str) -> None:
        for obj in self.manifest.objects:
            if obj.physics.get("merged_into") == old_id:
                obj.physics["merged_into"] = new_id
            merged_from = obj.physics.get("merged_from")
            if isinstance(merged_from, list) and old_id in merged_from:
                obj.physics["merged_from"] = [
                    new_id if x == old_id else x for x in merged_from
                ]
            notes = str(obj.physics.get("review_notes") or "")
            if old_id in notes:
                obj.physics["review_notes"] = notes.replace(old_id, new_id)
        approvals = dict(self.manifest.approvals or {})
        old_gate = f"asset:{old_id}"
        new_gate = f"asset:{new_id}"
        if old_gate in approvals and new_gate not in approvals:
            approvals[new_gate] = approvals.pop(old_gate)
        elif old_gate in approvals:
            approvals.pop(old_gate, None)
        self.manifest.approvals = approvals

    def rename_instance(self, instance_id: str, new_instance_id: str) -> ObjectRecord:
        """Rename an instance id and move on-disk mask / overlay / journal dirs."""
        import shutil

        new_id = new_instance_id.strip()
        if not new_id:
            raise RuntimeError("New instance id is empty")
        if new_id == instance_id:
            return self.object(instance_id)
        if any(obj.instance_id == new_id for obj in self.manifest.objects):
            raise RuntimeError(f"Instance id already exists: {new_id}")

        obj = self.object(instance_id)
        masks_root = Path(
            self.cfg.segmentation.masks_dir or self.cfg.workspace_dir / "masks"
        )
        old_mask_dir = Path(obj.mask_dir) if obj.mask_dir else masks_root / instance_id
        new_mask_dir = masks_root / new_id
        if new_mask_dir.exists() and new_mask_dir.resolve() != old_mask_dir.resolve():
            raise RuntimeError(f"Mask directory already exists: {new_mask_dir}")

        if old_mask_dir.is_dir() and old_mask_dir.resolve() != new_mask_dir.resolve():
            new_mask_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_mask_dir), str(new_mask_dir))
            obj.mask_dir = str(new_mask_dir.resolve())
        elif obj.mask_dir:
            # Path string pointed at old id but dir missing — retarget
            obj.mask_dir = str(new_mask_dir.resolve())

        old_overlays = self.cfg.workspace_dir / "build" / "review" / instance_id
        new_overlays = self.cfg.workspace_dir / "build" / "review" / new_id
        if old_overlays.is_dir() and not new_overlays.exists():
            new_overlays.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_overlays), str(new_overlays))

        old_journal = self.cfg.workspace_dir / "build" / "review" / "merge_journal" / instance_id
        new_journal = self.cfg.workspace_dir / "build" / "review" / "merge_journal" / new_id
        if old_journal.is_dir() and not new_journal.exists():
            new_journal.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_journal), str(new_journal))

        self._replace_instance_id_refs(instance_id, new_id)
        obj.instance_id = new_id
        obj.display_name = new_id.replace("_", " ").title()
        # Replace in-place by old id lookup is gone — remove old slot then upsert
        self.manifest.objects = [
            o for o in self.manifest.objects if o.instance_id != instance_id
        ]
        # obj still had old id in list removed; now has new id
        self.manifest.upsert_object(obj)
        self.save()
        return obj

    def update_object(
        self,
        instance_id: str,
        *,
        class_name: str,
        movable: bool,
        review_state: str,
        observed_background_coverage: float,
        physics_template: str = "generic",
        notes: str = "",
        reclassify: bool = True,
    ) -> ObjectRecord:
        preview = self.preview_reclassify(instance_id, class_name)
        new_class = preview["class_name"]
        obj = self.object(instance_id)
        obj.class_name = new_class
        obj.movable = bool(movable)
        obj.review_state = review_state
        obj.observed_background_coverage = float(observed_background_coverage)
        obj.physics["template"] = physics_template.strip() or "generic"
        if notes:
            obj.physics["review_notes"] = notes
        self.manifest.upsert_object(obj)
        self.save()

        if reclassify and preview["will_rename"]:
            obj = self.rename_instance(instance_id, preview["new_instance_id"])
            # Ensure class/display stay aligned after rename
            obj.class_name = new_class
            obj.display_name = obj.instance_id.replace("_", " ").title()
            self.manifest.upsert_object(obj)
            self.save()
        return obj
    def import_corrected_masks(self, instance_id: str, files: list[Path]) -> int:
        obj = self.object(instance_id)
        target_dir = Path(obj.mask_dir) if obj.mask_dir else (
            Path(self.cfg.segmentation.masks_dir or self.cfg.workspace_dir / "masks")
            / instance_id
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for source in files:
            source = Path(source)
            if not source.is_file():
                continue
            with Image.open(source) as raw:
                mask = np.asarray(raw.convert("L"))
            binary = np.where(mask > 127, 255, 0).astype(np.uint8)
            Image.fromarray(binary).save(target_dir / f"{source.stem}.png")
            count += 1
        obj.mask_dir = str(target_dir.resolve())
        obj.review_state = "pending"
        self.manifest.upsert_object(obj)
        self.save()
        return count

    def mask_files(self, instance_id: str) -> list[Path]:
        obj = self.object(instance_id)
        if not obj.mask_dir or not Path(obj.mask_dir).is_dir():
            return []
        return sorted(Path(obj.mask_dir).glob("*.png"))

    @staticmethod
    def _mask_foreground_count(path: Path) -> int:
        with Image.open(path) as raw:
            mask = np.asarray(raw.convert("L"))
        return int((mask > 127).sum())

    def _merge_journal_dir(self, primary_id: str) -> Path:
        path = self.cfg.workspace_dir / "build" / "review" / "merge_journal" / primary_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _clear_overlays(self, instance_id: str) -> None:
        overlays = self.cfg.workspace_dir / "build" / "review" / instance_id
        if overlays.is_dir():
            for path in overlays.glob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)

    @staticmethod
    def _load_binary_mask(path: Path) -> np.ndarray:
        with Image.open(path) as raw:
            mask = np.asarray(raw.convert("L"))
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    def merge_instances(
        self,
        primary_id: str,
        source_ids: list[str],
    ) -> dict[str, Any]:
        """
        Merge duplicate instance proposals into ``primary_id``.

        Copies missing frame masks from each source into the primary mask dir.
        When both have the same frame, OR-unions the foregrounds. Sources are
        marked ``rejected`` with merge metadata; primary review decisions are kept.
        A journal + mask backups are written so the merge can be reverted.
        """
        import json
        import shutil

        primary = self.object(primary_id)
        sources = [sid.strip() for sid in source_ids if sid and sid.strip()]
        sources = [sid for sid in sources if sid != primary_id]
        if not sources:
            raise RuntimeError("Provide one or more source instance ids to merge")

        target_dir = Path(primary.mask_dir) if primary.mask_dir else (
            Path(self.cfg.segmentation.masks_dir or self.cfg.workspace_dir / "masks")
            / primary_id
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        primary.mask_dir = str(target_dir.resolve())

        mask_count_before = len(self.mask_files(primary_id))
        frames_added = 0
        frames_replaced = 0
        frames_identical = 0
        merged_ids: list[str] = []
        journal_root = self._merge_journal_dir(primary_id)
        coverage_before = float(primary.observed_background_coverage)

        for source_id in sources:
            source = self.object(source_id)
            if source.physics.get("merged_into"):
                raise RuntimeError(
                    f"{source_id} is already merged into {source.physics.get('merged_into')}"
                )

            entry_dir = journal_root / source_id
            backup_dir = entry_dir / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            added: list[str] = []
            replaced: list[str] = []

            for src_mask in self.mask_files(source_id):
                dest = target_dir / src_mask.name
                src_bin = self._load_binary_mask(src_mask)
                if not dest.is_file():
                    Image.fromarray(src_bin).save(dest)
                    frames_added += 1
                    added.append(src_mask.name)
                    continue
                dst_bin = self._load_binary_mask(dest)
                if np.array_equal(src_bin, dst_bin):
                    frames_identical += 1
                    continue
                union = np.maximum(src_bin, dst_bin)
                if np.array_equal(union, dst_bin):
                    frames_identical += 1
                    continue
                shutil.copy2(dest, backup_dir / src_mask.name)
                Image.fromarray(union).save(dest)
                frames_replaced += 1
                replaced.append(src_mask.name)

            prev_state = source.review_state
            prev_notes = str(source.physics.get("review_notes") or "")
            source.review_state = "rejected"
            note = f"Merged into {primary_id} (was {prev_state})"
            existing = prev_notes.strip()
            source.physics["review_notes"] = (
                f"{existing}; {note}".strip("; ") if existing else note
            )
            source.physics["merged_into"] = primary_id
            source.physics["merged_prev_state"] = prev_state
            source.physics["merged_prev_notes"] = prev_notes
            self.manifest.upsert_object(source)
            merged_ids.append(source_id)

            (entry_dir / "journal.json").write_text(
                json.dumps(
                    {
                        "primary_id": primary_id,
                        "source_id": source_id,
                        "prev_review_state": prev_state,
                        "prev_notes": prev_notes,
                        "source_coverage": float(source.observed_background_coverage),
                        "primary_coverage_before": coverage_before,
                        "mask_count_before": mask_count_before,
                        "frames_added": added,
                        "frames_replaced": replaced,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            self._clear_overlays(source_id)

        prior = list(primary.physics.get("merged_from") or [])
        for sid in merged_ids:
            if sid not in prior:
                prior.append(sid)
        primary.physics["merged_from"] = prior
        for source_id in merged_ids:
            src = self.object(source_id)
            primary.observed_background_coverage = max(
                float(primary.observed_background_coverage),
                float(src.observed_background_coverage),
            )
        self.manifest.upsert_object(primary)
        self.save()
        self._clear_overlays(primary_id)

        mask_count = len(self.mask_files(primary_id))
        return {
            "primary_id": primary_id,
            "merged": merged_ids,
            "mask_count_before": mask_count_before,
            "mask_count": mask_count,
            "frames_added": frames_added,
            "frames_replaced": frames_replaced,
            "frames_identical": frames_identical,
        }

    def unmerge_instances(
        self,
        primary_id: str,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Revert one or more merges into ``primary_id``.

        Restores primary masks from the merge journal (when present), returns
        sources to the list (clears ``merged_into``), and restores their prior
        review state when recorded.
        """
        import json
        import shutil

        primary = self.object(primary_id)
        merged_from = [str(x) for x in (primary.physics.get("merged_from") or [])]
        if not merged_from:
            raise RuntimeError(f"{primary_id} has no recorded merges to revert")

        to_undo = [sid.strip() for sid in (source_ids or merged_from) if sid and sid.strip()]
        to_undo = [sid for sid in to_undo if sid in merged_from]
        if not to_undo:
            raise RuntimeError("No matching merged sources to revert")

        mask_count_before = len(self.mask_files(primary_id))
        target_dir = Path(primary.mask_dir) if primary.mask_dir else None
        journal_root = self._merge_journal_dir(primary_id)
        restored: list[str] = []
        warnings: list[str] = []
        frames_removed = 0
        frames_restored = 0
        coverage_restore: float | None = None

        for source_id in to_undo:
            source = self.object(source_id)
            journal_path = journal_root / source_id / "journal.json"
            journal: dict[str, Any] | None = None
            if journal_path.is_file():
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
                if coverage_restore is None and journal.get("primary_coverage_before") is not None:
                    coverage_restore = float(journal["primary_coverage_before"])

            if journal and target_dir and target_dir.is_dir():
                backup_dir = journal_root / source_id / "backup"
                for name in journal.get("frames_added") or []:
                    path = target_dir / str(name)
                    if path.is_file():
                        path.unlink()
                        frames_removed += 1
                for name in journal.get("frames_replaced") or []:
                    backup = backup_dir / str(name)
                    dest = target_dir / str(name)
                    if backup.is_file():
                        shutil.copy2(backup, dest)
                        frames_restored += 1
                    elif dest.is_file():
                        warnings.append(
                            f"{source_id}: missing backup for {name}; left current mask"
                        )
            elif not journal:
                warnings.append(
                    f"{source_id}: no merge journal (pre-upgrade merge); "
                    "source restored but primary masks were not rolled back"
                )

            prev_state = str(
                source.physics.get("merged_prev_state")
                or (journal or {}).get("prev_review_state")
                or "pending"
            )
            if prev_state not in {"pending", "approved", "rejected", "degraded"}:
                prev_state = "pending"
            source.review_state = prev_state
            prev_notes = source.physics.get("merged_prev_notes")
            if prev_notes is None and journal is not None:
                prev_notes = journal.get("prev_notes", "")
            if prev_notes is not None:
                source.physics["review_notes"] = str(prev_notes)
            source.physics.pop("merged_into", None)
            source.physics.pop("merged_prev_state", None)
            source.physics.pop("merged_prev_notes", None)
            self.manifest.upsert_object(source)
            restored.append(source_id)
            self._clear_overlays(source_id)

            # Drop journal for this source after successful restore
            entry_dir = journal_root / source_id
            if entry_dir.is_dir():
                shutil.rmtree(entry_dir, ignore_errors=True)

        remaining = [sid for sid in merged_from if sid not in restored]
        if remaining:
            primary.physics["merged_from"] = remaining
            primary.observed_background_coverage = max(
                [float(self.object(sid).observed_background_coverage) for sid in remaining]
                + [0.0]
            )
        else:
            primary.physics.pop("merged_from", None)
            if coverage_restore is not None:
                primary.observed_background_coverage = coverage_restore
        self.manifest.upsert_object(primary)
        self.save()
        self._clear_overlays(primary_id)

        return {
            "primary_id": primary_id,
            "restored": restored,
            "remaining_merged_from": remaining,
            "mask_count_before": mask_count_before,
            "mask_count": len(self.mask_files(primary_id)),
            "frames_removed": frames_removed,
            "frames_restored": frames_restored,
            "warnings": warnings,
        }

    def delete_mask(self, instance_id: str, mask_name: str) -> dict[str, Any]:
        """Delete one mask PNG for ``instance_id`` and clear its overlay."""
        return self.delete_masks(instance_id, [mask_name])

    def delete_masks(self, instance_id: str, mask_names: list[str]) -> dict[str, Any]:
        """Delete one or more mask PNGs for ``instance_id`` and clear their overlays."""
        obj = self.object(instance_id)
        if not obj.mask_dir:
            raise RuntimeError(f"{instance_id} has no mask directory")
        target_dir = Path(obj.mask_dir)
        overlays = self.cfg.workspace_dir / "build" / "review" / instance_id
        deleted: list[str] = []
        missing: list[str] = []
        for raw_name in mask_names:
            raw = str(raw_name).strip()
            if not raw or "/" in raw or "\\" in raw or ".." in raw:
                raise RuntimeError(f"Invalid mask name: {raw_name}")
            name = Path(raw).name
            if not name.endswith(".png"):
                name = f"{name}.png"
            path = target_dir / name
            if not path.is_file():
                missing.append(name)
                continue
            path.unlink()
            for overlay in overlays.glob(f"{Path(name).stem}_overlay.*"):
                overlay.unlink(missing_ok=True)
            deleted.append(name)
        if not deleted and missing:
            raise RuntimeError(f"Mask not found: {', '.join(missing)}")
        return {
            "instance_id": instance_id,
            "deleted": deleted,
            "missing": missing,
            "mask_count": len(self.mask_files(instance_id)),
        }

    def render_overlays(self, instance_id: str, *, limit: int | None = None) -> list[dict[str, str]]:
        """Render RGB+mask overlays for every mask (or up to ``limit``)."""
        overlays_dir = self.cfg.workspace_dir / "build" / "review" / instance_id
        overlays_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.cfg.nerfstudio_data_dir / "images"
        outputs: list[dict[str, str]] = []
        masks = self.mask_files(instance_id)
        if limit is not None:
            masks = masks[: max(0, int(limit))]
        for mask_path in masks:
            image_path = next(
                (
                    candidate
                    for candidate in (
                        images_dir / f"{mask_path.stem}.jpg",
                        images_dir / f"{mask_path.stem}.jpeg",
                        images_dir / f"{mask_path.stem}.png",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if image_path is None:
                continue
            with Image.open(image_path) as image_raw, Image.open(mask_path) as mask_raw:
                image = np.asarray(image_raw.convert("RGB")).copy()
                mask = np.asarray(
                    mask_raw.convert("L").resize(
                        (image.shape[1], image.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                )
            foreground = mask > 127
            image[foreground] = (
                image[foreground].astype(np.float32) * 0.45
                + np.array([255, 40, 40], dtype=np.float32) * 0.55
            ).astype(np.uint8)
            output = overlays_dir / f"{mask_path.stem}_overlay.jpg"
            Image.fromarray(image).save(output, quality=90)
            outputs.append(
                {
                    "path": str(output),
                    "mask_name": mask_path.name,
                    "stem": mask_path.stem,
                }
            )
        return outputs

    def approve_segmentation(self, *, reviewer: str, notes: str = "") -> None:
        # Only instances with on-disk masks are reviewable. Maskless proposals stay in
        # the manifest but do not block the gate (and are hidden in the Review UI list).
        reviewable = [
            obj for obj in self.manifest.objects if self.mask_files(obj.instance_id)
        ]
        if not reviewable:
            raise RuntimeError("No masked object instances exist to approve")
        failures: list[str] = []
        for obj in reviewable:
            if obj.review_state == "rejected":
                continue
            valid_masks = len(self.mask_files(obj.instance_id))
            if obj.review_state != "approved":
                failures.append(f"{obj.instance_id}: state={obj.review_state}")
            if valid_masks < self.cfg.segmentation.min_views_per_object:
                failures.append(
                    f"{obj.instance_id}: {valid_masks} masks "
                    f"(< {self.cfg.segmentation.min_views_per_object})"
                )
            if (
                self.manifest.build_mode == "production"
                and obj.movable
                and not self.cfg.qa.allow_background_holes
                and obj.observed_background_coverage < self.cfg.qa.min_background_coverage
                and self.cfg.capture.clean_plate_dir is None
            ):
                failures.append(f"{obj.instance_id}: insufficient hidden-background coverage")
        if failures:
            raise RuntimeError("Segmentation review failed:\n- " + "\n- ".join(failures))
        self.manifest.approve("segmentation", reviewer=reviewer, notes=notes)
        self.manifest.review_state = "approved"
        self.save()

    def approve_object_asset(self, instance_id: str, *, reviewer: str, notes: str = "") -> None:
        obj = self.object(instance_id)
        required_paths = [obj.render_mesh, obj.collision_mesh, obj.baked_texture]
        if any(not value or not Path(str(value)).is_file() for value in required_paths):
            raise RuntimeError(f"{instance_id} is missing generated geometry/material files")
        required_physics = {"mass_kg", "diagonal_inertia_kg_m2", "friction", "restitution"}
        missing = sorted(required_physics - set(obj.physics))
        if missing:
            raise RuntimeError(f"{instance_id} is missing physical properties: {missing}")
        obj.physics["approved"] = True
        obj.physics["materials_approved"] = True
        obj.physics["asset_reviewer"] = reviewer
        obj.physics["asset_review_notes"] = notes
        self.manifest.upsert_object(obj)
        self.manifest.approve(f"asset:{instance_id}", reviewer=reviewer, notes=notes)
        self.save()

    def approve_lighting(self, *, reviewer: str, notes: str = "") -> None:
        from scan2usd.lighting.estimate import approve_lighting

        approve_lighting(self.manifest, reviewer=reviewer, notes=notes)
        self.save()


def launch_review_app(
    cfg: SceneConfig,
    manifest_path: Path,
    *,
    share: bool = False,
) -> Any:
    """Launch the local approval UI; processing remains in ``ReviewSession`` for testability."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Review UI requires: pip install -e ".[review]"') from exc

    session = ReviewSession(cfg, manifest_path)
    ids = session.instance_ids()
    if not ids:
        raise RuntimeError("No proposed objects in the manifest; run segmentation first")

    def load_instance(instance_id: str):
        obj = session.object(instance_id)
        overlays = [item["path"] for item in session.render_overlays(instance_id)]
        return (
            obj.class_name,
            obj.movable,
            obj.review_state,
            obj.observed_background_coverage,
            str(obj.physics.get("template", "generic")),
            str(obj.physics.get("review_notes", "")),
            overlays,
        )

    def save_instance(
        instance_id: str,
        class_name: str,
        movable: bool,
        state: str,
        coverage: float,
        template: str,
        notes: str,
        corrected_files,
    ) -> str:
        if corrected_files:
            paths = [
                Path(getattr(item, "name", item))
                for item in (corrected_files if isinstance(corrected_files, list) else [corrected_files])
            ]
            session.import_corrected_masks(instance_id, paths)
        session.update_object(
            instance_id,
            class_name=class_name,
            movable=movable,
            review_state=state,
            observed_background_coverage=coverage,
            physics_template=template,
            notes=notes,
        )
        return f"Saved {instance_id}"

    with gr.Blocks(title=f"Scan2USD review — {session.manifest.scene_name}") as app:
        gr.Markdown("# Scan2USD object and mask review")
        instance = gr.Dropdown(ids, value=ids[0], label="Instance")
        with gr.Row():
            class_name = gr.Textbox(label="Class")
            movable = gr.Checkbox(label="Movable")
            state = gr.Dropdown(
                ["pending", "approved", "rejected", "degraded"],
                label="Review state",
            )
        coverage = gr.Slider(0.0, 1.0, step=0.01, label="Observed hidden-background coverage")
        template = gr.Textbox(label="Physics template")
        notes = gr.Textbox(label="Notes")
        corrected = gr.File(file_count="multiple", label="Corrected foreground-white masks")
        gallery = gr.Gallery(label="Mask overlays", columns=3)
        save_button = gr.Button("Save instance")
        save_status = gr.Textbox(label="Status", interactive=False)

        instance.change(
            load_instance,
            inputs=instance,
            outputs=[class_name, movable, state, coverage, template, notes, gallery],
        )
        save_button.click(
            save_instance,
            inputs=[
                instance,
                class_name,
                movable,
                state,
                coverage,
                template,
                notes,
                corrected,
            ],
            outputs=save_status,
        )
        app.load(
            load_instance,
            inputs=instance,
            outputs=[class_name, movable, state, coverage, template, notes, gallery],
        )
    return app.launch(share=share)
