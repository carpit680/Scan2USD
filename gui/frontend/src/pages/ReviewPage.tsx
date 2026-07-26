import { useEffect, useState } from "react";
import { client, overlayUrl } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Tip } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

interface InstanceRow {
  instance_id: string;
  class_name: string;
  movable: boolean;
  review_state: string;
  observed_background_coverage: number;
  physics_template: string;
  notes: string;
  mask_count: number;
  merged_into?: string | null;
  merged_from?: string[];
}

interface OverlayItem {
  path: string;
  mask_name: string;
  stem: string;
}

function reviewStateStyle(state: string): { row: string; badge: string } {
  switch (state) {
    case "approved":
      return {
        row: "border-l-ok bg-ok/5 hover:bg-ok/10",
        badge: "text-ok",
      };
    case "rejected":
      return {
        row: "border-l-danger bg-danger/5 hover:bg-danger/10",
        badge: "text-danger",
      };
    case "pending":
      return {
        row: "border-l-warn bg-warn/5 hover:bg-warn/10",
        badge: "text-warn",
      };
    case "degraded":
      return {
        row: "border-l-orange-400 bg-orange-400/5 hover:bg-orange-400/10",
        badge: "text-orange-400",
      };
    default:
      return {
        row: "border-l-ink-500 bg-ink-900/40 hover:bg-ink-900",
        badge: "text-ink-400",
      };
  }
}

export function ReviewPage() {
  const { loaded, setError } = useProject();
  const [instances, setInstances] = useState<InstanceRow[]>([]);
  const [totalInstances, setTotalInstances] = useState(0);
  const [mergedAway, setMergedAway] = useState(0);
  const [selected, setSelected] = useState<string>("");
  const [className, setClassName] = useState("");
  const [movable, setMovable] = useState(true);
  const [reviewState, setReviewState] = useState("pending");
  const [coverage, setCoverage] = useState(0);
  const [template, setTemplate] = useState("generic");
  const [notes, setNotes] = useState("");
  const [overlays, setOverlays] = useState<OverlayItem[]>([]);
  const [overlayEpoch, setOverlayEpoch] = useState(0);
  const [status, setStatus] = useState("");
  const [mergeIds, setMergeIds] = useState<string[]>([]);
  const [mergedFrom, setMergedFrom] = useState<string[]>([]);
  const [maskCount, setMaskCount] = useState(0);
  const [pendingDeleteMasks, setPendingDeleteMasks] = useState<string[] | null>(null);
  const [selectedMasks, setSelectedMasks] = useState<string[]>([]);
  const [pendingMerge, setPendingMerge] = useState(false);
  const [pendingUnmerge, setPendingUnmerge] = useState<string[] | "all" | null>(null);
  const [pendingSave, setPendingSave] = useState<{
    reclassify: boolean;
    newInstanceId?: string;
  } | null>(null);

  async function refreshList() {
    const data = await client.reviewInstances();
    const all = (data.instances as InstanceRow[]) || [];
    const rows = all.filter(
      (r) => Number(r.mask_count) > 0 && !r.merged_into,
    );
    setInstances(rows);
    setTotalInstances(Number(data.total_instances ?? all.length));
    setMergedAway(Number(data.merged_away ?? all.filter((r) => r.merged_into).length));
    setMergeIds((prev) => prev.filter((id) => rows.some((r) => r.instance_id === id)));
    setSelected((prev) => {
      if (prev && rows.some((r) => r.instance_id === prev)) return prev;
      return rows[0]?.instance_id || "";
    });
    return rows;
  }

  function parseOverlays(raw: unknown): OverlayItem[] {
    if (!Array.isArray(raw)) return [];
    return raw
      .map((item) => {
        if (typeof item === "string") {
          const base = item.split("/").pop() || item;
          const stem = base.replace(/_overlay\.[^.]+$/, "");
          return { path: item, mask_name: `${stem}.png`, stem };
        }
        if (item && typeof item === "object") {
          const o = item as Record<string, unknown>;
          const path = String(o.path || "");
          const mask_name = String(o.mask_name || "");
          const stem = String(o.stem || mask_name.replace(/\.png$/i, ""));
          if (!path) return null;
          return { path, mask_name: mask_name || `${stem}.png`, stem };
        }
        return null;
      })
      .filter((x): x is OverlayItem => Boolean(x));
  }

  async function loadOne(id: string) {
    const data = await client.reviewInstance(id);
    setClassName(String(data.class_name || ""));
    setMovable(Boolean(data.movable));
    setReviewState(String(data.review_state || "pending"));
    setCoverage(Number(data.observed_background_coverage || 0));
    setTemplate(String(data.physics_template || "generic"));
    setNotes(String(data.notes || ""));
    setOverlays(parseOverlays(data.overlays));
    setOverlayEpoch(Date.now());
    setSelectedMasks([]);
    const count = Number(
      data.mask_count ?? (data.mask_files as string[] | undefined)?.length ?? 0,
    );
    setMaskCount(count);
    setInstances((prev) =>
      prev.map((row) =>
        row.instance_id === id ? { ...row, mask_count: count } : row,
      ),
    );
    setMergedFrom(
      Array.isArray(data.merged_from) ? (data.merged_from as string[]) : [],
    );
  }

  useEffect(() => {
    if (!loaded) return;
    refreshList().catch((e) => setError(String(e.message || e)));
  }, [loaded]);

  useEffect(() => {
    if (!selected) return;
    loadOne(selected).catch((e) => setError(String(e.message || e)));
  }, [selected]);

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  async function onSave() {
    if (!selected) return;
    try {
      const preview = await client.reclassifyPreview(selected, className);
      if (preview.will_rename) {
        setPendingSave({
          reclassify: true,
          newInstanceId: String(preview.new_instance_id || ""),
        });
        return;
      }
      await confirmSave(true);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  async function confirmSave(reclassify: boolean) {
    if (!selected) return;
    setPendingSave(null);
    try {
      const res = await client.updateInstance(selected, {
        class_name: className,
        movable,
        review_state: reviewState,
        observed_background_coverage: coverage,
        physics_template: template,
        notes,
        reclassify,
      });
      const newId = String(res.instance_id || selected);
      if (res.renamed && newId !== selected) {
        setStatus(`Saved and renamed ${selected} → ${newId}`);
        setSelected(newId);
      } else {
        setStatus(`Saved ${newId}`);
      }
      await refreshList();
      await loadOne(newId);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  async function onUpload(files: FileList | null) {
    if (!files || !files.length) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    try {
      const res = await fetch(`/api/review/instances/${selected}/masks`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      const body = await res.json();
      setStatus(`Imported ${body.imported} masks`);
      await loadOne(selected);
      await refreshList();
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  function toggleMergeId(id: string) {
    setMergeIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  }

  async function onMerge() {
    if (!selected || mergeIds.length === 0) return;
    setPendingMerge(true);
  }

  async function confirmMerge() {
    if (!selected || mergeIds.length === 0) return;
    setPendingMerge(false);
    try {
      const res = await client.mergeInstances(selected, mergeIds);
      setMergeIds([]);
      const before = Number(res.mask_count_before ?? maskCount);
      const after = Number(res.mask_count ?? before);
      setMaskCount(after);
      setInstances((prev) =>
        prev.map((row) =>
          row.instance_id === selected ? { ...row, mask_count: after } : row,
        ),
      );
      const identical = Number(res.frames_identical ?? 0);
      const added = Number(res.frames_added ?? 0);
      const unioned = Number(res.frames_replaced ?? 0);
      const countNote =
        before === after
          ? " File count unchanged — sources only covered frames the primary already had."
          : ` File count rose by ${after - before} (new frame names from sources).`;
      setStatus(
        `Merged ${String(res.merged)} → ${selected}: masks ${before} → ${after}` +
          ` (+${added} frames copied, ${unioned} OR-unioned` +
          (identical ? `, ${identical} already identical` : "") +
          `).${countNote}`,
      );
      await refreshList();
      await loadOne(selected);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  async function onUnmerge(sourceIds?: string[]) {
    if (!selected) return;
    setPendingUnmerge(sourceIds?.length ? sourceIds : "all");
  }

  async function confirmUnmerge() {
    if (!selected || pendingUnmerge == null) return;
    const sourceIds = pendingUnmerge === "all" ? undefined : pendingUnmerge;
    setPendingUnmerge(null);
    try {
      const res = await client.unmergeInstances(selected, sourceIds);
      const warnings = Array.isArray(res.warnings) ? (res.warnings as string[]) : [];
      const before = Number(res.mask_count_before ?? maskCount);
      const after = Number(res.mask_count ?? before);
      setMaskCount(after);
      setInstances((prev) =>
        prev.map((row) =>
          row.instance_id === selected ? { ...row, mask_count: after } : row,
        ),
      );
      setStatus(
        `Restored ${String(res.restored)} from ${selected}: masks ${before} → ${after}` +
          ` (−${res.frames_removed ?? 0} frames, ${res.frames_restored ?? 0} restored)` +
          (warnings.length ? ` (warnings: ${warnings.join("; ")})` : ""),
      );
      await refreshList();
      await loadOne(selected);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  function toggleMaskSelect(maskName: string) {
    setSelectedMasks((prev) =>
      prev.includes(maskName) ? prev.filter((n) => n !== maskName) : [...prev, maskName],
    );
  }

  function selectAllMasks() {
    setSelectedMasks(overlays.map((o) => o.mask_name));
  }

  function clearMaskSelection() {
    setSelectedMasks([]);
  }

  async function onDeleteMask(maskName: string) {
    if (!selected || !maskName) return;
    setPendingDeleteMasks([maskName]);
  }

  async function onDeleteSelectedMasks() {
    if (!selected || selectedMasks.length === 0) return;
    setPendingDeleteMasks([...selectedMasks]);
  }

  async function confirmDeleteMasks() {
    const names = pendingDeleteMasks;
    if (!selected || !names?.length) return;
    setPendingDeleteMasks(null);
    try {
      const res = await client.deleteMasks(selected, names);
      const deleted = Array.isArray(res.deleted) ? (res.deleted as string[]) : [];
      const after = Number(res.mask_count ?? Math.max(0, maskCount - deleted.length));
      setMaskCount(after);
      setSelectedMasks([]);
      setStatus(
        `Removed ${deleted.length} mask${deleted.length === 1 ? "" : "s"} → ${after} remaining`,
      );
      if (after <= 0) {
        await refreshList();
      } else {
        setInstances((prev) =>
          prev.map((row) =>
            row.instance_id === selected ? { ...row, mask_count: after } : row,
          ),
        );
        await loadOne(selected);
      }
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  return (
    <div className="space-y-6">
      <ConfirmDialog
        open={pendingDeleteMasks != null && pendingDeleteMasks.length > 0}
        title={pendingDeleteMasks?.length === 1 ? "Remove mask" : "Remove masks"}
        message={
          pendingDeleteMasks?.length === 1
            ? `Remove ${pendingDeleteMasks[0]} from ${selected}? This deletes the mask file from disk.`
            : pendingDeleteMasks?.length
              ? `Remove ${pendingDeleteMasks.length} selected masks from ${selected}? This deletes the mask files from disk.`
              : ""
        }
        confirmLabel="Remove"
        danger
        onCancel={() => setPendingDeleteMasks(null)}
        onConfirm={() => void confirmDeleteMasks()}
      />
      <ConfirmDialog
        open={pendingMerge}
        title="Merge instances"
        message={
          selected && mergeIds.length
            ? `Merge ${mergeIds.join(", ")} into ${selected}? Masks are copied into the primary; sources become rejected. Your primary review state is kept.`
            : ""
        }
        confirmLabel="Merge"
        onCancel={() => setPendingMerge(false)}
        onConfirm={() => void confirmMerge()}
      />
      <ConfirmDialog
        open={pendingUnmerge != null}
        title="Revert merge"
        message={
          pendingUnmerge == null
            ? ""
            : pendingUnmerge === "all"
              ? `Revert all merges into ${selected}? Sources return to the list with their prior review state. Primary masks are rolled back when a merge journal exists.`
              : `Restore ${pendingUnmerge.join(", ")} from ${selected}? Sources return to the list with their prior review state. Primary masks are rolled back when a merge journal exists.`
        }
        confirmLabel="Revert"
        danger
        onCancel={() => setPendingUnmerge(null)}
        onConfirm={() => void confirmUnmerge()}
      />
      <ConfirmDialog
        open={pendingSave != null}
        title="Rename instance?"
        message={
          pendingSave?.reclassify && pendingSave.newInstanceId
            ? `Class changed. Rename ${selected} → ${pendingSave.newInstanceId}? Masks and review files move with the new id (next free index for that class).`
            : ""
        }
        confirmLabel="Rename & save"
        cancelLabel="Cancel"
        onCancel={() => setPendingSave(null)}
        onConfirm={() => void confirmSave(true)}
      />
      <div>
        <h1 className="font-display text-3xl font-semibold">Review</h1>
        <p className="mt-1 text-ink-400">
          Set each keeper’s status to <span className="text-ink-200">approved</span> (dropdown).
          Only instances with masks are listed. Check duplicates and use{" "}
          <span className="text-ink-200">Merge into selected</span> when they are the same physical
          object.
        </p>
      </div>

      {instances.length === 0 ? (
        <p className="rounded-lg border border-dashed border-ink-700 p-4 text-sm text-ink-500">
          No masked instances yet
          {totalInstances > 0 ? ` (${totalInstances} proposals without masks).` : "."} Run
          segment-usd (with masks), then refresh.
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
          <div className="space-y-1 rounded-xl border border-ink-800 bg-ink-900/40 p-2">
            <div className="px-2 pb-1 text-[11px] text-ink-500">
              {instances.length} with masks
              {mergedAway > 0 ? ` · ${mergedAway} merged away` : null}
              {totalInstances > instances.length + mergedAway
                ? ` · ${totalInstances - instances.length - mergedAway} hidden`
                : null}
              {mergeIds.length > 0 ? (
                <span className="ml-1 text-accent"> · {mergeIds.length} to merge</span>
              ) : null}
            </div>
            {instances.map((i) => {
              const style = reviewStateStyle(i.review_state);
              const isPrimary = selected === i.instance_id;
              const checked = mergeIds.includes(i.instance_id);
              return (
                <div
                  key={i.instance_id}
                  className={`flex items-stretch gap-1 rounded-md border-l-4 ${style.row} ${
                    isPrimary ? "ring-1 ring-accent/50" : ""
                  }`}
                >
                  <label
                    className="flex items-center px-2"
                    title={
                      isPrimary
                        ? "Primary (merge target) — uncheck others into this"
                        : "Mark as duplicate to merge into the selected primary"
                    }
                  >
                    <input
                      type="checkbox"
                      className="rounded border-ink-600"
                      disabled={isPrimary}
                      checked={checked}
                      onChange={() => toggleMergeId(i.instance_id)}
                    />
                  </label>
                  <button
                    type="button"
                    onClick={() => {
                      setSelected(i.instance_id);
                      setMergeIds((prev) => prev.filter((id) => id !== i.instance_id));
                    }}
                    className="min-w-0 flex-1 px-2 py-2 text-left text-sm"
                  >
                    <div className="font-mono text-xs text-ink-100">
                      {i.instance_id}
                      {isPrimary ? (
                        <span className="ml-1 text-[10px] uppercase tracking-wide text-accent">
                          primary
                        </span>
                      ) : null}
                    </div>
                    <div className="text-ink-400">
                      {i.class_name} ·{" "}
                      <span className={`font-medium ${style.badge}`}>{i.review_state}</span> ·{" "}
                      {i.mask_count} masks
                    </div>
                  </button>
                </div>
              );
            })}
          </div>

          <div className="space-y-4 rounded-xl border border-ink-800 bg-ink-900/40 p-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="font-mono text-sm text-ink-100">{selected}</h2>
              <p className="text-sm text-ink-400">
                <span className="font-medium text-ink-200">{maskCount}</span> masks on disk
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="space-y-1 text-sm">
                <span className="flex items-center text-ink-300">
                  Class
                  <Tip text="Semantic class. Saving a different class renames the instance to class_NNN (next free index) and moves its masks." anchor="segmentation" />
                </span>
                <input
                  className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2"
                  value={className}
                  onChange={(e) => setClassName(e.target.value)}
                />
              </label>
              <label className="space-y-1 text-sm">
                <span className="flex items-center text-ink-300">
                  Review state
                  <Tip text="pending / approved / rejected / degraded. Mark keepers approved; reject bad masks. Only approved objects are built into the USD package." />
                </span>
                <select
                  className={`w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2 ${
                    reviewStateStyle(reviewState).badge
                  }`}
                  value={reviewState}
                  onChange={(e) => setReviewState(e.target.value)}
                >
                  {["pending", "approved", "rejected", "degraded"].map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-ink-300">
                <input type="checkbox" checked={movable} onChange={(e) => setMovable(e.target.checked)} />
                Movable
                <Tip text="Movable objects get separate rigid meshes and physics." />
              </label>
              <label className="space-y-1 text-sm sm:col-span-2">
                <span className="flex items-center text-ink-300">
                  Observed hidden-background coverage ({coverage.toFixed(2)})
                  <Tip text="Fraction of the object footprint seen without the object (clean plate / multi-pass). Production blocks low coverage unless Config → QA → Allow background holes is on (dev holes OK)." anchor="segmentation" />
                </span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={coverage}
                  onChange={(e) => setCoverage(parseFloat(e.target.value))}
                  className="w-full accent-accent"
                />
              </label>
              <label className="space-y-1 text-sm">
                <span className="text-ink-300">Physics template</span>
                <input
                  className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2"
                  value={template}
                  onChange={(e) => setTemplate(e.target.value)}
                />
              </label>
              <label className="space-y-1 text-sm">
                <span className="text-ink-300">Notes</span>
                <input
                  className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </label>
              <label className="space-y-1 text-sm sm:col-span-2">
                <span className="flex items-center text-ink-300">
                  Corrected masks
                  <Tip text="Upload foreground-white PNG masks to replace instance masks." />
                </span>
                <input type="file" multiple accept="image/png,image/*" onChange={(e) => onUpload(e.target.files)} />
              </label>
            </div>

            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={onSave} className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-ink-950">
                Save instance
              </button>
              <button
                type="button"
                disabled={mergeIds.length === 0 || !selected}
                onClick={() => void onMerge()}
                className="rounded-md border border-accent/60 px-4 py-2 text-sm font-semibold text-accent hover:bg-accent/10 disabled:opacity-40"
                title="Copy masks from checked duplicates into the selected primary"
              >
                Merge {mergeIds.length || ""} into selected
              </button>
              {mergedFrom.length > 0 ? (
                <button
                  type="button"
                  onClick={() => void onUnmerge()}
                  className="rounded-md border border-warn/60 px-4 py-2 text-sm font-semibold text-warn hover:bg-warn/10"
                  title="Restore all instances previously merged into this primary"
                >
                  Revert all merges ({mergedFrom.length})
                </button>
              ) : null}
            </div>
            {mergeIds.length > 0 ? (
              <p className="text-xs text-ink-500">
                Will merge <span className="font-mono text-ink-300">{mergeIds.join(", ")}</span> →{" "}
                <span className="font-mono text-accent">{selected}</span>
              </p>
            ) : null}
            {mergedFrom.length > 0 ? (
              <div className="rounded-md border border-ink-800 bg-ink-950/50 p-3 text-xs text-ink-400">
                <div className="mb-2 text-ink-300">Merged into this instance (click to restore one):</div>
                <div className="flex flex-wrap gap-2">
                  {mergedFrom.map((id) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() => void onUnmerge([id])}
                      className="rounded border border-ink-700 px-2 py-1 font-mono text-[11px] text-ink-200 hover:border-warn hover:text-warn"
                    >
                      Restore {id}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-ink-500">
                {selectedMasks.length > 0
                  ? `${selectedMasks.length} selected`
                  : "Select masks to batch-delete"}
              </span>
              <button
                type="button"
                disabled={overlays.length === 0}
                onClick={selectAllMasks}
                className="rounded border border-ink-700 px-2 py-0.5 text-[11px] text-ink-300 hover:border-ink-500 disabled:opacity-40"
              >
                Select all
              </button>
              <button
                type="button"
                disabled={selectedMasks.length === 0}
                onClick={clearMaskSelection}
                className="rounded border border-ink-700 px-2 py-0.5 text-[11px] text-ink-300 hover:border-ink-500 disabled:opacity-40"
              >
                Clear
              </button>
              <button
                type="button"
                disabled={selectedMasks.length === 0}
                onClick={() => void onDeleteSelectedMasks()}
                className="rounded border border-danger/60 px-2 py-0.5 text-[11px] font-medium text-danger hover:bg-danger/10 disabled:opacity-40"
              >
                Delete selected ({selectedMasks.length || 0})
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
              {overlays.map((o) => {
                const checked = selectedMasks.includes(o.mask_name);
                return (
                  <div
                    key={o.mask_name}
                    className={`relative ${checked ? "ring-2 ring-accent" : ""}`}
                  >
                    <button
                      type="button"
                      onClick={() => toggleMaskSelect(o.mask_name)}
                      className="block w-full text-left"
                      title={checked ? "Deselect" : "Select for batch delete"}
                    >
                      <img
                        src={overlayUrl(o.path, overlayEpoch)}
                        alt={o.stem}
                        className="w-full rounded-md border border-ink-700 object-cover"
                      />
                    </button>
                    <label
                      className="absolute left-1.5 top-1.5 flex h-6 w-6 cursor-pointer items-center justify-center rounded bg-ink-950/80"
                      title="Select for batch delete"
                    >
                      <input
                        type="checkbox"
                        className="rounded border-ink-600"
                        checked={checked}
                        onChange={() => toggleMaskSelect(o.mask_name)}
                      />
                    </label>
                    <button
                      type="button"
                      title={`Remove ${o.mask_name}`}
                      onClick={() => void onDeleteMask(o.mask_name)}
                      className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded bg-ink-950/80 text-sm leading-none text-ink-200 hover:bg-danger hover:text-white"
                    >
                      ×
                    </button>
                    <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-ink-950/70 px-1 font-mono text-[10px] text-ink-300">
                      {o.stem}
                    </span>
                  </div>
                );
              })}
              {overlays.length === 0 ? (
                <p className="col-span-full text-sm text-ink-500">No mask overlays yet.</p>
              ) : null}
            </div>
            {overlays.length > 0 && overlays.length < maskCount ? (
              <p className="text-xs text-warn">
                Showing {overlays.length} of {maskCount} masks (missing matching RGB frames for the
                rest).
              </p>
            ) : null}

            {status ? <p className="text-sm text-ok">{status}</p> : null}
          </div>
        </div>
      )}
    </div>
  );
}
