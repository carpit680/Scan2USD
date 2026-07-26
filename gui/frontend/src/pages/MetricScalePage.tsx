import { useCallback, useEffect, useState } from "react";
import { client } from "../api/client";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { SparseCloudPicker } from "../components/SparseCloudPicker";
import { Tip } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

interface MetricScene {
  has_ply: boolean;
  has_floor: boolean;
  ready: boolean;
  ply_url: string | null;
  floor_matrix: number[][] | null;
  meters_approved: boolean;
  existing_scale: number | null;
  scale_method: string | null;
}

export function MetricScalePage() {
  const { loaded, runCommand, setError, activeJob } = useProject();
  const [scene, setScene] = useState<MetricScene | null>(null);
  const [sourceLength, setSourceLength] = useState<number | null>(null);
  const [knownLengthM, setKnownLengthM] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [pickerKey, setPickerKey] = useState(0);
  const [pendingApply, setPendingApply] = useState(false);
  const [status, setStatus] = useState("");

  const refresh = useCallback(async () => {
    try {
      const data = (await client.metricScene()) as unknown as MetricScene;
      setScene(data);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, [setError]);

  useEffect(() => {
    if (!loaded) return;
    void refresh();
  }, [loaded, refresh]);

  useEffect(() => {
    if (!activeJob) return;
    if (
      activeJob.command === "apply-metric-scale" &&
      (activeJob.status === "succeeded" ||
        activeJob.status === "failed" ||
        activeJob.status === "cancelled")
    ) {
      void refresh();
      if (activeJob.status === "succeeded") {
        setStatus("Metric scale applied. Rebuild baked geometry and package-usd when ready.");
      }
    }
  }, [activeJob?.status, activeJob?.id, activeJob?.command, refresh]);

  if (!loaded) {
    return <p className="text-ink-400">Open a project first.</p>;
  }

  const metersPerUnit =
    sourceLength && sourceLength > 0 && Number(knownLengthM) > 0
      ? Number(knownLengthM) / sourceLength
      : null;

  function onResetPicks() {
    setSourceLength(null);
    setPickerKey((k) => k + 1);
  }

  function onApplyClick() {
    if (!sourceLength || sourceLength <= 0) {
      setError("Pick two points on an edge first.");
      return;
    }
    if (!knownLengthM || Number(knownLengthM) <= 0) {
      setError("Enter the real-world length in meters.");
      return;
    }
    if (!reviewer.trim()) {
      setError("Enter a reviewer name.");
      return;
    }
    setPendingApply(true);
  }

  async function confirmApply() {
    setPendingApply(false);
    if (!sourceLength) return;
    try {
      setStatus("Starting apply-metric-scale…");
      await runCommand("apply-metric-scale", {
        reviewer: reviewer.trim(),
        known_length_m: Number(knownLengthM),
        source_length: sourceLength,
      });
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  return (
    <div className="space-y-6">
      <ConfirmDialog
        open={pendingApply}
        title="Apply metric scale"
        message={
          sourceLength && Number(knownLengthM) > 0
            ? `Approve scale using edge ${sourceLength.toFixed(4)} scene units = ${knownLengthM} m` +
              (metersPerUnit != null
                ? ` (${metersPerUnit.toFixed(6)} m/unit)? This updates the COLMAP→USD transform.`
                : "?")
            : ""
        }
        confirmLabel="Apply scale"
        onCancel={() => setPendingApply(false)}
        onConfirm={() => void confirmApply()}
      />

      <div>
        <h1 className="font-display text-3xl font-semibold">Metric scale</h1>
        <p className="mt-1 text-ink-400">
          Click two points on a known edge in the floor-aligned sparse cloud, enter its length in
          meters, then apply scale for production physics.
        </p>
      </div>

      {!scene ? (
        <p className="text-ink-500">Loading scene…</p>
      ) : !scene.ready ? (
        <div className="rounded-xl border border-dashed border-ink-700 p-5 text-sm text-ink-400">
          <p className="font-medium text-ink-200">Scene not ready for measuring</p>
          <ul className="mt-2 list-inside list-disc space-y-1">
            {!scene.has_ply ? (
              <li>
                Missing <span className="font-mono text-ink-300">sparse_pc.ply</span> — run{" "}
                <span className="text-ink-200">reconstruct</span> first.
              </li>
            ) : null}
            {!scene.has_floor ? (
              <li>
                Missing floor transform — run <span className="text-ink-200">align-floor</span>{" "}
                first.
              </li>
            ) : null}
          </ul>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
          <div className="space-y-2">
            <SparseCloudPicker
              key={pickerKey}
              plyUrl={scene.ply_url!}
              floorMatrix={scene.floor_matrix!}
              onEdgeChange={(_pts, len) => setSourceLength(len)}
            />
            <p className="text-xs text-ink-500">
              Drag to orbit. Click once for each endpoint of an edge (desk width, door, etc.).
            </p>
          </div>

          <div className="space-y-4 rounded-xl border border-ink-800 bg-ink-900/40 p-4">
            <div>
              <div className="text-xs uppercase tracking-wide text-ink-500">Edge in scene units</div>
              <div className="mt-1 font-mono text-lg text-ink-100">
                {sourceLength != null ? sourceLength.toFixed(6) : "—"}
              </div>
            </div>

            <label className="block space-y-1 text-sm">
              <span className="flex items-center text-ink-300">
                Real length (m)
                <Tip text="Tape-measure the same edge in the physical room." anchor="metric" />
              </span>
              <input
                type="number"
                min={0}
                step="any"
                className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2"
                value={knownLengthM}
                onChange={(e) => setKnownLengthM(e.target.value)}
                placeholder="e.g. 0.91"
              />
            </label>

            {metersPerUnit != null ? (
              <div className="text-xs text-ink-400">
                Implied scale:{" "}
                <span className="font-mono text-ink-200">{metersPerUnit.toFixed(6)}</span> m/unit
              </div>
            ) : null}

            <label className="block space-y-1 text-sm">
              <span className="text-ink-300">Reviewer</span>
              <input
                className="w-full rounded-md border border-ink-700 bg-ink-950 px-3 py-2"
                value={reviewer}
                onChange={(e) => setReviewer(e.target.value)}
                placeholder="Your name"
              />
            </label>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={onResetPicks}
                className="rounded-md border border-ink-600 px-3 py-1.5 text-sm text-ink-200 hover:border-ink-400"
              >
                Reset picks
              </button>
              <button
                type="button"
                onClick={onApplyClick}
                disabled={
                  sourceLength == null ||
                  !knownLengthM ||
                  !reviewer.trim() ||
                  activeJob?.status === "running"
                }
                className="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-ink-950 disabled:opacity-40"
              >
                Apply scale
              </button>
            </div>

            {scene.meters_approved ? (
              <p className="text-xs text-ok">
                Metric scale already approved
                {scene.existing_scale != null
                  ? ` (${scene.existing_scale.toFixed(6)} m/unit)`
                  : ""}
                {scene.scale_method ? ` · ${scene.scale_method}` : ""}. Applying again will overwrite.
              </p>
            ) : (
              <p className="text-xs text-ink-500">Metric scale not approved yet.</p>
            )}

            {status ? <p className="text-sm text-ok">{status}</p> : null}
          </div>
        </div>
      )}
    </div>
  );
}
