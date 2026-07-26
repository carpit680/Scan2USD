import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { client, type StageDef } from "../api/client";
import { useProject } from "../state/ProjectContext";
import { Tip } from "../components/ParamField";

export function PipelinePage() {
  const navigate = useNavigate();
  const { loaded, runCommand, cancelJob, setError, schema, activeJob } = useProject();
  const [state, setState] = useState<Record<string, unknown> | null>(null);
  const [mode, setMode] = useState("production");
  // Re-run completed stages (segment / align-floor / cleanup / build). Off by default.
  const [force, setForce] = useState(false);
  // Hybrid USD path trains visuals later via 3DGRUT — skip Nerfstudio train by default.
  const [skipTrain, setSkipTrain] = useState(true);
  const [videoStride, setVideoStride] = useState(15);
  const [videoMaxFrames, setVideoMaxFrames] = useState(600);

  function commandSupportsForce(command: string): boolean {
    const cmd = schema?.commands?.find((c) => c.id === command);
    if (cmd?.options.some((o) => o.config_path === "force")) return true;
    // Fallback if schema not loaded yet / stale API
    return ["reconstruct", "segment-usd", "align-floor", "cleanup-splat", "build-usd"].includes(command);
  }

  async function refresh() {
    try {
      setState(await client.pipelineState());
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  useEffect(() => {
    if (loaded) refresh();
  }, [loaded]);

  // Refresh pipeline badges when a job finishes
  useEffect(() => {
    if (!activeJob) return;
    if (activeJob.status === "succeeded" || activeJob.status === "failed" || activeJob.status === "cancelled") {
      refresh();
    }
  }, [activeJob?.status, activeJob?.id]);

  if (!loaded) {
    return <p className="text-ink-400">Open a project first.</p>;
  }

  const stages = (state?.pipeline_stages as StageDef[]) || schema?.pipeline_stages || [];
  const legacy = (state?.legacy_stages as { id: string; label: string; command: string }[]) || [];
  const done = new Set((state?.stages_done as string[]) || []);
  const manifest = state?.manifest as Record<string, unknown> | null;
  const objects = (state?.objects as Record<string, unknown>[]) || [];
  const jobRunning = activeJob?.status === "running" || activeJob?.status === "queued";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-semibold">Pipeline</h1>
          <p className="mt-1 text-ink-400">
            Follow these steps in order. Live command output streams in the console at the bottom of
            the page. If Build stops with exit code 2, mark keepers approved in Review and run Build
            again.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-ink-300">
            Mode
            <Tip text="production enforces review gates; preview is looser for iteration." anchor="hybrid" />
            <select
              className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              <option value="production">production</option>
              <option value="preview">preview</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-ink-300">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
              className="rounded border-ink-600"
              disabled={jobRunning}
            />
            Force
            <Tip text="When on, steps that support --force re-run work (reconstruct re-extracts frames; build rebuilds completed stages). Leave off to resume safely." />
          </label>
          <button type="button" onClick={refresh} className="rounded-md border border-ink-600 px-3 py-1.5 text-sm">
            Refresh
          </button>
        </div>
      </div>

      {jobRunning && activeJob ? (
        <div className="rounded-lg border border-accent/40 bg-accent/5 px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <div>
              <span className="font-medium text-accent">Running</span>{" "}
              <code className="font-mono text-ink-200">{activeJob.command}</code>
              <span className="ml-2 text-ink-500">see console below</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-2 text-xs text-ink-400">
                <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
                {activeJob.status}
              </span>
              <button
                type="button"
                onClick={() => void cancelJob()}
                className="rounded-md border border-danger/60 bg-danger/15 px-3 py-1.5 text-sm font-semibold text-danger hover:bg-danger/25"
              >
                Stop
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {manifest ? (
        <div className="rounded-lg border border-ink-800 bg-ink-900/40 px-4 py-3 text-sm text-ink-300">
          Manifest <span className="text-accent">{String(manifest.scene_name)}</span> · mode{" "}
          {String(manifest.build_mode)} · objects {String(manifest.object_count)} · review{" "}
          {String(manifest.review_state)}
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-ink-700 px-4 py-3 text-sm text-ink-500">
          No scene_manifest.json yet — run Init USD after reconstruct.
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        {stages.map((stage, i) => {
          const complete = done.has(stage.id) || (stage.command ? done.has(stage.command) : false);
          const isActive = Boolean(jobRunning && activeJob && stage.command === activeJob.command);
          return (
            <div
              key={stage.id}
              className={`flex flex-col rounded-xl border p-4 ${
                isActive
                  ? "border-accent/50 bg-accent/5"
                  : complete
                    ? "border-ok/40 bg-ok/5"
                    : "border-ink-800 bg-ink-900/50"
              }`}
            >
              <div className="text-[11px] uppercase tracking-wider text-ink-500">
                Step {i + 1}
                {isActive ? <span className="ml-2 text-accent">running</span> : null}
                {complete && !isActive ? <span className="ml-2 text-ok">done</span> : null}
              </div>
              <div className="mt-1 flex items-center gap-1 font-display text-lg font-semibold">
                {stage.label}
                {stage.guide_anchor ? <Tip text={stage.description} anchor={stage.guide_anchor} /> : null}
              </div>
              <p className="mt-2 flex-1 text-sm text-ink-400">{stage.description}</p>
              {stage.command === "reconstruct" ? (
                <div className="mt-3 space-y-2 rounded-md border border-ink-800/80 bg-ink-950/50 p-2.5 text-xs text-ink-300">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={skipTrain}
                      onChange={(e) => setSkipTrain(e.target.checked)}
                      className="rounded border-ink-600"
                      disabled={jobRunning}
                    />
                    Skip train
                    <Tip text="Only ns-process-data + COLMAP export. Recommended for hybrid 3DGRUT (skips Nerfstudio splatfacto)." />
                  </label>
                  <div className="flex flex-wrap gap-3">
                    <label className="flex items-center gap-1.5">
                      Stride
                      <Tip text="When extracting from video (empty frames folder, or Force): keep every Nth frame. Higher = faster COLMAP." />
                      <input
                        type="number"
                        min={1}
                        max={120}
                        value={videoStride}
                        disabled={jobRunning}
                        onChange={(e) => setVideoStride(Number(e.target.value) || 15)}
                        className="w-14 rounded border border-ink-700 bg-ink-900 px-1.5 py-0.5"
                      />
                    </label>
                    <label className="flex items-center gap-1.5">
                      Max frames
                      <Tip text="Cap frames extracted from video (0 = no cap)." />
                      <input
                        type="number"
                        min={0}
                        max={5000}
                        value={videoMaxFrames}
                        disabled={jobRunning}
                        onChange={(e) => setVideoMaxFrames(Number(e.target.value) || 0)}
                        className="w-16 rounded border border-ink-700 bg-ink-900 px-1.5 py-0.5"
                      />
                    </label>
                  </div>
                </div>
              ) : null}
              <div className="mt-4 flex flex-wrap gap-2">
                {stage.href ? (
                  <Link
                    to={stage.href}
                    className="inline-block rounded-md bg-ink-800 px-3 py-1.5 text-sm text-accent hover:bg-ink-700"
                  >
                    Open review
                  </Link>
                ) : stage.command ? (
                  <button
                    type="button"
                    disabled={jobRunning}
                    className="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-ink-950 hover:bg-accent-dim disabled:opacity-40"
                    onClick={() => {
                      const opts: Record<string, unknown> = {};
                      if (
                        ["init-usd", "segment-usd", "build-usd", "align-floor", "cleanup-splat"].includes(
                          stage.command!,
                        )
                      ) {
                        opts.mode = mode;
                      }
                      if (stage.command === "reconstruct") {
                        opts.skip_train = skipTrain;
                        opts.video_stride = videoStride;
                        opts.video_max_frames = videoMaxFrames;
                      }
                      if (commandSupportsForce(stage.command!)) {
                        opts.force = force;
                      }
                      if (stage.command === "apply-metric-scale") {
                        navigate("/metric");
                        return;
                      }
                      runCommand(stage.command!, opts);
                    }}
                  >
                    {(() => {
                      if (isActive) return "Running…";
                      const cmd = stage.command!;
                      const supportsForce = commandSupportsForce(cmd);
                      if (force && supportsForce) {
                        return complete ? `Force re-run ${cmd}` : `Force run ${cmd}`;
                      }
                      // "Resume" only for commands that skip completed work when force is off.
                      if (complete && supportsForce) return `Resume ${cmd}`;
                      if (complete) return `Re-run ${cmd}`;
                      return `Run ${cmd}`;
                    })()}
                  </button>
                ) : null}
                {isActive ? (
                  <button
                    type="button"
                    onClick={() => void cancelJob()}
                    className="rounded-md border border-danger/60 px-3 py-1.5 text-sm font-semibold text-danger hover:bg-danger/15"
                  >
                    Stop
                  </button>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>

      <section>
        <h2 className="font-display text-xl font-semibold">Legacy YOLO strip</h2>
        <p className="mt-1 text-sm text-ink-500">Not used for production USD assets.</p>
        <div className="mt-3 flex flex-wrap gap-2">
          {legacy.map((s) => (
            <button
              key={s.id}
              type="button"
              disabled={jobRunning}
              className="rounded-md border border-ink-700 px-3 py-1.5 text-sm hover:border-accent disabled:opacity-40"
              onClick={() => {
                const opts: Record<string, unknown> = {};
                if (s.command === "export-dataset") opts.mode = "mixed";
                if (s.command === "benchmark") opts.experiment = "all";
                runCommand(s.command, opts);
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </section>

      {objects.length > 0 ? (
        <section>
          <h2 className="font-display text-xl font-semibold">Objects</h2>
          <div className="mt-3 overflow-auto rounded-lg border border-ink-800">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-ink-900 text-ink-400">
                <tr>
                  <th className="px-3 py-2">ID</th>
                  <th className="px-3 py-2">Class</th>
                  <th className="px-3 py-2">Movable</th>
                  <th className="px-3 py-2">State</th>
                  <th className="px-3 py-2">Coverage</th>
                </tr>
              </thead>
              <tbody>
                {objects.map((o) => (
                  <tr key={String(o.instance_id)} className="border-t border-ink-800">
                    <td className="px-3 py-2 font-mono text-xs">{String(o.instance_id)}</td>
                    <td className="px-3 py-2">{String(o.class_name)}</td>
                    <td className="px-3 py-2">{o.movable ? "yes" : "no"}</td>
                    <td className="px-3 py-2">{String(o.review_state)}</td>
                    <td className="px-3 py-2">{String(o.observed_background_coverage)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
