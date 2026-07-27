import { useCallback, useEffect, useState } from "react";
import { client } from "../api/client";
import { Tip } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

interface Trial {
  trial_id: string;
  kind: string;
  params: Record<string, unknown>;
  status: string;
  quality_score: number | null;
  metrics?: Record<string, unknown>;
  error?: string | null;
  finished_at?: string | null;
}

interface TuningData {
  trials: Trial[];
  best_trial: Trial | null;
  scene_quality: Record<string, unknown> | null;
  tuned_config: string | null;
  budgets: { max_cheap_trials: number; max_retrain_trials: number; lpips: boolean };
  ready: { raw_splat: boolean; held_out: boolean; isaac_python: boolean };
  paths: { trials_json: string; scene_quality: string };
}

function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

export function TuningPage() {
  const { loaded, setError, runCommand } = useProject();
  const [data, setData] = useState<TuningData | null>(null);

  const refresh = useCallback(async () => {
    try {
      setData((await client.tuningTrials()) as unknown as TuningData);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, [setError]);

  useEffect(() => {
    if (!loaded) return;
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    return () => clearInterval(timer);
  }, [loaded, refresh]);

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  const ready = data?.ready;
  const blockers: string[] = [];
  if (ready && !ready.raw_splat)
    blockers.push("No environment_splat_raw.usd — run Build USD (with splat cleanup) first.");
  if (ready && !ready.held_out)
    blockers.push("No held_out.json — the 3DGRUT visual build stages held-out views.");
  if (ready && !ready.isaac_python)
    blockers.push("external.isaac_python is not configured (Config → External tools).");

  const quality = data?.scene_quality as
    | { quality_score?: number | null; photorealism?: Record<string, unknown> }
    | null
    | undefined;
  const photo = (quality?.photorealism || {}) as Record<string, unknown>;
  const trials = [...(data?.trials || [])].sort(
    (a, b) => (b.quality_score ?? -1) - (a.quality_score ?? -1),
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-semibold">Auto-tuning</h1>
          <p className="mt-1 flex items-center text-ink-400">
            Tune config → export USD → render held-out views in Isaac → score → retune
            <Tip
              text="Cheap trials sweep splat-cleanup thresholds without retraining. Retrain trials re-run 3DGRUT training (hours each). The winner can be promoted to a *_tuned.yaml config."
              anchor="hybrid"
            />
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => runCommand("tune", { retrain_trials: 0 })}
            disabled={blockers.length > 0}
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-ink-950 disabled:opacity-50"
          >
            Tune (cheap only)
          </button>
          <button
            type="button"
            onClick={() => runCommand("tune", {})}
            disabled={blockers.length > 0}
            className="rounded-md border border-ink-600 px-4 py-2 text-sm"
            title="Uses tuning.max_retrain_trials from Config (0 = cheap only)"
          >
            Tune (config budgets)
          </button>
          <button
            type="button"
            onClick={() => runCommand("render-heldout", {})}
            className="rounded-md border border-ink-600 px-4 py-2 text-sm"
          >
            Score current scene
          </button>
        </div>
      </div>

      {blockers.length > 0 && (
        <div className="rounded-lg border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-warn">
          <p className="font-semibold">Tuning is blocked:</p>
          <ul className="ml-5 list-disc">
            {blockers.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-ink-400">Quality score</p>
          <p className="mt-1 text-2xl font-semibold">
            {fmt(quality?.quality_score, 1)}
          </p>
        </div>
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-ink-400">PSNR</p>
          <p className="mt-1 text-2xl font-semibold">{fmt(photo.mean_psnr, 2)}</p>
        </div>
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-ink-400">SSIM</p>
          <p className="mt-1 text-2xl font-semibold">{fmt(photo.mean_ssim, 3)}</p>
        </div>
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-ink-400">LPIPS ↓</p>
          <p className="mt-1 text-2xl font-semibold">{fmt(photo.mean_lpips, 3)}</p>
        </div>
        <div className="rounded-lg border border-ink-700 bg-ink-900 px-4 py-3">
          <p className="text-xs uppercase tracking-wide text-ink-400">Views</p>
          <p className="mt-1 text-2xl font-semibold">
            {String(photo.evaluated_views ?? "—")}/{String(photo.expected_views ?? "—")}
          </p>
        </div>
      </div>

      {data?.tuned_config && (
        <div className="rounded-lg border border-ok/40 bg-ok/10 px-4 py-3 text-sm text-ok">
          Promoted config: <code className="font-mono">{data.tuned_config}</code> — open it as
          the project config to build with the winning parameters.
        </div>
      )}

      <div>
        <h2 className="mb-2 font-display text-xl font-semibold">
          Trials {data?.best_trial ? `(best: ${data.best_trial.trial_id})` : ""}
        </h2>
        {trials.length === 0 ? (
          <p className="text-ink-400">
            No trials yet. Start a tuning run — progress streams into the job console and this
            table refreshes automatically.
          </p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-ink-700">
            <table className="w-full text-left text-sm">
              <thead className="bg-ink-900 text-xs uppercase tracking-wide text-ink-400">
                <tr>
                  <th className="px-3 py-2">Trial</th>
                  <th className="px-3 py-2">Kind</th>
                  <th className="px-3 py-2">Parameters</th>
                  <th className="px-3 py-2">Score</th>
                  <th className="px-3 py-2">PSNR</th>
                  <th className="px-3 py-2">SSIM</th>
                  <th className="px-3 py-2">LPIPS</th>
                  <th className="px-3 py-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {trials.map((trial) => (
                  <tr
                    key={trial.trial_id}
                    className={`border-t border-ink-800 ${
                      data?.best_trial?.trial_id === trial.trial_id ? "bg-ok/10" : ""
                    }`}
                  >
                    <td className="px-3 py-2 font-mono">{trial.trial_id}</td>
                    <td className="px-3 py-2">{trial.kind}</td>
                    <td className="px-3 py-2 font-mono text-xs">
                      {Object.entries(trial.params)
                        .map(([key, value]) => `${key}=${String(value)}`)
                        .join(" ")}
                    </td>
                    <td className="px-3 py-2 font-semibold">{fmt(trial.quality_score, 1)}</td>
                    <td className="px-3 py-2">{fmt(trial.metrics?.mean_psnr, 2)}</td>
                    <td className="px-3 py-2">{fmt(trial.metrics?.mean_ssim, 3)}</td>
                    <td className="px-3 py-2">{fmt(trial.metrics?.mean_lpips, 3)}</td>
                    <td className="px-3 py-2">
                      {trial.status === "failed" ? (
                        <span className="text-err" title={trial.error ?? undefined}>
                          failed
                        </span>
                      ) : (
                        trial.status
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
