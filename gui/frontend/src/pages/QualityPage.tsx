import { useCallback, useEffect, useState } from "react";
import { client } from "../api/client";
import { useProject } from "../state/ProjectContext";

interface Metric {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  status: "pass" | "warn" | "fail" | "info" | "unknown";
  note: string;
}

interface QualityData {
  workspace: string;
  appearance: Metric[];
  clarity: Metric[];
  removed: Record<string, number>;
  kept: number | null;
  input_count: number | null;
  population: Record<string, number> | null;
  blocking: Record<string, number> | null;
  baseline_score: number | null;
  usable: boolean | null;
  warnings: string[];
  have: Record<string, boolean>;
}

const TONE: Record<string, string> = {
  pass: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10",
  warn: "text-amber-400 border-amber-500/40 bg-amber-500/10",
  fail: "text-rose-400 border-rose-500/40 bg-rose-500/10",
  info: "text-ink-300 border-ink-700 bg-ink-900/40",
  unknown: "text-ink-500 border-ink-800 bg-ink-900/20",
};

function show(m: Metric): string {
  if (m.value === null) return "—";
  if (m.unit === "fraction") return `${(m.value * 100).toFixed(1)}%`;
  if (Number.isInteger(m.value) && Math.abs(m.value) >= 1000)
    return m.value.toLocaleString();
  return Math.abs(m.value) >= 100 ? m.value.toFixed(1) : m.value.toFixed(3);
}

function Card({ metric }: { metric: Metric }) {
  return (
    <div className={`rounded border p-3 ${TONE[metric.status] ?? TONE.info}`}>
      <div className="text-xs uppercase tracking-wide opacity-70">{metric.label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">
        {show(metric)}
        {metric.unit && metric.unit !== "fraction" ? (
          <span className="ml-1 text-sm opacity-60">{metric.unit}</span>
        ) : null}
      </div>
      {metric.note ? <p className="mt-1 text-xs opacity-70">{metric.note}</p> : null}
    </div>
  );
}

export function QualityPage() {
  const { loaded, setError, runCommand } = useProject();
  const [data, setData] = useState<QualityData | null>(null);

  const refresh = useCallback(async () => {
    try {
      setData((await client.quality()) as unknown as QualityData);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, [setError]);

  useEffect(() => {
    if (!loaded) return;
    void refresh();
    const timer = setInterval(() => void refresh(), 10000);
    return () => clearInterval(timer);
  }, [loaded, refresh]);

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;
  if (!data) return <p className="text-ink-400">Loading…</p>;

  const removed = Object.entries(data.removed || {});
  const totalRemoved = removed.reduce((sum, [, n]) => sum + (n || 0), 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Scene quality</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          Two different questions. <strong>Appearance</strong> asks whether the
          held-out capture frames reproduce. <strong>Clarity</strong> asks whether
          the air in the room is clear — which appearance cannot see, because haze
          between the camera and a wall renders roughly the pixels the wall would.
        </p>
      </div>

      <section>
        <div className="mb-2 flex items-baseline gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-300">
            Appearance
          </h2>
          {data.baseline_score !== null ? (
            <span className="text-xs text-ink-500">
              baseline {data.baseline_score.toFixed(2)}
            </span>
          ) : null}
          {!data.have.scene_quality ? (
            <button
              className="text-xs text-sky-400 underline"
              onClick={() => runCommand("render-heldout", {})}
            >
              never scored — render held-out views
            </button>
          ) : null}
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {data.appearance.map((m) => (
            <Card key={m.key} metric={m} />
          ))}
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-baseline gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-300">
            Clarity
          </h2>
          {!data.have.cleanup ? (
            <span className="text-xs text-ink-500">no cleanup report yet</span>
          ) : null}
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {data.clarity.map((m) => (
            <Card key={m.key} metric={m} />
          ))}
        </div>
      </section>

      {removed.length ? (
        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-300">
            What cleanup removed
          </h2>
          <div className="overflow-x-auto rounded border border-ink-800">
            <table className="w-full text-sm">
              <tbody>
                {removed
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, count]) => (
                    <tr key={name} className="border-b border-ink-800/60 last:border-0">
                      <td className="px-3 py-1.5 capitalize text-ink-300">
                        {name.replace(/_/g, " ")}
                      </td>
                      <td className="px-3 py-1.5 text-right tabular-nums">
                        {count.toLocaleString()}
                      </td>
                      <td className="w-1/2 px-3 py-1.5">
                        <div className="h-1.5 rounded bg-ink-800">
                          <div
                            className="h-1.5 rounded bg-sky-500/70"
                            style={{ width: `${(count / totalRemoved) * 100}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="mt-1 text-xs text-ink-500">
            {data.kept?.toLocaleString()} kept of {data.input_count?.toLocaleString()} trained.
          </p>
        </section>
      ) : null}

      {data.warnings.length ? (
        <section>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-300">
            Warnings
          </h2>
          <ul className="space-y-1 text-sm text-amber-300/90">
            {data.warnings.map((w) => (
              <li key={w} className="rounded border border-amber-500/25 bg-amber-500/5 px-3 py-2">
                {w}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="flex flex-wrap gap-2 text-sm">
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800"
          onClick={() => runCommand("tool-analyze-splat", {})}
        >
          Re-measure fog
        </button>
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800"
          onClick={() => runCommand("render-heldout", {})}
        >
          Re-score held-out views
        </button>
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800"
          onClick={() => runCommand("tool-isaac-render-orbit", {})}
        >
          Render exterior orbit
        </button>
      </div>
    </div>
  );
}
