import { useState } from "react";
import { client } from "../api/client";
import { Tip } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

export function DoctorPage() {
  const { loaded, setError, runCommand } = useProject();
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      setReport(await client.doctor());
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setLoading(false);
    }
  }

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  const groups = (report?.groups as Record<string, Record<string, unknown>[]>) || {};

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-semibold">Doctor</h1>
          <p className="mt-1 flex items-center text-ink-400">
            Dependency health for COLMAP, Nerfstudio, hybrid USD tools, and Python
            <Tip text="Mirrors scan2usd doctor with structured results for the GUI." anchor="setup" />
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={run}
            disabled={loading}
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-ink-950 disabled:opacity-50"
          >
            {loading ? "Checking…" : "Run checks"}
          </button>
          <button
            type="button"
            onClick={() => runCommand("doctor")}
            className="rounded-md border border-ink-600 px-4 py-2 text-sm"
          >
            CLI doctor (logs)
          </button>
        </div>
      </div>

      {report ? (
        <>
          <div
            className={`rounded-lg border px-4 py-3 text-sm ${
              report.reconstruct_ready
                ? "border-ok/40 bg-ok/10 text-ok"
                : "border-warn/40 bg-warn/10 text-warn"
            }`}
          >
            {report.reconstruct_ready
              ? "Core reconstruct toolchain looks ready."
              : "Some required checks failed — see groups below."}
          </div>
          {report.apt_install_line ? (
            <pre className="overflow-auto rounded-lg border border-ink-800 bg-ink-950 p-3 font-mono text-xs text-ink-300">
              {String(report.apt_install_line)}
            </pre>
          ) : null}
          {Object.entries(groups).map(([name, items]) => (
            <section key={name}>
              <h2 className="font-display text-lg font-semibold capitalize">{name}</h2>
              <ul className="mt-2 space-y-1">
                {items.map((it) => (
                  <li
                    key={String(it.label)}
                    className="flex flex-wrap items-baseline gap-2 rounded-md border border-ink-800/80 px-3 py-2 text-sm"
                  >
                    <span className={it.ok ? "text-ok" : "text-danger"}>{it.ok ? "OK" : "MISS"}</span>
                    <span className="font-medium">{String(it.label)}</span>
                    <span className="font-mono text-xs text-ink-500">{String(it.detail)}</span>
                    {!it.ok && it.pip_hint ? (
                      <span className="w-full text-xs text-ink-400">{String(it.pip_hint)}</span>
                    ) : null}
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </>
      ) : (
        <p className="text-sm text-ink-500">Run checks to see structured results.</p>
      )}
    </div>
  );
}
