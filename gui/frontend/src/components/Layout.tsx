import { NavLink, Outlet } from "react-router-dom";
import { useProject } from "../state/ProjectContext";
import { useHelp } from "../state/HelpContext";
import { JobConsole } from "./JobConsole";
import { HelpDock } from "./HelpDock";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/pipeline", label: "Pipeline" },
  { to: "/config", label: "Config" },
  { to: "/commands", label: "Commands" },
  { to: "/review", label: "Review" },
  { to: "/metric", label: "Metric" },
  { to: "/tuning", label: "Tuning" },
  { to: "/artifacts", label: "Artifacts" },
  { to: "/doctor", label: "Doctor" },
  { to: "/guide", label: "Guide" },
];

export function Layout() {
  const { configPath, loaded, error, setError, activeJob, cancelJob } = useProject();
  const { open, toggleHelp } = useHelp();
  const jobRunning = activeJob?.status === "running" || activeJob?.status === "queued";

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-ink-800/80 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[1600px] items-center gap-4 px-4 py-3">
          <div className="shrink-0">
            <div className="font-display text-lg font-semibold tracking-tight text-ink-100">
              Scan<span className="text-accent">2</span>USD
            </div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-ink-400">
              Pipeline GUI
            </div>
          </div>
          <nav className="flex flex-1 flex-wrap gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `rounded-md px-3 py-1.5 text-sm transition ${
                    isActive
                      ? "bg-ink-800 text-accent"
                      : "text-ink-400 hover:bg-ink-900 hover:text-ink-100"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <button
            type="button"
            onClick={toggleHelp}
            className={`shrink-0 rounded-md px-3 py-1.5 text-sm font-medium ${
              open
                ? "bg-accent text-ink-950"
                : "border border-ink-600 text-ink-300 hover:border-accent hover:text-accent"
            }`}
            aria-pressed={open}
            title="Toggle side-by-side help (Esc to close)"
          >
            Help
          </button>
          {jobRunning ? (
            <button
              type="button"
              onClick={() => void cancelJob()}
              className="shrink-0 rounded-md border border-danger/60 bg-danger/15 px-3 py-1.5 text-sm font-semibold text-danger hover:bg-danger/25"
              title={`Stop ${activeJob?.command || "job"}`}
            >
              Stop
            </button>
          ) : null}
          <div className="hidden max-w-[180px] truncate font-mono text-[11px] text-ink-400 xl:block">
            {loaded ? configPath : "No project open"}
            {jobRunning ? (
              <span className="ml-2 inline-block h-2 w-2 animate-pulse rounded-full bg-accent" />
            ) : null}
          </div>
        </div>
      </header>

      {error ? (
        <div className="border-b border-danger/40 bg-danger/10 px-4 py-2 text-sm text-danger">
          <div className="mx-auto flex max-w-[1600px] items-start justify-between gap-4">
            <span>{error}</span>
            <button type="button" className="text-ink-300 hover:text-ink-100" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        </div>
      ) : null}

      <div className="mx-auto flex w-full max-w-[1600px] flex-1">
        <main className="flex min-w-0 flex-1 flex-col gap-4 px-4 py-6">
          <Outlet />
          <JobConsole />
        </main>
        <HelpDock />
      </div>
    </div>
  );
}
