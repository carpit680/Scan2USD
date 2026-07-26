import { useEffect, useRef, useState } from "react";
import { useProject } from "../state/ProjectContext";

export function JobConsole() {
  const { activeJob, jobLines, cancelJob } = useProject();
  const preRef = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const el = preRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [jobLines]);

  async function copyLogs() {
    const text = jobLines.join("\n");
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be denied */
    }
  }

  if (!activeJob && jobLines.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-ink-700 bg-ink-900/40 p-4 text-sm text-ink-400">
        Job output will appear here when you run a command.
      </div>
    );
  }
  const running = activeJob?.status === "running" || activeJob?.status === "queued";
  const canCopy = jobLines.length > 0;
  return (
    <div className="overflow-hidden rounded-lg border border-ink-700 bg-ink-950">
      <div className="flex items-center justify-between gap-2 border-b border-ink-800 px-3 py-2">
        <div className="min-w-0 font-mono text-xs text-ink-400">
          {activeJob ? (
            <>
              <span className="text-accent">{activeJob.command}</span>
              <span className="mx-2 text-ink-600">·</span>
              <span className={running ? "text-warn" : ""}>{activeJob.status}</span>
              {activeJob.exit_code !== null ? (
                <span className="ml-2">exit {activeJob.exit_code}</span>
              ) : null}
              <span className="ml-2 text-ink-600">{jobLines.length} lines</span>
            </>
          ) : (
            "console"
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            disabled={!canCopy}
            onClick={() => void copyLogs()}
            className="rounded-md border border-ink-600 px-3 py-1 text-xs font-medium text-ink-300 hover:border-accent hover:text-accent disabled:opacity-40"
            title="Copy all log lines"
          >
            {copied ? "Copied" : "Copy"}
          </button>
          {running ? (
            <button
              type="button"
              onClick={() => void cancelJob()}
              className="rounded-md border border-danger/60 bg-danger/15 px-3 py-1 text-xs font-semibold text-danger hover:bg-danger/25"
            >
              Stop
            </button>
          ) : null}
        </div>
      </div>
      <pre
        ref={preRef}
        className="max-h-[28rem] overflow-auto p-3 font-mono text-[11px] leading-relaxed text-ink-300"
      >
        {jobLines.join("\n") || "Waiting for process output…"}
      </pre>
    </div>
  );
}
