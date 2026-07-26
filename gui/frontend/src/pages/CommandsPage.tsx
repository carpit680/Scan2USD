import { useEffect, useMemo, useState } from "react";
import type { CommandDef, ParamDef } from "../api/client";
import { ParamField } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

const CATS = ["setup", "capture", "hybrid", "legacy", "tools"] as const;

/** Fallback when schema is stale / missing default_from (command id → option name → path key). */
const PATH_DEFAULT_FROM: Record<string, Record<string, string>> = {
  view: { objects_npz: "objects_3d" },
  "apply-metric-scale": {
    floor_json: "colmap_to_usd_floor",
    output: "colmap_to_usd_metric",
  },
  "tool-sam2": {
    images: "frames_dir",
    proposals: "proposals_json",
    output: "masks_dir",
  },
  "tool-masked-object-recon": {
    images: "frames_dir",
    masks: "masks_dir",
    colmap: "colmap_txt_dir",
  },
  "tool-cleanup-splat-usd": {
    input: "environment_splat_cleanup_input",
    output: "environment_splat",
    report: "splat_cleanup_report",
    raw_backup: "environment_splat_raw",
  },
  "tool-isaac-view": { stage: "root_usd" },
  "tool-isaac-validate": {
    stage: "root_usd",
    output: "isaac_validate_report",
  },
  "tool-isaac-convert": { input: "root_usd" },
};

/** Fallback config dotted paths for command option defaults. */
const CONFIG_DEFAULT_FROM: Record<string, Record<string, string>> = {
  "cleanup-splat": {
    outlier_std: "reconstruction.splat_cleanup.outlier_std",
    min_opacity: "reconstruction.splat_cleanup.min_opacity",
  },
  "tool-cleanup-splat-usd": {
    outlier_std: "reconstruction.splat_cleanup.outlier_std",
    min_opacity: "reconstruction.splat_cleanup.min_opacity",
    max_scale: "reconstruction.splat_cleanup.max_scale",
  },
};

function joinPath(...parts: string[]): string {
  return parts
    .filter(Boolean)
    .map((p, i) => (i === 0 ? p.replace(/\/+$/, "") : p.replace(/^\/+|\/+$/g, "")))
    .filter(Boolean)
    .join("/");
}

function relativize(path: string, cwd: string): string {
  const norm = path.replace(/\\/g, "/");
  const base = cwd.replace(/\\/g, "/").replace(/\/+$/, "");
  if (base && (norm === base || norm.startsWith(base + "/"))) {
    return norm === base ? "." : norm.slice(base.length + 1);
  }
  return path;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function str(value: unknown): string | null {
  if (typeof value === "string" && value.length > 0) return value;
  return null;
}

/** Build command path defaults from resolved scene config (+ optional workspace.paths). */
function buildPathDefaults(
  config: Record<string, unknown> | null,
  workspace: Record<string, unknown> | null,
  cwd: string,
): Record<string, string> {
  const out: Record<string, string> = {};

  const put = (key: string, value: string | null | undefined) => {
    if (value) out[key] = relativize(value, cwd);
  };

  if (config) {
    const ws = str(config.workspace_dir) || "";
    const usd = asRecord(config.usd) || {};
    const seg = asRecord(config.segmentation) || {};
    const usdDir = str(usd.output_dir) || (ws ? joinPath(ws, "usd") : null);
    const rootName = str(usd.root_filename) || "scene.usd";
    const masks = str(seg.masks_dir) || (ws ? joinPath(ws, "masks") : null);
    const visual = ws ? joinPath(ws, "build", "visual") : null;
    const envSplat = visual ? joinPath(visual, "environment_splat.usd") : null;
    const envRaw = visual ? joinPath(visual, "environment_splat_raw.usd") : null;

    put("workspace_dir", ws || null);
    put("frames_dir", str(config.frames_dir));
    put("colmap_txt_dir", str(config.colmap_txt_dir));
    put("nerfstudio_data_dir", str(config.nerfstudio_data_dir));
    put("renders_dir", str(config.renders_dir));
    put("dataset_dir", str(config.dataset_dir));
    put("masks_dir", masks);
    put("usd_dir", usdDir);
    put("root_usd", usdDir ? joinPath(usdDir, rootName) : null);
    put("environment_splat", envSplat);
    put("environment_splat_raw", envRaw);
    put("environment_splat_cleanup_input", envRaw || envSplat);
    put("splat_cleanup_report", visual ? joinPath(visual, "splat_cleanup_report.json") : null);
    put("proposals_json", ws ? joinPath(ws, "build", "segmentation", "proposals.json") : null);
    put("segmentation_dir", ws ? joinPath(ws, "build", "segmentation") : null);
    put("colmap_to_usd_floor", ws ? joinPath(ws, "colmap_to_usd_floor.json") : null);
    put("colmap_to_usd_metric", ws ? joinPath(ws, "colmap_to_usd_metric.json") : null);
    put("objects_3d", ws ? joinPath(ws, "objects_3d.npz") : null);
    put("video_path", str(config.video_path));
    put("validate_report", ws ? joinPath(ws, "build", "validate_report.json") : null);
    put("isaac_validate_report", ws ? joinPath(ws, "build", "isaac_validate_report.json") : null);
  }

  // Prefer API workspace.paths when present (manifest-aware splat location, etc.)
  const nested = asRecord(workspace?.paths);
  if (nested) {
    for (const [k, v] of Object.entries(nested)) {
      const s = str(v);
      if (s) out[k] = relativize(s, cwd);
    }
  }
  if (workspace) {
    for (const key of ["workspace_dir", "usd_dir", "frames_dir", "masks_dir"] as const) {
      const s = str(workspace[key]);
      if (s && !out[key]) out[key] = relativize(s, cwd);
    }
  }

  // Prefer raw backup as cleanup input when the API says so; otherwise keep derived.
  if (!out.environment_splat_cleanup_input) {
    out.environment_splat_cleanup_input =
      out.environment_splat_raw || out.environment_splat || "";
    if (!out.environment_splat_cleanup_input) delete out.environment_splat_cleanup_input;
  }

  return out;
}

function getByPath(obj: Record<string, unknown> | null, path: string): unknown {
  if (!obj) return undefined;
  let cur: unknown = obj;
  for (const part of path.split(".")) {
    if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

function optionPathDefaultFrom(commandId: string, opt: ParamDef): string | null {
  if (opt.default_from) return opt.default_from;
  const key = opt.config_path || opt.id;
  return PATH_DEFAULT_FROM[commandId]?.[key] || null;
}

function optionConfigDefaultFrom(commandId: string, opt: ParamDef): string | null {
  if (opt.config_default_from) return opt.config_default_from;
  const key = opt.config_path || opt.id;
  return CONFIG_DEFAULT_FROM[commandId]?.[key] || null;
}

function isEmptyOptionValue(value: unknown): boolean {
  return value === null || value === undefined || value === "";
}

function initialOptionValue(
  commandId: string,
  opt: ParamDef,
  paths: Record<string, string>,
  config: Record<string, unknown> | null,
): unknown {
  const pathKey = optionPathDefaultFrom(commandId, opt);
  if (pathKey && paths[pathKey]) return paths[pathKey];

  const configKey = optionConfigDefaultFrom(commandId, opt);
  if (configKey) {
    const fromCfg = getByPath(config, configKey);
    if (!isEmptyOptionValue(fromCfg)) return fromCfg;
  }

  return opt.default;
}

function buildInitialOpts(
  cmd: CommandDef,
  paths: Record<string, string>,
  config: Record<string, unknown> | null,
): Record<string, unknown> {
  const initial: Record<string, unknown> = {};
  for (const o of cmd.options) {
    const key = o.config_path || o.id;
    initial[key] = initialOptionValue(cmd.id, o, paths, config);
  }
  return initial;
}

export function CommandsPage() {
  const { loaded, schema, config, workspace, cwd, runCommand, setError } = useProject();
  const [cat, setCat] = useState<string>("hybrid");
  const [selected, setSelected] = useState<string | null>(null);
  const [opts, setOpts] = useState<Record<string, unknown>>({});

  const paths = useMemo(
    () => buildPathDefaults(config, workspace, cwd || "."),
    [config, workspace, cwd],
  );

  const commands = useMemo(() => {
    return (schema?.commands || []).filter((c) => c.category === cat);
  }, [schema, cat]);

  const cmd: CommandDef | undefined = (schema?.commands || []).find((c) => c.id === selected);

  // Fill empty path fields; keep config-backed knobs in sync with the open scene YAML.
  useEffect(() => {
    if (!cmd) return;
    setOpts((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const o of cmd.options) {
        const key = o.config_path || o.id;
        const cur = next[key];
        const configKey = optionConfigDefaultFrom(cmd.id, o);
        if (configKey) {
          const fromCfg = getByPath(config, configKey);
          if (!isEmptyOptionValue(fromCfg) && fromCfg !== cur) {
            next[key] = fromCfg;
            changed = true;
          }
          continue;
        }
        if (o.type !== "path" || !isEmptyOptionValue(cur)) continue;
        const filled = initialOptionValue(cmd.id, o, paths, config);
        if (!isEmptyOptionValue(filled) && filled !== cur) {
          next[key] = filled;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [cmd, paths, config]);

  if (!loaded || !schema) {
    return <p className="text-ink-400">Open a project to run commands.</p>;
  }

  function pick(c: CommandDef) {
    setSelected(c.id);
    setOpts(buildInitialOpts(c, paths, config));
  }

  async function onRun() {
    if (!cmd) return;
    if (cmd.dangerous) {
      const ok = window.confirm(
        `Run dangerous command "${cmd.id}"? Use dry_run first when cleaning.`,
      );
      if (!ok) return;
    }
    try {
      await runCommand(cmd.id, opts);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  const isTool = Boolean(cmd?.job_kind === "tool" || cmd?.category === "tools");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-semibold">Commands</h1>
        <p className="mt-1 text-ink-400">
          Every Scan2USD CLI command plus allowlisted <code className="text-ink-300">tools/</code>{" "}
          runners. Path fields default from the open scene config.
        </p>
      </div>

      <div className="flex flex-wrap gap-1">
        {CATS.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => setCat(c)}
            className={`rounded-md px-3 py-1.5 text-sm capitalize ${
              cat === c ? "bg-ink-800 text-accent" : "text-ink-400 hover:bg-ink-900"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <div className="space-y-1 rounded-xl border border-ink-800 bg-ink-900/40 p-2">
          {commands.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => pick(c)}
              className={`block w-full rounded-md px-3 py-2 text-left text-sm ${
                selected === c.id ? "bg-ink-800 text-accent" : "hover:bg-ink-900"
              }`}
            >
              <div className="font-medium">{c.label}</div>
              <div className="font-mono text-[10px] text-ink-500">{c.id}</div>
            </button>
          ))}
        </div>

        <div className="rounded-xl border border-ink-800 bg-ink-900/40 p-5">
          {cmd ? (
            <div className="space-y-4">
              <div>
                <h2 className="font-display text-xl font-semibold">{cmd.label}</h2>
                <p className="mt-1 text-sm text-ink-400">{cmd.description}</p>
                <code className="mt-2 inline-block font-mono text-xs text-accent">
                  {isTool ? `python ${cmd.script || "tools/…"}` : `scan2usd ${cmd.id}`}
                </code>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {cmd.options.map((o) => {
                  const key = o.config_path || o.id;
                  return (
                    <ParamField
                      key={o.id}
                      param={o}
                      value={opts[key]}
                      onChange={(v) => setOpts((prev) => ({ ...prev, [key]: v }))}
                    />
                  );
                })}
                {cmd.options.length === 0 ? (
                  <p className="text-sm text-ink-500 sm:col-span-2">
                    No extra options — uses scene YAML only.
                  </p>
                ) : null}
              </div>
              <button
                type="button"
                onClick={onRun}
                className={`rounded-md px-4 py-2 text-sm font-semibold ${
                  cmd.dangerous
                    ? "bg-danger/90 text-ink-950 hover:bg-danger"
                    : "bg-accent text-ink-950 hover:bg-accent-dim"
                }`}
              >
                Run
              </button>
            </div>
          ) : (
            <p className="text-sm text-ink-500">Select a command.</p>
          )}
        </div>
      </div>
    </div>
  );
}
