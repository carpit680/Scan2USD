import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ParamField } from "../components/ParamField";
import { useProject } from "../state/ProjectContext";

/** Keys that default to <workspace_dir>/<leaf> — drop them when workspace changes. */
const WORKSPACE_DERIVED_PATHS = [
  "frames_dir",
  "colmap_txt_dir",
  "nerfstudio_data_dir",
  "renders_dir",
  "dataset_dir",
  "segmentation.masks_dir",
  "usd.output_dir",
] as const;

function getByPath(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

function setByPath(obj: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const parts = path.split(".");
  const root = structuredClone(obj);
  let cur: Record<string, unknown> = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i];
    const next = cur[key];
    if (next === null || typeof next !== "object" || Array.isArray(next)) {
      cur[key] = {};
    }
    cur = cur[key] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1]] = value;
  return root;
}

function deleteByPath(obj: Record<string, unknown>, path: string): Record<string, unknown> {
  const parts = path.split(".");
  const root = structuredClone(obj);
  let cur: Record<string, unknown> = root;
  const stack: { parent: Record<string, unknown>; key: string }[] = [];
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i];
    const next = cur[key];
    if (next === null || typeof next !== "object" || Array.isArray(next)) {
      return root;
    }
    stack.push({ parent: cur, key });
    cur = next as Record<string, unknown>;
  }
  delete cur[parts[parts.length - 1]];
  for (let i = stack.length - 1; i >= 0; i--) {
    const { parent, key } = stack[i];
    const child = parent[key];
    if (
      child &&
      typeof child === "object" &&
      !Array.isArray(child) &&
      Object.keys(child as object).length === 0
    ) {
      delete parent[key];
    } else {
      break;
    }
  }
  return root;
}

function stripWorkspaceDerivedPaths(obj: Record<string, unknown>): Record<string, unknown> {
  let next = obj;
  for (const path of WORKSPACE_DERIVED_PATHS) {
    next = deleteByPath(next, path);
  }
  return next;
}

type ConfigMode = "essential" | "advanced" | "raw";

export function ConfigPage() {
  const { loaded, raw, config, yamlText, schema, saveRaw, saveYamlText, setError } = useProject();
  const [mode, setMode] = useState<ConfigMode>("essential");
  const [group, setGroup] = useState("project");
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [yamlDraft, setYamlDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [showYaml, setShowYaml] = useState(false);

  useEffect(() => {
    if (mode === "raw" && yamlDraft === null && yamlText != null) {
      setYamlDraft(yamlText);
    }
  }, [mode, yamlText, yamlDraft]);

  const active = draft || raw;
  const formDirty = draft !== null;
  const yamlDirty = yamlDraft !== null && yamlDraft !== yamlText;
  const dirty = mode === "raw" ? yamlDirty : formDirty;

  const params = useMemo(() => {
    if (!schema) return [];
    if (mode === "essential") {
      return schema.config_params.filter((p) => p.help_level === "essential");
    }
    if (mode === "advanced") {
      return schema.config_params.filter((p) => p.group === group);
    }
    return [];
  }, [schema, group, mode]);

  if (!loaded || !raw || !schema) {
    return (
      <div className="space-y-3">
        <p className="text-ink-400">Open a project first so we know which YAML to edit.</p>
        <Link to="/" className="text-accent hover:underline">
          ← Back to Project
        </Link>
      </div>
    );
  }

  const working = active!;

  async function onSave() {
    setSaving(true);
    setSavedMsg(null);
    try {
      if (mode === "raw") {
        await saveYamlText(yamlDraft ?? yamlText ?? "");
        setYamlDraft(null);
      } else {
        await saveRaw(working);
        setDraft(null);
      }
      setSavedMsg("Saved. Scan2USD validated the config.");
    } catch (e) {
      setError(String((e as Error).message || e));
    } finally {
      setSaving(false);
    }
  }

  function onDiscard() {
    if (mode === "raw") {
      setYamlDraft(yamlText);
    } else {
      setDraft(null);
    }
    setSavedMsg(null);
  }

  function onFieldChange(configPath: string, value: unknown) {
    let next = setByPath(working, configPath, value);
    if (configPath === "workspace_dir") {
      next = stripWorkspaceDerivedPaths(next);
    }
    setDraft(next);
    setSavedMsg(null);
  }

  return (
    <div className="space-y-6">
      <div className="sticky top-14 z-30 -mx-4 flex flex-wrap items-end justify-between gap-4 border-b border-ink-800/80 bg-ink-950/95 px-4 py-3 backdrop-blur">
        <div>
          <h1 className="font-display text-3xl font-semibold">Config</h1>
          <p className="mt-1 text-sm text-ink-400">
            Friendly controls for every setting. Hover <span className="text-accent">?</span> for help.
            {dirty ? <span className="ml-2 text-warn">Unsaved changes</span> : null}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {mode !== "raw" ? (
            <button
              type="button"
              className="rounded-md border border-ink-700 px-3 py-1.5 text-sm text-ink-300 hover:border-ink-500"
              onClick={() => setShowYaml((s) => !s)}
            >
              {showYaml ? "Hide JSON" : "Show JSON"}
            </button>
          ) : null}
          {dirty ? (
            <button
              type="button"
              className="rounded-md border border-ink-700 px-3 py-1.5 text-sm text-ink-300 hover:border-ink-500"
              onClick={onDiscard}
            >
              Discard
            </button>
          ) : null}
          <button
            type="button"
            disabled={!dirty || saving}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-ink-950 disabled:opacity-40"
            onClick={() => void onSave()}
          >
            {saving ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
      {savedMsg ? <p className="text-sm text-ok">{savedMsg}</p> : null}

      <div className="flex flex-wrap gap-2">
        {(
          [
            ["essential", "Essential"],
            ["advanced", "Advanced"],
            ["raw", "Raw YAML"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => {
              if (id === "raw") {
                setYamlDraft(yamlText);
              }
              setMode(id);
            }}
            className={`rounded-md px-3 py-1.5 text-sm ${
              mode === id ? "bg-ink-800 text-accent" : "border border-ink-800 text-ink-400 hover:border-ink-600"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === "raw" ? (
        <textarea
          className="min-h-[28rem] w-full rounded-lg border border-ink-800 bg-ink-950 p-3 font-mono text-xs text-ink-200"
          value={yamlDraft ?? ""}
          onChange={(e) => {
            setYamlDraft(e.target.value);
            setSavedMsg(null);
          }}
          spellCheck={false}
        />
      ) : (
        <div className="flex gap-6">
          {mode === "advanced" ? (
            <aside className="flex w-44 shrink-0 flex-col gap-1">
              {schema.config_groups.map((g) => (
                <button
                  key={g.id}
                  type="button"
                  onClick={() => setGroup(g.id)}
                  className={`rounded-md px-3 py-1.5 text-left text-sm ${
                    group === g.id ? "bg-ink-800 text-accent" : "text-ink-400 hover:bg-ink-900"
                  }`}
                >
                  {g.label}
                </button>
              ))}
            </aside>
          ) : null}
          <div className="grid flex-1 gap-4 sm:grid-cols-2">
            {mode === "essential" ? (
              <p className="sm:col-span-2 text-sm text-ink-500">
                Set the scene name, source video, and workspace folder once — frames, masks, USD, and
                build artifacts follow that workspace. Sensor type and review gates live here too;
                everything else is under Advanced.
              </p>
            ) : null}
            {params.map((p) => {
              if (!p.config_path) return null;
              const rawValue = getByPath(working, p.config_path);
              const resolved =
                rawValue !== undefined
                  ? rawValue
                  : config
                    ? getByPath(config as Record<string, unknown>, p.config_path)
                    : undefined;
              return (
                <ParamField
                  key={p.id}
                  param={p}
                  value={resolved === undefined ? p.default : resolved}
                  onChange={(v) => onFieldChange(p.config_path!, v)}
                />
              );
            })}
            {params.length === 0 ? (
              <p className="text-sm text-ink-500 sm:col-span-2">No parameters in this section.</p>
            ) : null}
          </div>
        </div>
      )}

      {showYaml && mode !== "raw" ? (
        <pre className="max-h-80 overflow-auto rounded-lg border border-ink-800 bg-ink-950 p-3 font-mono text-[11px] text-ink-300">
          {JSON.stringify(working, null, 2)}
        </pre>
      ) : null}
    </div>
  );
}
