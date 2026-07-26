import { useState } from "react";
import { Link } from "react-router-dom";
import { client, LS_CWD } from "../api/client";
import { PathPickerModal } from "../components/PathPickerModal";
import { Tip } from "../components/ParamField";
import { useHelp } from "../state/HelpContext";
import { useProject } from "../state/ProjectContext";

export function ProjectPage() {
  const { loaded, configPath, cwd, workspace, openProject, setError, runCommand, config } =
    useProject();
  const { openHelp } = useHelp();
  const [path, setPath] = useState(configPath || "configs/example_scene.yaml");
  const [workCwd, setWorkCwd] = useState(cwd || localStorage.getItem(LS_CWD) || ".");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [yamlPicker, setYamlPicker] = useState(false);
  const [cwdPicker, setCwdPicker] = useState(false);
  const [createPath, setCreatePath] = useState("");
  const [createPicker, setCreatePicker] = useState(false);

  async function onOpen() {
    try {
      await openProject(path, workCwd);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  async function startFromExample() {
    const dest =
      createPath ||
      `${workCwd.replace(/\/$/, "")}/configs/my_scene.yaml`;
    try {
      // Prefer writing next to cwd
      const abs = dest.startsWith("/") ? dest : `${workCwd.replace(/\/$/, "")}/${dest}`;
      await client.createProject(abs, "configs/example_scene.yaml", workCwd);
      await openProject(abs, workCwd);
    } catch (e) {
      // If file exists, just open example
      try {
        await openProject(
          workCwd.replace(/\/$/, "") + "/configs/example_scene.yaml",
          workCwd,
        );
      } catch {
        setError(String((e as Error).message || e));
      }
    }
  }

  const hasVideo = Boolean(config && (config as { video_path?: string }).video_path);

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <h1 className="font-display text-3xl font-semibold tracking-tight">Welcome</h1>
        <p className="max-w-2xl text-ink-400">
          Scan2USD turns a room video into an Isaac Sim–ready OpenUSD scene. You do not need to
          edit YAML by hand — open a project, pick your video in Config, and follow the pipeline.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-ink-800 bg-ink-900/50 p-5">
          <h2 className="font-display text-lg font-semibold">Open existing YAML</h2>
          <p className="mt-1 text-sm text-ink-500">Browse for a scene config you already have.</p>
          <div className="mt-3 flex gap-2">
            <input
              className="min-w-0 flex-1 rounded-md border border-ink-700 bg-ink-950 px-2 py-1.5 font-mono text-xs"
              value={path}
              onChange={(e) => setPath(e.target.value)}
            />
            <button
              type="button"
              className="rounded-md border border-ink-600 px-2 text-sm hover:border-accent"
              onClick={() => setYamlPicker(true)}
            >
              Browse
            </button>
          </div>
          <button
            type="button"
            onClick={onOpen}
            className="mt-3 w-full rounded-md bg-accent px-3 py-2 text-sm font-semibold text-ink-950"
          >
            Open project
          </button>
        </div>

        <div className="rounded-xl border border-accent/30 bg-accent/5 p-5">
          <h2 className="font-display text-lg font-semibold">Start from example</h2>
          <p className="mt-1 text-sm text-ink-500">
            Copies <code className="text-ink-300">configs/example_scene.yaml</code> so you can edit safely.
          </p>
          <button
            type="button"
            onClick={startFromExample}
            className="mt-3 w-full rounded-md bg-accent px-3 py-2 text-sm font-semibold text-ink-950"
          >
            Create my_scene.yaml
          </button>
          <button
            type="button"
            className="mt-2 text-xs text-ink-500 hover:text-accent"
            onClick={() => setCreatePicker(true)}
          >
            Choose save location…
          </button>
        </div>

        <div className="rounded-xl border border-ink-800 bg-ink-900/50 p-5">
          <h2 className="font-display text-lg font-semibold">Learn the flow</h2>
          <p className="mt-1 text-sm text-ink-500">
            Opens the side Help dock — you stay on this page.
          </p>
          <button
            type="button"
            onClick={() => openHelp("getting-started")}
            className="mt-3 inline-block rounded-md border border-ink-600 px-3 py-2 text-sm hover:border-accent hover:text-accent"
          >
            Open Getting started →
          </button>
          <button
            type="button"
            onClick={() => openHelp("faq")}
            className="mt-2 block text-sm text-ink-400 hover:text-accent"
          >
            FAQ
          </button>
        </div>
      </section>

      <button
        type="button"
        className="text-sm text-ink-500 hover:text-accent"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "Hide" : "Show"} advanced path settings
      </button>
      {showAdvanced ? (
        <section className="space-y-3 rounded-xl border border-ink-800 bg-ink-900/40 p-4">
          <div className="flex items-center text-sm font-medium text-ink-300">
            Working directory (cwd)
            <Tip
              text="Relative paths like workspace/ are resolved from this folder. For beginners, set it to your Scan2USD repo root."
              anchor="paths"
            />
          </div>
          <div className="flex gap-2">
            <input
              className="flex-1 rounded-md border border-ink-700 bg-ink-950 px-3 py-2 font-mono text-sm"
              value={workCwd}
              onChange={(e) => setWorkCwd(e.target.value)}
            />
            <button
              type="button"
              className="rounded-md border border-ink-600 px-3 text-sm"
              onClick={() => setCwdPicker(true)}
            >
              Browse
            </button>
          </div>
        </section>
      ) : null}

      {loaded ? (
        <>
          <section className="rounded-xl border border-ink-800 bg-ink-900/40 p-5">
            <h2 className="font-display text-xl font-semibold">Next steps</h2>
            <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-ink-300">
              <li className={hasVideo ? "text-ok" : ""}>
                {hasVideo ? "✓" : ""} Set your source video in{" "}
                <Link className="text-accent hover:underline" to="/config">
                  Config → Essentials
                </Link>{" "}
                (Browse or <span className="text-ink-200">From phone…</span> QR) and Save
              </li>
              <li>
                Run{" "}
                <button type="button" className="text-accent hover:underline" onClick={() => runCommand("doctor")}>
                  Doctor
                </button>{" "}
                to check installed tools
              </li>
              <li>
                Follow the{" "}
                <Link className="text-accent hover:underline" to="/pipeline">
                  Pipeline
                </Link>{" "}
                checklist (start with “Build cameras from your video”)
              </li>
              <li>
                Read the{" "}
                <button
                  type="button"
                  className="text-accent hover:underline"
                  onClick={() => openHelp("sample-workflow")}
                >
                  sample workflow
                </button>{" "}
                (side Help panel)
              </li>
            </ol>
          </section>

          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {(
              [
                ["Config", configPath],
                ["Workspace", workspace?.workspace_dir],
                ["Manifest", workspace?.has_manifest ? "yes" : "no"],
                ["Pipeline state", workspace?.has_pipeline_state ? "yes" : "no"],
              ] as [string, unknown][]
            ).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-ink-800 bg-ink-900/40 p-4">
                <div className="text-[11px] uppercase tracking-wider text-ink-500">{k}</div>
                <div className="mt-1 truncate font-mono text-sm text-ink-200">{String(v ?? "—")}</div>
              </div>
            ))}
          </section>
        </>
      ) : null}

      <PathPickerModal
        open={yamlPicker}
        title="Choose scene YAML"
        kind="file"
        ext=".yaml,.yml"
        onClose={() => setYamlPicker(false)}
        onSelect={setPath}
      />
      <PathPickerModal
        open={cwdPicker}
        title="Choose working directory"
        kind="dir"
        onClose={() => setCwdPicker(false)}
        onSelect={setWorkCwd}
      />
      <PathPickerModal
        open={createPicker}
        title="Folder for new YAML (then name my_scene.yaml)"
        kind="dir"
        onClose={() => setCreatePicker(false)}
        onSelect={(dir) => setCreatePath(`${dir.replace(/\/$/, "")}/my_scene.yaml`)}
      />
    </div>
  );
}
