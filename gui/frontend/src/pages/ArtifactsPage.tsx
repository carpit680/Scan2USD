import { useEffect, useState } from "react";
import { artifactFileUrl, client } from "../api/client";
import { useProject } from "../state/ProjectContext";

export function ArtifactsPage() {
  const { loaded, setError } = useProject();
  const [roots, setRoots] = useState<{ id: string; path: string; exists: boolean }[]>([]);
  const [root, setRoot] = useState("workspace");
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<
    { name: string; path: string; is_dir: boolean; size: number | null; suffix: string }[]
  >([]);

  async function loadRoots() {
    const data = await client.artifactRoots();
    setRoots(data.roots);
  }

  async function loadList(r: string, p: string) {
    const data = await client.artifactList(r, p);
    setEntries(data.entries || []);
    setPath(data.path || p);
  }

  useEffect(() => {
    if (!loaded) return;
    loadRoots()
      .then(() => loadList(root, ""))
      .catch((e) => setError(String(e.message || e)));
  }, [loaded]);

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  const crumbs = path ? path.split("/") : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-semibold">Artifacts</h1>
        <p className="mt-1 text-ink-400">Browse workspace frames, masks, USD layers, and reports.</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {roots.map((r) => (
          <button
            key={r.id}
            type="button"
            disabled={!r.exists}
            onClick={() => {
              setRoot(r.id);
              loadList(r.id, "").catch((e) => setError(String(e.message || e)));
            }}
            className={`rounded-md px-3 py-1.5 text-sm ${
              root === r.id ? "bg-ink-800 text-accent" : "border border-ink-700 text-ink-400"
            } disabled:opacity-40`}
          >
            {r.id}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-1 font-mono text-xs text-ink-500">
        <button type="button" className="hover:text-accent" onClick={() => loadList(root, "")}>
          {root}
        </button>
        {crumbs.map((c, i) => {
          const sub = crumbs.slice(0, i + 1).join("/");
          return (
            <span key={sub}>
              /
              <button type="button" className="hover:text-accent" onClick={() => loadList(root, sub)}>
                {c}
              </button>
            </span>
          );
        })}
      </div>

      <div className="overflow-hidden rounded-xl border border-ink-800">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-ink-900 text-ink-400">
            <tr>
              <th className="px-3 py-2">Name</th>
              <th className="px-3 py-2">Size</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody>
            {path ? (
              <tr className="border-t border-ink-800">
                <td className="px-3 py-2" colSpan={3}>
                  <button
                    type="button"
                    className="text-accent"
                    onClick={() => {
                      const parts = path.split("/");
                      parts.pop();
                      loadList(root, parts.join("/"));
                    }}
                  >
                    ..
                  </button>
                </td>
              </tr>
            ) : null}
            {entries.map((e) => (
              <tr key={e.path} className="border-t border-ink-800 hover:bg-ink-900/50">
                <td className="px-3 py-2 font-mono text-xs">
                  {e.is_dir ? (
                    <button type="button" className="text-left hover:text-accent" onClick={() => loadList(root, e.path)}>
                      {e.name}/
                    </button>
                  ) : (
                    e.name
                  )}
                </td>
                <td className="px-3 py-2 text-ink-500">{e.size ?? "—"}</td>
                <td className="px-3 py-2 text-right">
                  {!e.is_dir ? (
                    <a
                      className="text-accent hover:underline"
                      href={artifactFileUrl(root, e.path)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open
                    </a>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
