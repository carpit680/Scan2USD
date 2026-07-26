import { useCallback, useEffect, useState } from "react";
import { client } from "../api/client";
import { MobileUploadDialog } from "./MobileUploadDialog";

export interface PathPickerProps {
  open: boolean;
  title?: string;
  kind?: "file" | "dir" | "any";
  ext?: string | null;
  initialPath?: string | null;
  onClose: () => void;
  onSelect: (path: string) => void;
  allowUpload?: boolean;
  allowPhoneUpload?: boolean;
}

export function PathPickerModal({
  open,
  title = "Choose a path",
  kind = "any",
  ext = null,
  initialPath = null,
  onClose,
  onSelect,
  allowUpload = false,
  allowPhoneUpload = false,
}: PathPickerProps) {
  const [roots, setRoots] = useState<string[]>([]);
  const [path, setPath] = useState("");
  const [parent, setParent] = useState<string | null>(null);
  const [entries, setEntries] = useState<
    { name: string; path: string; is_dir: boolean; suffix: string; size: number | null }[]
  >([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [phoneOpen, setPhoneOpen] = useState(false);

  const load = useCallback(
    async (browsePath?: string | null) => {
      setLoading(true);
      setError(null);
      try {
        const data = await client.fsBrowse(browsePath || undefined, kind, ext || undefined);
        setPath(data.path);
        setParent(data.parent);
        setEntries(data.entries);
        setRoots(data.roots);
        setSelected(null);
      } catch (e) {
        setError(String((e as Error).message || e));
      } finally {
        setLoading(false);
      }
    },
    [kind, ext],
  );

  useEffect(() => {
    if (!open) return;
    load(initialPath || null);
  }, [open, initialPath, load]);

  if (!open) return null;

  async function onUpload(file: File | null) {
    if (!file) return;
    try {
      const res = await client.fsUpload(file);
      onSelect(res.path);
      onClose();
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }

  const canSelectDir = kind === "dir" || kind === "any";
  const canConfirm =
    selected !== null ||
    (canSelectDir && path && (kind === "dir" || kind === "any"));
  const showPhone = allowPhoneUpload || allowUpload;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-ink-700 bg-ink-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-ink-800 px-4 py-3">
          <h2 className="font-display text-lg font-semibold">{title}</h2>
          <button type="button" className="text-ink-400 hover:text-ink-100" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="flex flex-wrap gap-1 border-b border-ink-800 px-3 py-2">
          {roots.map((r) => (
            <button
              key={r}
              type="button"
              className="rounded px-2 py-1 font-mono text-[10px] text-ink-400 hover:bg-ink-800 hover:text-accent"
              onClick={() => load(r)}
            >
              {r}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 border-b border-ink-800 px-3 py-2 font-mono text-xs text-ink-400">
          {parent ? (
            <button type="button" className="hover:text-accent" onClick={() => load(parent)}>
              ↑ Up
            </button>
          ) : null}
          <span className="truncate">{path}</span>
        </div>

        {error ? <div className="bg-danger/10 px-4 py-2 text-sm text-danger">{error}</div> : null}

        <div className="min-h-0 flex-1 overflow-auto">
          {loading ? (
            <p className="p-4 text-sm text-ink-500">Loading…</p>
          ) : (
            <ul className="divide-y divide-ink-800">
              {entries.map((e) => (
                <li key={e.path}>
                  <button
                    type="button"
                    className={`flex w-full items-center justify-between px-4 py-2 text-left text-sm hover:bg-ink-800 ${
                      selected === e.path ? "bg-ink-800 text-accent" : ""
                    }`}
                    onDoubleClick={() => {
                      if (e.is_dir) load(e.path);
                      else if (kind !== "dir") {
                        onSelect(e.path);
                        onClose();
                      }
                    }}
                    onClick={() => {
                      if (e.is_dir) {
                        if (kind === "dir" || kind === "any") setSelected(e.path);
                      } else if (kind !== "dir") {
                        setSelected(e.path);
                      }
                    }}
                  >
                    <span className="font-mono text-xs">
                      {e.is_dir ? "[dir] " : ""}
                      {e.name}
                      {e.is_dir ? "/" : ""}
                    </span>
                    {e.is_dir ? (
                      <button
                        type="button"
                        className="text-xs text-accent"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          load(e.path);
                        }}
                      >
                        Open
                      </button>
                    ) : null}
                  </button>
                </li>
              ))}
              {entries.length === 0 ? (
                <li className="px-4 py-6 text-sm text-ink-500">No matching entries here.</li>
              ) : null}
            </ul>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-ink-800 px-4 py-3">
          <div className="flex items-center gap-2">
            {allowUpload ? (
              <label className="cursor-pointer rounded-md border border-ink-600 px-3 py-1.5 text-sm hover:border-accent">
                Upload…
                <input
                  type="file"
                  className="hidden"
                  onChange={(e) => onUpload(e.target.files?.[0] || null)}
                />
              </label>
            ) : null}
            {showPhone ? (
              <button
                type="button"
                className="rounded-md border border-ink-600 px-3 py-1.5 text-sm hover:border-accent"
                onClick={() => setPhoneOpen(true)}
              >
                From phone…
              </button>
            ) : null}
            <span className="max-w-xs truncate font-mono text-[10px] text-ink-500">
              {selected || (kind === "dir" ? path : "Select a file")}
            </span>
          </div>
          <div className="flex gap-2">
            <button type="button" className="rounded-md px-3 py-1.5 text-sm text-ink-400" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              disabled={!canConfirm}
              className="rounded-md bg-accent px-4 py-1.5 text-sm font-semibold text-ink-950 disabled:opacity-40"
              onClick={() => {
                const chosen = selected || (kind === "dir" || kind === "any" ? path : null);
                if (!chosen) return;
                onSelect(chosen);
                onClose();
              }}
            >
              Select
            </button>
          </div>
        </div>
      </div>

      <MobileUploadDialog
        open={phoneOpen}
        onClose={() => setPhoneOpen(false)}
        onSelect={(p) => {
          onSelect(p);
          setPhoneOpen(false);
          onClose();
        }}
      />
    </div>
  );
}
