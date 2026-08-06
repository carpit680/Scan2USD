import { useCallback, useEffect, useRef, useState } from "react";
import { client } from "../api/client";
import { useProject } from "../state/ProjectContext";

interface PreviewStatus {
  has_small: boolean;
  small_bytes: number;
  exists: boolean;
  path: string;
  bytes: number;
  stale: boolean;
}

/**
 * Browser preview of the cleaned splat, before any USD or Isaac step.
 *
 * Checking a cleanup setting used to mean launching Isaac: ~80 seconds of
 * startup and the whole GPU held, which also blocks training. Here the splat is
 * a static file the browser renders, so two settings can be compared quickly and
 * a bad one is caught before it reaches a USD build.
 */
export function PreviewPage() {
  const { loaded, setError, runCommand } = useProject();
  const container = useRef<HTMLDivElement>(null);
  const disposer = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<PreviewStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStatus((await client.previewStatus()) as unknown as PreviewStatus);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, [setError]);

  useEffect(() => {
    if (loaded) void refresh();
  }, [loaded, refresh]);

  useEffect(() => () => disposer.current?.(), []);

  const show = useCallback(async (small = false) => {
    if (!container.current) return;
    setLoading(true);
    setProgress("loading viewer…");
    try {
      // Imported lazily: the engine is large, and most visits to this page are
      // to read the status line rather than render a scene.
      const aholo = await import("@manycore/aholo-viewer");
      const { createViewer, SplatLoader, SplatUtils } = aholo as unknown as {
        createViewer: (name: string, el: HTMLElement, cfg: object) => any;
        SplatLoader: {
          parseSplatData: (
            type: number,
            input: Uint8Array,
            packType?: number,
          ) => Promise<unknown>;
          SplatFileType: Record<string, number>;
          SplatPackType: Record<string, number>;
        };
        SplatUtils: { createSplat: (data: unknown) => Promise<unknown> };
      };

      disposer.current?.();
      container.current.innerHTML = "";
      const viewer = createViewer("scan2usd-preview", container.current, {});

      setProgress("downloading splat…");
      const response = await fetch(`/api/quality/preview.ply${small ? "?small=true" : ""}`);
      if (!response.ok) throw new Error(await response.text());
      const buffer = new Uint8Array(await response.arrayBuffer());

      setProgress(`parsing ${(buffer.length / 1048576).toFixed(0)} MB…`);
      // The engine runs parsing in a module Worker built from a blob URL. When
      // that worker fails to start, the returned promise never settles and the
      // page sits on "parsing" forever with nothing in the console, so bound it
      // and say so rather than hanging.
      const data = await Promise.race([
        SplatLoader.parseSplatData(
          SplatLoader.SplatFileType.PLY,
          buffer,
          SplatLoader.SplatPackType?.Raw,
        ),
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(
                new Error(
                  "Parsing did not finish in 120s. The splat worker most likely " +
                    "failed to start — check the browser console for a Worker or " +
                    "SecurityError. A smaller preview (Max Gaussians in the export " +
                    "tool) will confirm whether it is size-related.",
                ),
              ),
            120000,
          ),
        ),
      ]);
      const splat = await SplatUtils.createSplat(data);
      viewer.getScene().add(splat);
      disposer.current = () => {
        try {
          viewer.destroy?.();
        } catch {
          /* engine already torn down */
        }
      };
      setProgress("");
    } catch (e) {
      setError(String((e as Error).message || e));
      setProgress("");
    } finally {
      setLoading(false);
    }
  }, [setError]);

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  const mb = status?.bytes ? (status.bytes / 1048576).toFixed(1) : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Preview</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          The cleaned splat, in the browser, before any USD or Isaac step. Isaac
          takes about 80 seconds to start and holds the GPU — which also stalls
          training — so this is the faster way to judge a cleanup setting.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800 disabled:opacity-50"
          disabled={!status?.exists || loading}
          onClick={() => void show(false)}
        >
          {loading ? progress || "loading…" : "Show scene"}
        </button>
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800 disabled:opacity-50"
          disabled={!status?.has_small || loading}
          title="50k Gaussians — if this renders and the full one does not, the problem is size, not the file."
          onClick={() => void show(true)}
        >
          Show small (50k)
        </button>
        <button
          className="rounded border border-ink-700 px-3 py-1.5 hover:bg-ink-800"
          onClick={() => runCommand("tool-export-splat-ply", {})}
        >
          {status?.exists ? "Rebuild preview" : "Build preview"}
        </button>
        <button className="text-xs text-ink-400 underline" onClick={() => void refresh()}>
          refresh status
        </button>
        {status?.exists ? (
          <span className="text-xs text-ink-500">{mb} MB</span>
        ) : (
          <span className="text-xs text-ink-500">
            no preview yet — build one from the cleaned splat
          </span>
        )}
        {status?.stale ? (
          <span className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-400">
            stale — the splat changed since this was built
          </span>
        ) : null}
      </div>

      <div
        ref={container}
        className="h-[70vh] w-full rounded border border-ink-800 bg-black"
      />
    </div>
  );
}
