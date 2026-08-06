import { useCallback, useEffect, useRef, useState } from "react";
import { client } from "../api/client";
import { useProject } from "../state/ProjectContext";

interface PreviewStatus {
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

  const show = useCallback(async () => {
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
          parseSplatData: (type: number, input: Uint8Array) => Promise<unknown>;
          SplatFileType: Record<string, number>;
        };
        SplatUtils: { createSplat: (data: unknown) => Promise<unknown> };
      };

      disposer.current?.();
      container.current.innerHTML = "";
      const viewer = createViewer("scan2usd-preview", container.current, {});

      setProgress("downloading splat…");
      const response = await fetch("/api/quality/preview.ply");
      if (!response.ok) throw new Error(await response.text());
      const buffer = new Uint8Array(await response.arrayBuffer());

      setProgress("parsing…");
      const data = await SplatLoader.parseSplatData(
        SplatLoader.SplatFileType.PLY,
        buffer,
      );
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
          onClick={() => void show()}
        >
          {loading ? progress || "loading…" : "Show scene"}
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
