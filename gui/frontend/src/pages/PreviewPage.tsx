import { useCallback, useEffect, useRef, useState } from "react";
import { client } from "../api/client";
import { useProject } from "../state/ProjectContext";

interface PreviewMeta {
  camera_position: [number, number, number];
  look_at: [number, number, number];
  up: [number, number, number];
}

interface PreviewStatus {
  has_small: boolean;
  small_bytes: number;
  exists: boolean;
  path: string;
  bytes: number;
  stale: boolean;
  meta: PreviewMeta | null;
}

/**
 * Browser preview of the cleaned splat, before any USD or Isaac step.
 *
 * Checking a cleanup setting used to mean launching Isaac: ~80 seconds of
 * startup and the whole GPU held, which also blocks training. Here the splat
 * renders in the browser's own GPU — which is also why it works from a remote
 * machine: the server only ships a static file.
 *
 * Uses @mkkellogg/gaussian-splats-3d rather than aholo: aholo parses inside a
 * worker whose failure leaves the promise permanently pending, so a broken
 * environment looked identical to a slow parse. This engine loads our PLY
 * directly and reports errors as errors.
 */
export function PreviewPage() {
  const { loaded, setError, runCommand } = useProject();
  const container = useRef<HTMLDivElement>(null);
  const disposer = useRef<(() => Promise<void> | void) | null>(null);
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

  // Poll rather than making the reader press a button. The status is a stat()
  // on two files, and the flag it drives — "stale" — is only useful if it
  // clears itself when a rebuild finishes. Requiring a click to find out
  // whether what you are looking at is current defeats the point of the flag.
  useEffect(() => {
    if (!loaded) return;
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [loaded, refresh]);

  useEffect(
    () => () => {
      void disposer.current?.();
    },
    [],
  );

  const show = useCallback(
    async (small = false) => {
      if (!container.current) return;
      setLoading(true);
      setProgress("loading engine…");
      try {
        const GS = await import("@mkkellogg/gaussian-splats-3d");

        // Tear down the previous viewer best-effort. dispose() removes its own
        // DOM nodes, and racing it against clearing the container throws
        // "removeChild: not a child" — so swallow that class of failure and
        // clear whatever it left behind afterwards.
        try {
          await disposer.current?.();
        } catch {
          /* the old viewer's DOM is going away regardless */
        }
        disposer.current = null;
        container.current.innerHTML = "";
        // Each viewer gets its own mount node, so its internal removals can
        // never collide with a sibling's.
        const mount = document.createElement("div");
        mount.style.width = "100%";
        mount.style.height = "100%";
        mount.style.position = "relative";
        container.current.appendChild(mount);

        const meta = status?.meta;
        const viewer = new GS.Viewer({
          rootElement: mount,
          // The export applies the floor transform, so the scene is Z-up.
          cameraUp: meta?.up ?? [0, 0, 1],
          // A real capture pose partway along the path: the reconstruction is
          // only valid where the camera actually went, and framing the whole
          // scene would start outside the room, in the halo.
          initialCameraPosition: meta?.camera_position ?? [0, -2, 1.5],
          initialCameraLookAt: meta?.look_at ?? [0, 0, 1.2],
          // Avoids needing COOP/COEP headers, which the Vite dev server and the
          // ZeroTier remote path do not set.
          sharedMemoryForWorkers: false,
        });

        setProgress("loading splat…");
        await viewer.addSplatScene(
          `/api/quality/preview.ply${small ? "?small=true" : ""}`,
          {
            // The query string defeats extension sniffing, so say it is a PLY.
            format: GS.SceneFormat.Ply,
            showLoadingUI: true,
            progressiveLoad: true,
          },
        );
        viewer.start();
        disposer.current = async () => {
          try {
            await viewer.dispose();
          } finally {
            mount.remove();
          }
        };
        setProgress("");
      } catch (e) {
        setError(String((e as Error).message || e));
        setProgress("");
      } finally {
        setLoading(false);
      }
    },
    [setError, status],
  );

  if (!loaded) return <p className="text-ink-400">Open a project first.</p>;

  const mb = status?.bytes ? (status.bytes / 1048576).toFixed(1) : null;

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Preview</h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-400">
          The cleaned splat, rendered by your browser — no Isaac, no USD build,
          works remotely. Drag to orbit, right-drag to pan, scroll to zoom.
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
          title="50k Gaussians — loads in seconds; use to sanity-check before the full scene."
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
        {status?.exists ? (
          <span className="text-xs text-ink-500">{mb} MB</span>
        ) : (
          <span className="text-xs text-ink-500">
            no preview yet — build one from the cleaned splat
          </span>
        )}
        {status?.stale ? (
          <span className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-400">
            stale — the splat changed since this was built; rebuild to see it
          </span>
        ) : (
          <span className="text-xs text-emerald-500/80">up to date</span>
        )}
      </div>

      <div
        ref={container}
        className="relative h-[70vh] w-full overflow-hidden rounded border border-ink-800 bg-black"
      />
    </div>
  );
}
