import { useEffect, useState } from "react";
import { client } from "../api/client";

export interface MobileUploadDialogProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
}

export function MobileUploadDialog({ open, onClose, onSelect }: MobileUploadDialogProps) {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [url, setUrl] = useState<string | null>(null);
  const [urls, setUrls] = useState<string[]>([]);
  const [qrSvg, setQrSvg] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setError(null);
    setLoading(true);
    setStatus("pending");
    setUrl(null);
    setQrSvg(null);
    client
      .mobileCreateSession()
      .then((s) => {
        if (cancelled) return;
        setSessionId(s.id);
        setToken(s.token);
        setUrl(s.url);
        setUrls(s.urls || []);
        setQrSvg(s.qr_svg);
      })
      .catch((e) => {
        if (!cancelled) setError(String((e as Error).message || e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open || !sessionId || !token) return;
    const timer = window.setInterval(() => {
      client
        .mobileSessionStatus(sessionId, token)
        .then((s) => {
          setStatus(s.status);
          if (s.status === "completed" && s.path) {
            onSelect(s.path);
            onClose();
          }
          if (s.status === "expired") {
            setError("Upload link expired. Close and open From phone again.");
          }
        })
        .catch((e) => setError(String((e as Error).message || e)));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [open, sessionId, token, onSelect, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-xl border border-ink-700 bg-ink-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-ink-800 px-4 py-3">
          <h2 className="font-display text-lg font-semibold">Upload from phone</h2>
          <button type="button" className="text-ink-400 hover:text-ink-100" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="space-y-3 p-4">
          <p className="text-sm text-ink-400">
            Scan this QR with your phone (same Wi‑Fi). Use <code className="text-ink-300">make gui-lan</code> so
            the API listens on the LAN. Waiting for upload…
          </p>
          {error ? <div className="rounded-md bg-danger/10 px-3 py-2 text-sm text-danger">{error}</div> : null}
          {loading ? <p className="text-sm text-ink-500">Creating session…</p> : null}
          {qrSvg ? (
            <div
              className="mx-auto flex max-w-[220px] justify-center rounded-lg bg-white p-3 [&_svg]:h-auto [&_svg]:w-full"
              dangerouslySetInnerHTML={{ __html: qrSvg }}
            />
          ) : null}
          {url ? (
            <div className="space-y-1">
              <p className="text-[11px] uppercase tracking-wide text-ink-500">Phone URL</p>
              <code className="block break-all rounded-md border border-ink-800 bg-ink-950 px-2 py-1.5 font-mono text-[11px] text-accent">
                {url}
              </code>
              {urls.length > 1 ? (
                <details className="text-xs text-ink-500">
                  <summary className="cursor-pointer hover:text-ink-300">Other LAN addresses</summary>
                  <ul className="mt-1 space-y-1">
                    {urls.map((u) => (
                      <li key={u} className="break-all font-mono text-[10px]">
                        {u}
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}
            </div>
          ) : null}
          <p className="text-xs text-ink-500">
            Status: <span className="text-ink-300">{status}</span>
          </p>
        </div>
      </div>
    </div>
  );
}
