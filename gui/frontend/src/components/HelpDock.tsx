import { Link } from "react-router-dom";
import { GuideMarkdown } from "./GuideMarkdown";
import { useHelp } from "../state/HelpContext";

export function HelpDock() {
  const { open, sectionId, toc, sections, loading, closeHelp, setSectionId } = useHelp();

  if (!open) return null;

  const section = sections.find((s) => s.id === sectionId) || sections[0];

  return (
    <>
      {/* Mobile overlay sheet */}
      <div
        className="fixed inset-0 z-40 bg-black/50 lg:hidden"
        onClick={closeHelp}
        aria-hidden
      />
      <aside
        className="fixed inset-x-0 bottom-0 z-50 flex max-h-[70vh] flex-col border-t border-ink-700 bg-ink-900 shadow-2xl lg:sticky lg:top-[57px] lg:z-auto lg:h-[calc(100vh-57px)] lg:max-h-none lg:w-[360px] lg:shrink-0 lg:border-l lg:border-t-0 lg:shadow-none"
        aria-label="Help"
      >
        <div className="flex items-center justify-between border-b border-ink-800 px-3 py-2">
          <div>
            <div className="font-display text-sm font-semibold text-ink-100">Help</div>
            <div className="text-[10px] uppercase tracking-wider text-ink-500">Side-by-side guide</div>
          </div>
          <button
            type="button"
            onClick={closeHelp}
            className="rounded-md px-2 py-1 text-sm text-ink-400 hover:bg-ink-800 hover:text-ink-100"
          >
            Close
          </button>
        </div>

        <div className="border-b border-ink-800 px-3 py-2">
          <label className="mb-1 block text-[11px] text-ink-500">Topic</label>
          <select
            className="w-full rounded-md border border-ink-700 bg-ink-950 px-2 py-1.5 text-sm text-ink-100"
            value={section?.id || sectionId}
            onChange={(e) => setSectionId(e.target.value)}
          >
            {toc.map((t) => (
              <option key={t.id} value={t.id}>
                {t.title}
              </option>
            ))}
          </select>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          {loading ? (
            <p className="text-sm text-ink-500">Loading guide…</p>
          ) : section ? (
            <GuideMarkdown body={section.body} />
          ) : (
            <p className="text-sm text-ink-500">No guide content.</p>
          )}
        </div>

        <div className="border-t border-ink-800 px-3 py-2">
          <Link
            to={`/guide${section ? `#${section.id}` : ""}`}
            className="text-xs text-accent hover:underline"
          >
            Open full Guide →
          </Link>
        </div>
      </aside>
    </>
  );
}
