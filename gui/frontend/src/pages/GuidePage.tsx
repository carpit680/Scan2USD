import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { GuideMarkdown } from "../components/GuideMarkdown";
import { useHelp } from "../state/HelpContext";

export function GuidePage() {
  const { toc, sections, setSectionId, openHelp } = useHelp();
  const loc = useLocation();

  useEffect(() => {
    const id = loc.hash.replace("#", "");
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [loc.hash, sections]);

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_1fr]">
      <aside className="space-y-1 lg:sticky lg:top-24 lg:self-start">
        <h1 className="font-display text-2xl font-semibold">Guide</h1>
        <p className="pb-2 text-xs text-ink-500">
          Full archive. Prefer the header <strong className="text-ink-300">Help</strong> button for
          side-by-side help while you work.
        </p>
        <button
          type="button"
          onClick={() => openHelp()}
          className="mb-3 w-full rounded-md border border-accent/40 bg-accent/10 px-3 py-2 text-sm text-accent hover:bg-accent/20"
        >
          Open side Help dock
        </button>
        {toc.map((t) => (
          <a
            key={t.id}
            href={`#${t.id}`}
            onClick={() => setSectionId(t.id)}
            className="block rounded-md px-2 py-1 text-sm text-ink-400 hover:bg-ink-900 hover:text-accent"
          >
            {t.title}
          </a>
        ))}
      </aside>
      <div className="space-y-10">
        {sections.map((s) => (
          <article key={s.id} id={s.id} className="scroll-mt-28">
            <GuideMarkdown body={s.body} />
          </article>
        ))}
      </div>
    </div>
  );
}
