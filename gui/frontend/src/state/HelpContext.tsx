import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocation } from "react-router-dom";
import { client } from "../api/client";

const LS_HELP_OPEN = "scan2usd.gui.helpOpen";

export type GuideSection = { id: string; title: string; body: string };
export type GuideTocItem = { id: string; title: string };

const ROUTE_SECTION: Record<string, string> = {
  "/": "getting-started",
  "/config": "paths",
  "/pipeline": "sample-workflow",
  "/review": "segmentation",
  "/doctor": "setup",
  "/commands": "hybrid",
  "/artifacts": "sample-workflow",
  "/guide": "getting-started",
};

function sectionForPath(pathname: string): string {
  return ROUTE_SECTION[pathname] || "getting-started";
}

interface HelpState {
  open: boolean;
  sectionId: string;
  pinned: boolean;
  toc: GuideTocItem[];
  sections: GuideSection[];
  loading: boolean;
  openHelp: (section?: string | null) => void;
  closeHelp: () => void;
  toggleHelp: () => void;
  setSectionId: (id: string) => void;
}

const Ctx = createContext<HelpState | null>(null);

export function HelpProvider({ children }: { children: ReactNode }) {
  const location = useLocation();
  const [open, setOpen] = useState(() => localStorage.getItem(LS_HELP_OPEN) === "1");
  const [sectionId, setSectionIdState] = useState(() => sectionForPath(location.pathname));
  const [pinned, setPinned] = useState(false);
  const [toc, setToc] = useState<GuideTocItem[]>([]);
  const [sections, setSections] = useState<GuideSection[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    client
      .guide()
      .then((g) => {
        setToc(g.toc);
        setSections(g.sections);
      })
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    localStorage.setItem(LS_HELP_OPEN, open ? "1" : "0");
  }, [open]);

  // On navigation: follow the page's default topic again
  useEffect(() => {
    setPinned(false);
    setSectionIdState(sectionForPath(location.pathname));
  }, [location.pathname]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const openHelp = useCallback((section?: string | null) => {
    setOpen(true);
    if (section) {
      setSectionIdState(section);
      setPinned(true);
    }
  }, []);

  const closeHelp = useCallback(() => setOpen(false), []);

  const toggleHelp = useCallback(() => {
    setOpen((v) => !v);
  }, []);

  const setSectionId = useCallback((id: string) => {
    setSectionIdState(id);
    setPinned(true);
  }, []);

  const value = useMemo(
    () => ({
      open,
      sectionId,
      pinned,
      toc,
      sections,
      loading,
      openHelp,
      closeHelp,
      toggleHelp,
      setSectionId,
    }),
    [open, sectionId, pinned, toc, sections, loading, openHelp, closeHelp, toggleHelp, setSectionId],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useHelp() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useHelp outside HelpProvider");
  return ctx;
}
