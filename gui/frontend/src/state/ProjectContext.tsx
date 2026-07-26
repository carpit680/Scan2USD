import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  client,
  type JobSnapshot,
  type Schema,
  LS_CONFIG,
  LS_CWD,
  LS_JOB,
} from "../api/client";

interface ProjectState {
  loaded: boolean;
  configPath: string | null;
  cwd: string;
  raw: Record<string, unknown> | null;
  config: Record<string, unknown> | null;
  yamlText: string | null;
  workspace: Record<string, unknown> | null;
  schema: Schema | null;
  activeJob: JobSnapshot | null;
  jobLines: string[];
  error: string | null;
  refreshProject: () => Promise<void>;
  openProject: (configPath: string, cwd?: string) => Promise<void>;
  saveRaw: (raw: Record<string, unknown>) => Promise<void>;
  saveYamlText: (text: string) => Promise<void>;
  savePatch: (patch: Record<string, unknown>) => Promise<void>;
  runCommand: (command: string, options?: Record<string, unknown>) => Promise<void>;
  cancelJob: () => Promise<void>;
  setError: (msg: string | null) => void;
}

const Ctx = createContext<ProjectState | null>(null);

function jobStillActive(status: string | undefined): boolean {
  return status === "running" || status === "queued";
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [loaded, setLoaded] = useState(false);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [cwd, setCwd] = useState(".");
  const [raw, setRaw] = useState<Record<string, unknown> | null>(null);
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [yamlText, setYamlText] = useState<string | null>(null);
  const [workspace, setWorkspace] = useState<Record<string, unknown> | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [activeJob, setActiveJob] = useState<JobSnapshot | null>(null);
  const [jobLines, setJobLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const closeJobStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const openJobStream = useCallback(
    (jobId: string, after: number) => {
      closeJobStream();
      const es = new EventSource(`/api/jobs/${jobId}/events?after=${after}`);
      eventSourceRef.current = es;
      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "log") {
          setJobLines((prev) => [...prev, data.line]);
        } else if (data.type === "done") {
          setActiveJob((prev) =>
            prev && prev.id === jobId
              ? { ...prev, status: data.status, exit_code: data.exit_code }
              : prev,
          );
          es.close();
          if (eventSourceRef.current === es) eventSourceRef.current = null;
        }
      };
      es.onerror = () => {
        // Don't mark the job failed — the process may still be running.
        // Snapshot status; user can refresh again to reconnect.
        es.close();
        if (eventSourceRef.current === es) eventSourceRef.current = null;
        void client
          .getJob(jobId)
          .then((snap) => {
            setActiveJob(snap);
            if (jobStillActive(snap.status)) {
              setJobLines((prev) => [
                ...prev,
                "[gui] log stream disconnected — refresh the page to reconnect (job still running).",
              ]);
            }
          })
          .catch(() => {
            /* job gone */
          });
      };
    },
    [closeJobStream],
  );

  const attachJob = useCallback(
    async (jobId: string) => {
      const logs = await client.jobLogs(jobId, 0);
      const { lines, next, ...snap } = logs;
      setActiveJob(snap);
      setJobLines(Array.isArray(lines) ? lines : []);
      localStorage.setItem(LS_JOB, jobId);
      if (jobStillActive(snap.status)) {
        openJobStream(jobId, typeof next === "number" ? next : lines.length);
      } else {
        closeJobStream();
      }
    },
    [closeJobStream, openJobStream],
  );

  useEffect(() => {
    client.schema().then(setSchema).catch((e) => setError(String(e.message || e)));
    const restore = async () => {
      try {
        const p = await client.getProject();
        if (p.loaded) {
          setLoaded(true);
          setConfigPath(String(p.config_path));
          setCwd(String(p.cwd || "."));
          setRaw((p.raw as Record<string, unknown>) || null);
          setConfig((p.config as Record<string, unknown>) || null);
          setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : null);
          setWorkspace((p.workspace as Record<string, unknown>) || null);
        } else {
          const last = localStorage.getItem(LS_CONFIG);
          const lastCwd = localStorage.getItem(LS_CWD) || undefined;
          if (last) {
            try {
              const opened = await client.openProject(last, lastCwd);
              setLoaded(true);
              setConfigPath(String(opened.config_path));
              setCwd(String(opened.cwd || lastCwd || "."));
              setRaw((opened.raw as Record<string, unknown>) || null);
              setConfig((opened.config as Record<string, unknown>) || null);
              setYamlText(typeof opened.yaml_text === "string" ? opened.yaml_text : null);
              setWorkspace((opened.workspace as Record<string, unknown>) || null);
            } catch {
              /* stale path */
            }
          }
        }
      } catch {
        /* ignore */
      }

      // Restore job console from server buffer (survives page refresh).
      try {
        const { jobs } = await client.listJobs();
        const remembered = localStorage.getItem(LS_JOB);
        const pick =
          (remembered && jobs.find((j) => j.id === remembered)) ||
          jobs.find((j) => jobStillActive(j.status)) ||
          jobs[0];
        if (pick) {
          await attachJob(pick.id);
        }
      } catch {
        /* no jobs API / empty */
      }
    };
    void restore();
    return () => closeJobStream();
  }, [attachJob, closeJobStream]);

  const refreshProject = useCallback(async () => {
    const p = await client.getProject();
    setLoaded(Boolean(p.loaded));
    setConfigPath(p.config_path ? String(p.config_path) : null);
    setCwd(String(p.cwd || "."));
    setRaw((p.raw as Record<string, unknown>) || null);
    setConfig((p.config as Record<string, unknown>) || null);
    setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : null);
    setWorkspace((p.workspace as Record<string, unknown>) || null);
  }, []);

  const openProject = useCallback(async (path: string, workCwd?: string) => {
    setError(null);
    const p = await client.openProject(path, workCwd);
    setLoaded(true);
    setConfigPath(String(p.config_path));
    setCwd(String(p.cwd || workCwd || "."));
    setRaw((p.raw as Record<string, unknown>) || null);
    setConfig((p.config as Record<string, unknown>) || null);
    setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : null);
    setWorkspace((p.workspace as Record<string, unknown>) || null);
    localStorage.setItem(LS_CONFIG, String(p.config_path));
    localStorage.setItem(LS_CWD, String(p.cwd || workCwd || "."));
  }, []);

  const saveRaw = useCallback(async (next: Record<string, unknown>) => {
    const p = await client.putConfig({ raw: next });
    setRaw(p.raw);
    setConfig(p.config);
    setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : null);
    if (p.workspace) setWorkspace(p.workspace as Record<string, unknown>);
  }, []);

  const saveYamlText = useCallback(async (text: string) => {
    const p = await client.putConfig({ yaml_text: text });
    setRaw(p.raw);
    setConfig(p.config);
    setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : text);
    if (p.workspace) setWorkspace(p.workspace as Record<string, unknown>);
  }, []);

  const savePatch = useCallback(async (patch: Record<string, unknown>) => {
    const p = await client.putConfig({ patch });
    setRaw(p.raw);
    setConfig(p.config);
    setYamlText(typeof p.yaml_text === "string" ? p.yaml_text : null);
    if (p.workspace) setWorkspace(p.workspace as Record<string, unknown>);
  }, []);

  const runCommand = useCallback(
    async (command: string, options: Record<string, unknown> = {}) => {
      setError(null);
      closeJobStream();
      setJobLines([]);
      const job = await client.startJob(command, options);
      setActiveJob(job);
      localStorage.setItem(LS_JOB, job.id);
      openJobStream(job.id, 0);
    },
    [closeJobStream, openJobStream],
  );

  const cancelJob = useCallback(async () => {
    if (!activeJob) return;
    try {
      const snap = await client.cancelJob(activeJob.id);
      setActiveJob(snap);
      setJobLines((prev) => [...prev, "[gui] stop requested from UI"]);
    } catch (e) {
      setError(String((e as Error).message || e));
    }
  }, [activeJob]);

  const value = useMemo(
    () => ({
      loaded,
      configPath,
      cwd,
      raw,
      config,
      yamlText,
      workspace,
      schema,
      activeJob,
      jobLines,
      error,
      refreshProject,
      openProject,
      saveRaw,
      saveYamlText,
      savePatch,
      runCommand,
      cancelJob,
      setError,
    }),
    [
      loaded,
      configPath,
      cwd,
      raw,
      config,
      yamlText,
      workspace,
      schema,
      activeJob,
      jobLines,
      error,
      refreshProject,
      openProject,
      saveRaw,
      saveYamlText,
      savePatch,
      runCommand,
      cancelJob,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useProject() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useProject outside provider");
  return ctx;
}
