export type ParamType =
  | "string"
  | "int"
  | "float"
  | "bool"
  | "path"
  | "enum"
  | "string_list"
  | "json";

export type WidgetType =
  | "slider"
  | "number"
  | "select"
  | "toggle"
  | "path"
  | "tags"
  | "text"
  | "json";

export interface ParamDef {
  id: string;
  label: string;
  type: ParamType;
  group: string;
  tooltip: string;
  default: unknown;
  config_path: string | null;
  command: string | null;
  enum: string[] | null;
  min: number | null;
  max: number | null;
  step?: number | null;
  guide_anchor: string | null;
  required: boolean;
  widget?: WidgetType;
  help_level?: "essential" | "advanced";
  path_kind?: "file" | "dir" | "any";
  path_ext?: string | null;
  /** Key into project workspace.paths (from open scene config). */
  default_from?: string | null;
  /** Dotted path into resolved scene config (e.g. reconstruction.splat_cleanup.outlier_std). */
  config_default_from?: string | null;
}

export interface CommandDef {
  id: string;
  label: string;
  category: string;
  description: string;
  guide_anchor?: string;
  needs_config: boolean;
  dangerous?: boolean;
  job_kind?: string;
  script?: string;
  options: ParamDef[];
}

export interface Schema {
  config_params: ParamDef[];
  config_groups: { id: string; label: string }[];
  commands: CommandDef[];
  pipeline_stages: StageDef[];
  legacy_stages: { id: string; label: string; command: string }[];
}

export interface StageDef {
  id: string;
  label: string;
  command: string | null;
  href?: string;
  description: string;
  guide_anchor?: string;
}

export interface JobSnapshot {
  id: string;
  command: string;
  argv: string[];
  cwd: string;
  status: string;
  exit_code: number | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  line_count: number;
}

export interface FsBrowseResult {
  path: string;
  parent: string | null;
  kind: string;
  ext: string | null;
  entries: { name: string; path: string; is_dir: boolean; suffix: string; size: number | null }[];
  roots: string[];
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string>) };
  if (init?.body && !(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, { cache: "no-store", ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const client = {
  health: () => api<{ ok: boolean }>("/api/health"),
  schema: () => api<Schema>("/api/schema"),
  getProject: () => api<Record<string, unknown>>("/api/project"),
  openProject: (config_path: string, cwd?: string) =>
    api<Record<string, unknown>>("/api/project", {
      method: "PUT",
      body: JSON.stringify({ config_path, cwd }),
    }),
  createProject: (path: string, template?: string, cwd?: string) =>
    api<Record<string, unknown>>("/api/project/create", {
      method: "POST",
      body: JSON.stringify({ path, template, cwd }),
    }),
  getConfig: () =>
    api<{
      raw: Record<string, unknown>;
      config: Record<string, unknown>;
      yaml_text?: string;
      workspace?: Record<string, unknown>;
    }>("/api/config"),
  putConfig: (body: {
    raw?: Record<string, unknown>;
    patch?: Record<string, unknown>;
    yaml_text?: string;
  }) =>
    api<{
      raw: Record<string, unknown>;
      config: Record<string, unknown>;
      yaml_text?: string;
      workspace?: Record<string, unknown>;
    }>("/api/config", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  startJob: (command: string, options: Record<string, unknown> = {}) =>
    api<JobSnapshot>("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ command, options }),
    }),
  listJobs: () => api<{ jobs: JobSnapshot[] }>("/api/jobs"),
  getJob: (id: string) => api<JobSnapshot>(`/api/jobs/${id}`),
  jobLogs: (id: string, after = 0) =>
    api<JobSnapshot & { lines: string[]; next: number }>(
      `/api/jobs/${id}/logs?after=${encodeURIComponent(String(after))}`,
    ),
  cancelJob: (id: string) =>
    api<JobSnapshot>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  pipelineState: () => api<Record<string, unknown>>("/api/pipeline/state"),
  doctor: () => api<Record<string, unknown>>("/api/doctor"),
  guide: () =>
    api<{
      toc: { id: string; title: string }[];
      sections: { id: string; title: string; body: string }[];
    }>("/api/guide"),
  reviewInstances: () => api<Record<string, unknown>>("/api/review/instances"),
  metricScene: () => api<Record<string, unknown>>("/api/metric/scene"),
  reviewInstance: (id: string) => api<Record<string, unknown>>(`/api/review/instances/${id}`),
  reclassifyPreview: (id: string, className: string) =>
    api<Record<string, unknown>>(
      `/api/review/instances/${id}/reclassify-preview?class_name=${encodeURIComponent(className)}`,
    ),
  updateInstance: (id: string, body: Record<string, unknown>) =>
    api<Record<string, unknown>>(`/api/review/instances/${id}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  mergeInstances: (primaryId: string, sourceIds: string[]) =>
    api<Record<string, unknown>>(`/api/review/instances/${primaryId}/merge`, {
      method: "POST",
      body: JSON.stringify({ source_ids: sourceIds }),
    }),
  unmergeInstances: (primaryId: string, sourceIds?: string[]) =>
    api<Record<string, unknown>>(`/api/review/instances/${primaryId}/unmerge`, {
      method: "POST",
      body: JSON.stringify({ source_ids: sourceIds ?? null }),
    }),
  deleteMask: (instanceId: string, maskName: string) =>
    api<Record<string, unknown>>(
      `/api/review/instances/${instanceId}/masks/${encodeURIComponent(maskName)}`,
      { method: "DELETE" },
    ),
  deleteMasks: (instanceId: string, maskNames: string[]) =>
    api<Record<string, unknown>>(`/api/review/instances/${instanceId}/masks/delete`, {
      method: "POST",
      body: JSON.stringify({ mask_names: maskNames }),
    }),
  artifactRoots: () => api<{ roots: { id: string; path: string; exists: boolean }[] }>("/api/artifacts/roots"),
  artifactList: (root: string, path = "") =>
    api<{
      entries: { name: string; path: string; is_dir: boolean; size: number | null; suffix: string }[];
      path: string;
      exists: boolean;
    }>(`/api/artifacts/list?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`),
  fsRoots: () => api<{ roots: { path: string; label: string; is_home: boolean }[] }>("/api/fs/roots"),
  fsBrowse: (path?: string, kind: string = "any", ext?: string) => {
    const q = new URLSearchParams();
    if (path) q.set("path", path);
    q.set("kind", kind);
    if (ext) q.set("ext", ext);
    return api<FsBrowseResult>(`/api/fs/browse?${q}`);
  },
  fsUpload: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return api<{ path: string; filename: string }>("/api/fs/upload", { method: "POST", body: fd });
  },
  mobileCreateSession: () =>
    api<{
      id: string;
      token: string;
      url: string;
      urls: string[];
      lan_ips: string[];
      expires_at: number;
      qr_svg: string;
      port: number;
    }>("/api/mobile/sessions", { method: "POST" }),
  mobileSessionStatus: (id: string, token: string) =>
    api<{
      id: string;
      status: string;
      path: string | null;
      filename: string | null;
      expires_at: number;
    }>(`/api/mobile/sessions/${encodeURIComponent(id)}?token=${encodeURIComponent(token)}`),
};

export function overlayUrl(absPath: string, cacheKey?: string | number): string {
  const q = new URLSearchParams({ path: absPath });
  if (cacheKey != null && cacheKey !== "") q.set("v", String(cacheKey));
  return `/api/review/overlay?${q}`;
}

export function artifactFileUrl(root: string, path: string): string {
  return `/api/artifacts/file?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`;
}

export const LS_CONFIG = "scan2usd.gui.lastConfig";
export const LS_CWD = "scan2usd.gui.lastCwd";
export const LS_JOB = "scan2usd.gui.lastJobId";
