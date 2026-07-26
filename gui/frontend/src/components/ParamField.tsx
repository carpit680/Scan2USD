import { useState, type ReactNode } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import type { ParamDef } from "../api/client";
import { useHelp } from "../state/HelpContext";
import { PathPickerModal } from "./PathPickerModal";

export function Tip({ text, anchor }: { text: string; anchor?: string | null }) {
  const { openHelp } = useHelp();
  return (
    <Tooltip.Provider delayDuration={200}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          <button
            type="button"
            className="ml-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-ink-600 text-[10px] text-ink-400 hover:border-accent hover:text-accent"
            aria-label="Help"
          >
            ?
          </button>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side="top"
            className="z-50 max-w-sm rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-xs leading-relaxed text-ink-100 shadow-xl"
          >
            <p>{text}</p>
            {anchor ? (
              <button
                type="button"
                className="mt-1 inline-block text-accent hover:underline"
                onClick={() => openHelp(anchor)}
              >
                Learn more
              </button>
            ) : null}
            <Tooltip.Arrow className="fill-ink-700" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

function TagsInput({
  value,
  onChange,
}: {
  value: string[];
  onChange: (v: string[]) => void;
}) {
  const [draft, setDraft] = useState("");
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {value.map((tag) => (
          <button
            key={tag}
            type="button"
            className="rounded-full border border-ink-600 bg-ink-950 px-2 py-0.5 text-xs text-ink-200 hover:border-danger"
            onClick={() => onChange(value.filter((t) => t !== tag))}
            title="Remove"
          >
            {tag} ×
          </button>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm"
          value={draft}
          placeholder="Add a tag, press Enter"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              const t = draft.trim();
              if (t && !value.includes(t)) onChange([...value, t]);
              setDraft("");
            }
          }}
        />
        <button
          type="button"
          className="rounded-md border border-ink-600 px-3 text-sm"
          onClick={() => {
            const t = draft.trim();
            if (t && !value.includes(t)) onChange([...value, t]);
            setDraft("");
          }}
        >
          Add
        </button>
      </div>
    </div>
  );
}

export function ParamField({
  param,
  value,
  onChange,
}: {
  param: ParamDef;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const id = param.id;
  const widget = param.widget || (param.type === "path" ? "path" : param.type === "bool" ? "toggle" : "text");
  const common =
    "w-full rounded-md border border-ink-700 bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-accent";

  let control: ReactNode;

  if (widget === "toggle" || param.type === "bool") {
    control = (
      <button
        type="button"
        role="switch"
        aria-checked={Boolean(value)}
        id={id}
        onClick={() => onChange(!value)}
        className={`relative h-7 w-12 rounded-full transition ${
          value ? "bg-accent" : "bg-ink-700"
        }`}
      >
        <span
          className={`absolute top-0.5 h-6 w-6 rounded-full bg-ink-100 transition ${
            value ? "left-5" : "left-0.5"
          }`}
        />
      </button>
    );
  } else if (widget === "select" || (param.enum && param.enum.length)) {
    control = (
      <select
        id={id}
        className={common}
        value={String(value ?? param.default ?? "")}
        onChange={(e) => onChange(e.target.value)}
      >
        {(param.enum || []).map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    );
  } else if (widget === "slider" && param.min != null && param.max != null) {
    const num = typeof value === "number" ? value : Number(param.default ?? param.min);
    const step = param.step ?? (param.type === "int" ? 1 : 0.01);
    control = (
      <div className="space-y-1">
        <div className="flex items-center justify-between text-xs text-ink-400">
          <span>{param.min}</span>
          <span className="font-mono text-accent">{Number.isFinite(num) ? num : "—"}</span>
          <span>{param.max}</span>
        </div>
        <input
          id={id}
          type="range"
          min={param.min}
          max={param.max}
          step={step}
          className="w-full accent-accent"
          value={Number.isFinite(num) ? num : param.min}
          onChange={(e) => {
            const v = param.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
            onChange(v);
          }}
        />
        <input
          type="number"
          className={common}
          step={step}
          min={param.min}
          max={param.max}
          value={Number.isFinite(num) ? num : ""}
          onChange={(e) => {
            const t = e.target.value;
            if (t === "") onChange(null);
            else onChange(param.type === "int" ? parseInt(t, 10) : parseFloat(t));
          }}
        />
      </div>
    );
  } else if (widget === "tags" || param.type === "string_list") {
    const list = Array.isArray(value) ? (value as string[]) : [];
    control = <TagsInput value={list} onChange={onChange} />;
  } else if (widget === "json" || param.type === "json") {
    control = (
      <textarea
        id={id}
        rows={3}
        className={`${common} font-mono text-xs`}
        value={
          typeof value === "string"
            ? value
            : JSON.stringify(value ?? param.default ?? {}, null, 2)
        }
        onChange={(e) => {
          try {
            onChange(JSON.parse(e.target.value));
          } catch {
            onChange(e.target.value);
          }
        }}
      />
    );
  } else if (widget === "path" || param.type === "path") {
    control = (
      <div className="flex gap-2">
        <input
          id={id}
          type="text"
          className={`${common} font-mono text-xs`}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
          placeholder="Click Browse or type a path"
        />
        <button
          type="button"
          className="shrink-0 rounded-md border border-ink-600 px-3 text-sm hover:border-accent hover:text-accent"
          onClick={() => setPickerOpen(true)}
        >
          Browse…
        </button>
        <PathPickerModal
          open={pickerOpen}
          title={`Choose ${param.label}`}
          kind={param.path_kind || "any"}
          ext={param.path_ext}
          initialPath={typeof value === "string" ? value : null}
          allowUpload={param.path_kind === "file"}
          allowPhoneUpload={
            param.config_path === "video_path" ||
            Boolean(param.path_ext && /mp4|mov|webm|mkv|m4v|video/i.test(param.path_ext))
          }
          onClose={() => setPickerOpen(false)}
          onSelect={(p) => onChange(p)}
        />
      </div>
    );
  } else if (param.type === "int" || param.type === "float" || widget === "number") {
    control = (
      <input
        id={id}
        type="number"
        step={param.step ?? (param.type === "float" ? "any" : "1")}
        min={param.min ?? undefined}
        max={param.max ?? undefined}
        className={common}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => {
          const t = e.target.value;
          if (t === "") onChange(null);
          else onChange(param.type === "int" ? parseInt(t, 10) : parseFloat(t));
        }}
      />
    );
  } else {
    control = (
      <input
        id={id}
        type="text"
        className={common}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : e.target.value)}
      />
    );
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center">
        <label htmlFor={id} className="text-sm font-medium text-ink-300">
          {param.label}
          {param.required ? <span className="text-danger"> *</span> : null}
        </label>
        <Tip text={param.tooltip} anchor={param.guide_anchor} />
      </div>
      {control}
    </div>
  );
}
