export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 p-4"
      role="presentation"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        className="w-full max-w-md rounded-xl border border-ink-700 bg-ink-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-ink-800 px-4 py-3">
          <h2 id="confirm-dialog-title" className="font-display text-lg font-semibold text-ink-100">
            {title}
          </h2>
        </div>
        <div className="px-4 py-4 text-sm text-ink-300">{message}</div>
        <div className="flex justify-end gap-2 border-t border-ink-800 px-4 py-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md border border-ink-600 px-3 py-1.5 text-sm text-ink-200 hover:border-ink-400"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={
              danger
                ? "rounded-md bg-danger px-3 py-1.5 text-sm font-semibold text-white hover:bg-danger/90"
                : "rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-ink-950 hover:bg-accent/90"
            }
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
