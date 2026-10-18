import { useId } from "react";

/** A titled group of controls in the settings sidebar. */
export function PanelSection({ title, action, children }) {
  const headingId = useId();
  return (
    <section aria-labelledby={headingId} className="px-4 py-4">
      <div className="mb-3.5 flex h-5 items-center justify-between gap-2">
        <h2 id={headingId} className="text-[11px] font-semibold uppercase tracking-[0.08em] text-zinc-500">
          {title}
        </h2>
        {action}
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  );
}
