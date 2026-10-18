import { useId } from "react";
import { cx } from "../../utils/cx";
import { focusRing } from "./styles";

/** Labelled on/off switch (role="switch"). */
export function Toggle({ label, description, checked, onChange, disabled = false }) {
  const id = useId();
  const labelId = `${id}-label`;
  const descriptionId = `${id}-description`;
  return (
    <div className={cx("flex items-start justify-between gap-4", disabled && "opacity-50")}>
      <div className="min-w-0">
        <label id={labelId} htmlFor={id} className="block text-[13px] leading-5 text-zinc-300">
          {label}
        </label>
        {description && (
          <p id={descriptionId} className="mt-0.5 text-xs leading-snug text-zinc-500">
            {description}
          </p>
        )}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={labelId}
        aria-describedby={description ? descriptionId : undefined}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cx(
          "relative mt-0.5 inline-flex h-[18px] w-8 shrink-0 cursor-pointer items-center rounded-full transition-colors",
          checked ? "bg-violet-600" : "bg-zinc-700 hover:bg-zinc-600",
          focusRing
        )}
      >
        <span
          aria-hidden="true"
          className={cx(
            "h-3.5 w-3.5 rounded-full bg-white shadow-sm transition-transform",
            checked ? "translate-x-4" : "translate-x-0.5"
          )}
        />
      </button>
    </div>
  );
}
