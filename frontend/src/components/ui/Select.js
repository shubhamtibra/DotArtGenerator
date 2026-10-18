import { useId } from "react";
import { cx } from "../../utils/cx";
import { ChevronDownIcon } from "./icons";
import { focusRing } from "./styles";

/** Native select with a label on the left, styled to match the other controls. */
export function Select({ label, value, options, onChange, disabled = false }) {
  const id = useId();
  return (
    <div className="flex items-center justify-between gap-3">
      <label htmlFor={id} className="text-[13px] text-zinc-300">
        {label}
      </label>
      <div className="relative w-[11.5rem] max-w-[60%]">
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          className={cx(
            "h-8 w-full cursor-pointer appearance-none truncate rounded-md border border-zinc-800 bg-zinc-900 pl-2.5 pr-8 text-[13px] text-zinc-100 shadow-sm",
            "hover:border-zinc-700 disabled:cursor-not-allowed disabled:opacity-50",
            focusRing
          )}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDownIcon className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
      </div>
    </div>
  );
}
