import { useRef } from "react";
import { cx } from "../../utils/cx";
import { focusRing } from "./styles";

const SIZES = {
  sm: { box: "h-[22px] text-xs", padding: "px-2" },
  md: { box: "h-7 text-[13px]", padding: "px-3" },
};

const NEXT_KEYS = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

/**
 * A single-choice button group (ARIA radiogroup with roving focus).
 * Options: { value, label, icon? }. With `iconOnly`, labels become
 * accessible names and tooltips.
 */
export function SegmentedControl({ label, value, options, onChange, size = "md", iconOnly = false, className }) {
  const buttons = useRef([]);
  const selectedIndex = options.findIndex((option) => option.value === value);
  const focusIndex = selectedIndex === -1 ? 0 : selectedIndex;

  const select = (index) => {
    onChange(options[index].value);
    buttons.current[index]?.focus();
  };

  const handleKeyDown = (event) => {
    let index;
    if (event.key in NEXT_KEYS) index = (focusIndex + NEXT_KEYS[event.key] + options.length) % options.length;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = options.length - 1;
    else return;
    event.preventDefault();
    select(index);
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      onKeyDown={handleKeyDown}
      className={cx("inline-flex rounded-lg border border-zinc-800 bg-zinc-950 p-0.5", className)}
    >
      {options.map((option, index) => {
        const selected = index === selectedIndex;
        const IconComponent = option.icon;
        return (
          <button
            key={option.value}
            ref={(node) => {
              buttons.current[index] = node;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            aria-label={iconOnly ? option.label : undefined}
            title={iconOnly ? option.label : undefined}
            tabIndex={index === focusIndex ? 0 : -1}
            onClick={() => onChange(option.value)}
            className={cx(
              "inline-flex flex-1 items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-colors",
              SIZES[size].box,
              iconOnly ? "aspect-square" : SIZES[size].padding,
              selected
                ? "bg-zinc-800 text-zinc-50 shadow-sm ring-1 ring-inset ring-white/[0.06]"
                : "text-zinc-400 hover:text-zinc-200",
              focusRing
            )}
          >
            {IconComponent && <IconComponent className="h-3.5 w-3.5" />}
            {!iconOnly && option.label}
          </button>
        );
      })}
    </div>
  );
}
