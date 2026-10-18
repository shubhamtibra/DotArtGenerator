import { PRESETS } from "../constants";
import { cx } from "../utils/cx";
import { focusRing } from "./ui/styles";

export function PresetPicker({ activeId, onSelect }) {
  return (
    <ul className="grid grid-cols-2 gap-2">
      {PRESETS.map((preset) => {
        const active = preset.id === activeId;
        return (
          <li key={preset.id}>
            <button
              type="button"
              aria-pressed={active}
              onClick={() => onSelect(preset)}
              className={cx(
                "flex h-full w-full flex-col items-start rounded-lg border px-2.5 py-2 text-left transition-colors",
                active
                  ? "border-violet-500/50 bg-violet-500/10"
                  : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700 hover:bg-zinc-900",
                focusRing
              )}
            >
              <span className={cx("text-[13px] font-medium", active ? "text-violet-100" : "text-zinc-200")}>
                {preset.name}
              </span>
              <span className={cx("mt-0.5 text-[11px] leading-snug", active ? "text-violet-200/70" : "text-zinc-500")}>
                {preset.description}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
