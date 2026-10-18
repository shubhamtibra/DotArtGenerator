import { useEffect, useId, useState } from "react";
import { cx } from "../../utils/cx";
import { clamp } from "../../utils/format";
import { focusRing } from "./styles";

const percent = (value, min, max) => ((value - min) / (max - min)) * 100;

/**
 * Range slider with a labelled numeric readout.
 *
 * - `origin` fills the track from that value (e.g. 0 for -100..100 sliders).
 * - `editable` turns the readout into a number field that commits on
 *   Enter/blur, or immediately when the typed value is in range.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = String,
  origin = min,
  editable = false,
  disabled = false,
  hint,
}) {
  const id = useId();
  const hintId = `${id}-hint`;
  const from = percent(Math.min(origin, value), min, max);
  const to = percent(Math.max(origin, value), min, max);

  return (
    <div className={cx(disabled && "opacity-50")}>
      <div className="mb-2 flex min-h-[1.5rem] items-center justify-between gap-3">
        <label htmlFor={id} className="text-[13px] text-zinc-300">
          {label}
        </label>
        {editable ? (
          <NumberField
            label={label}
            value={value}
            min={min}
            max={max}
            step={step}
            disabled={disabled}
            onChange={onChange}
          />
        ) : (
          <output htmlFor={id} className="text-xs tabular-nums text-zinc-400">
            {format(value)}
          </output>
        )}
      </div>
      <input
        id={id}
        type="range"
        className="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-valuetext={format(value)}
        aria-describedby={hint ? hintId : undefined}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ "--range-from": `${from}%`, "--range-to": `${to}%` }}
      />
      {hint && (
        <p id={hintId} className="mt-1.5 text-xs text-zinc-500">
          {hint}
        </p>
      )}
    </div>
  );
}

function NumberField({ label, value, min, max, step, onChange, disabled }) {
  const [draft, setDraft] = useState(String(value));

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  const commit = (raw) => {
    const parsed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setDraft(String(value));
      return;
    }
    const next = clamp(Math.round(parsed / step) * step, min, max);
    setDraft(String(next));
    if (next !== value) onChange(next);
  };

  return (
    <input
      type="number"
      inputMode="numeric"
      aria-label={`${label} value`}
      min={min}
      max={max}
      step={step}
      value={draft}
      disabled={disabled}
      onChange={(event) => {
        const raw = event.target.value;
        setDraft(raw);
        const parsed = Number(raw);
        const onStep = Math.round(parsed / step) * step === parsed;
        if (raw.trim() !== "" && onStep && parsed >= min && parsed <= max && parsed !== value) {
          onChange(parsed);
        }
      }}
      onBlur={(event) => commit(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit(event.currentTarget.value);
      }}
      className={cx(
        "h-6 w-16 rounded-md border border-zinc-800 bg-zinc-950 px-1.5 text-right text-xs tabular-nums text-zinc-200",
        "hover:border-zinc-700 focus:border-violet-500/60",
        focusRing
      )}
    />
  );
}
