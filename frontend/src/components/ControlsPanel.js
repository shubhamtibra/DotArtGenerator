import { useId } from "react";
import { CHARSETS, DITHERS, LIMITS, MODES } from "../constants";
import { cx } from "../utils/cx";
import { formatDecimal, formatNumber, formatSigned } from "../utils/format";
import { cleanCharset, findActivePreset } from "../utils/settings";
import { PresetPicker } from "./PresetPicker";
import { Button } from "./ui/Button";
import { PanelSection } from "./ui/PanelSection";
import { SegmentedControl } from "./ui/SegmentedControl";
import { Select } from "./ui/Select";
import { Slider } from "./ui/Slider";
import { Toggle } from "./ui/Toggle";
import { ResetIcon } from "./ui/icons";
import { focusRing } from "./ui/styles";

const formatGamma = (value) => formatDecimal(value, 2);
const formatOne = (value) => formatDecimal(value, 1);
const formatEdges = (value) => `${Math.round(value * 100)}%`;

/**
 * All generation settings. `onChange` receives a partial settings patch.
 * `grid` is the predicted output size; `resolvedThreshold` is the threshold
 * the server last picked automatically.
 */
export function ControlsPanel({ settings, onChange, onApplyPreset, onReset, grid, resolvedThreshold, issue }) {
  const set = (key) => (value) => onChange({ [key]: value });
  const { mode } = settings;
  const usesThreshold = mode === "braille" && settings.dither === "none";
  const activePreset = findActivePreset(settings);

  return (
    <div className="divide-y divide-zinc-800/80">
      <PanelSection title="Output">
        <SegmentedControl label="Mode" value={mode} options={MODES} onChange={set("mode")} className="w-full" />
        <Slider
          label="Width"
          editable
          value={settings.width}
          {...LIMITS.width}
          onChange={set("width")}
          hint={
            grid ? `${formatNumber(grid.columns)} × ${formatNumber(grid.rows)} characters` : "Columns of characters"
          }
        />
      </PanelSection>

      <PanelSection title="Presets">
        <PresetPicker activeId={activePreset?.id} onSelect={onApplyPreset} />
      </PanelSection>

      <PanelSection title="Tone">
        <Slider
          label="Brightness"
          value={settings.brightness}
          {...LIMITS.brightness}
          origin={0}
          format={formatSigned}
          onChange={set("brightness")}
        />
        <Slider
          label="Contrast"
          value={settings.contrast}
          {...LIMITS.contrast}
          origin={0}
          format={formatSigned}
          onChange={set("contrast")}
        />
        <Slider
          label="Gamma"
          value={settings.gamma}
          {...LIMITS.gamma}
          origin={1}
          format={formatGamma}
          onChange={set("gamma")}
        />
      </PanelSection>

      <PanelSection title="Detail">
        <Slider
          label="Sharpen"
          value={settings.sharpen}
          {...LIMITS.sharpen}
          format={formatOne}
          onChange={set("sharpen")}
        />
        <Slider label="Edges" value={settings.edges} {...LIMITS.edges} format={formatEdges} onChange={set("edges")} />
        <Toggle
          label="Normalize"
          description="Stretch tones to the full range"
          checked={settings.normalize}
          onChange={set("normalize")}
        />
        <Toggle
          label="Equalize"
          description="Boost local contrast (CLAHE)"
          checked={settings.equalize}
          onChange={set("equalize")}
        />
      </PanelSection>

      <PanelSection title="Rendering">
        <Select label="Dithering" value={settings.dither} options={DITHERS} onChange={set("dither")} />

        {usesThreshold && (
          <>
            <Toggle
              label="Auto threshold"
              description="Pick the cut-off automatically (Otsu)"
              checked={settings.thresholdAuto}
              onChange={(auto) =>
                onChange(
                  auto
                    ? { thresholdAuto: true }
                    : { thresholdAuto: false, threshold: resolvedThreshold ?? settings.threshold }
                )
              }
            />
            <Slider
              label="Threshold"
              value={settings.thresholdAuto ? resolvedThreshold ?? settings.threshold : settings.threshold}
              {...LIMITS.threshold}
              disabled={settings.thresholdAuto}
              onChange={set("threshold")}
            />
          </>
        )}

        {mode === "ascii" && (
          <>
            <Select label="Characters" value={settings.charset} options={CHARSETS} onChange={set("charset")} />
            {settings.charset === "custom" && (
              <CustomCharsetField value={settings.customCharset} onChange={set("customCharset")} issue={issue} />
            )}
          </>
        )}

        {mode !== "braille" && (
          <Slider
            label="Character aspect"
            value={settings.charAspect}
            {...LIMITS.charAspect}
            format={formatGamma}
            onChange={set("charAspect")}
            hint="Width ÷ height of a character cell in your font"
          />
        )}

        <Toggle
          label="Invert"
          description="Swap ink and paper"
          checked={settings.invert}
          onChange={set("invert")}
        />
      </PanelSection>

      <div className="px-4 py-4">
        <Button variant="ghost" icon={ResetIcon} onClick={onReset} className="w-full">
          Reset to defaults
        </Button>
      </div>
    </div>
  );
}

function CustomCharsetField({ value, onChange, issue }) {
  const id = useId();
  const hintId = `${id}-hint`;
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-[13px] text-zinc-300">
        Custom characters
      </label>
      <input
        id={id}
        type="text"
        spellCheck="false"
        autoComplete="off"
        maxLength={LIMITS.customCharset.max}
        placeholder="e.g. #@*+:. "
        value={value}
        aria-invalid={Boolean(issue)}
        aria-describedby={hintId}
        onChange={(event) => onChange(cleanCharset(event.target.value))}
        style={{ whiteSpace: "pre" }}
        className={cx(
          "h-8 w-full rounded-md border bg-zinc-950 px-2.5 font-mono text-[13px] text-zinc-100 placeholder:text-zinc-600",
          "hover:border-zinc-700 focus:border-violet-500/60",
          issue ? "border-amber-500/50" : "border-zinc-800",
          focusRing
        )}
      />
      <p id={hintId} className={cx("mt-1.5 text-xs", issue ? "text-amber-400/90" : "text-zinc-500")}>
        {issue || "Densest to lightest. A trailing space keeps light areas blank."}
      </p>
    </div>
  );
}
