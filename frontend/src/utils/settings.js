import {
  BACKGROUNDS,
  CHARSETS,
  DEFAULT_SETTINGS,
  DEFAULT_VIEW,
  DITHERS,
  LIMITS,
  MAX_ROWS,
  MODES,
  PRESETS,
  ZOOM_STEPS,
} from "../constants";
import { clamp } from "./format";

const NUMERIC_KEYS = ["width", "brightness", "contrast", "gamma", "sharpen", "edges", "threshold", "charAspect"];
const INTEGER_KEYS = new Set(["width", "brightness", "contrast", "threshold"]);
const BOOLEAN_KEYS = ["thresholdAuto", "invert", "equalize", "normalize"];
const ENUMS = {
  mode: MODES.map((o) => o.value),
  dither: DITHERS.map((o) => o.value),
  charset: CHARSETS.map((o) => o.value),
};

/** Strip characters the API rejects in a custom ramp and enforce its length limit. */
export function cleanCharset(value) {
  return String(value).replace(/[\r\n\t]/g, "").slice(0, LIMITS.customCharset.max);
}

/** Coerce untrusted (e.g. persisted) settings into a valid settings object. */
export function sanitizeSettings(raw) {
  const input = raw && typeof raw === "object" ? raw : {};
  const settings = { ...DEFAULT_SETTINGS };

  for (const [key, allowed] of Object.entries(ENUMS)) {
    if (allowed.includes(input[key])) settings[key] = input[key];
  }
  for (const key of NUMERIC_KEYS) {
    const value = Number(input[key]);
    if (input[key] === null || input[key] === "" || !Number.isFinite(value)) continue;
    const { min, max } = LIMITS[key];
    settings[key] = clamp(INTEGER_KEYS.has(key) ? Math.round(value) : value, min, max);
  }
  for (const key of BOOLEAN_KEYS) {
    if (typeof input[key] === "boolean") settings[key] = input[key];
  }
  if (typeof input.customCharset === "string") settings.customCharset = cleanCharset(input.customCharset);
  return settings;
}

export function sanitizeView(raw) {
  const input = raw && typeof raw === "object" ? raw : {};
  return {
    background: BACKGROUNDS.includes(input.background) ? input.background : DEFAULT_VIEW.background,
    color: typeof input.color === "boolean" ? input.color : DEFAULT_VIEW.color,
    zoom: input.zoom === "fit" || ZOOM_STEPS.includes(input.zoom) ? input.zoom : DEFAULT_VIEW.zoom,
  };
}

export function applyPreset(settings, preset) {
  return {
    ...DEFAULT_SETTINGS,
    width: settings.width,
    invert: settings.invert,
    customCharset: settings.customCharset,
    ...preset.settings,
  };
}

/** The preset the current settings exactly correspond to, if any. */
export function findActivePreset(settings) {
  return (
    PRESETS.find((preset) => {
      const applied = applyPreset(settings, preset);
      return Object.keys(applied).every((key) => applied[key] === settings[key]);
    }) ?? null
  );
}

/** A reason the settings can't be sent yet (shown inline instead of an error round-trip). */
export function getSettingsIssue(settings) {
  if (
    settings.mode === "ascii" &&
    settings.charset === "custom" &&
    settings.customCharset.length < LIMITS.customCharset.min
  ) {
    return `Enter at least ${LIMITS.customCharset.min} custom characters, densest first.`;
  }
  return null;
}

/**
 * Translate UI settings into API request fields. Only fields that affect the
 * chosen mode are included, so unrelated tweaks don't trigger a regeneration.
 * Invert is relative to the preview background so the art always reads
 * correctly on it; the user's Invert toggle flips that.
 */
export function buildRequestOptions(settings, { background, color, includeProcessed }) {
  const { mode } = settings;
  const options = {
    mode,
    width: settings.width,
    dither: settings.dither,
    invert: settings.invert !== (background === "dark"),
    brightness: settings.brightness,
    contrast: settings.contrast,
    gamma: settings.gamma,
    sharpen: settings.sharpen,
    edges: settings.edges,
    equalize: settings.equalize,
    normalize: settings.normalize,
    color: Boolean(color),
    include_processed: Boolean(includeProcessed),
  };
  if (mode === "braille" && settings.dither === "none") {
    options.threshold = settings.thresholdAuto ? "auto" : settings.threshold;
  }
  if (mode === "ascii") {
    options.charset = settings.charset;
    if (settings.charset === "custom") options.custom_charset = settings.customCharset;
  }
  if (mode !== "braille") {
    options.char_aspect = settings.charAspect;
  }
  return options;
}

/**
 * Predict the output grid for an image with the given height/width ratio,
 * mirroring the server's sizing rules (including the row cap).
 */
export function estimateGrid({ mode, width, charAspect }, aspectRatio) {
  if (!aspectRatio || !Number.isFinite(aspectRatio)) return null;
  const rowsPerColumn = mode === "braille" ? aspectRatio / 2 : aspectRatio * charAspect;
  let columns = width;
  let rows = Math.max(1, Math.round(columns * rowsPerColumn));
  if (rows > MAX_ROWS) {
    rows = MAX_ROWS;
    columns = Math.max(1, Math.round(MAX_ROWS / rowsPerColumn));
  }
  return { columns, rows };
}
