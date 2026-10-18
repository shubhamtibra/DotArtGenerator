/** Shared option lists, defaults, presets and limits. Values mirror the API contract. */

export const MODES = [
  { value: "braille", label: "Braille" },
  { value: "ascii", label: "ASCII" },
  { value: "blocks", label: "Blocks" },
];

export const DITHERS = [
  { value: "floyd_steinberg", label: "Floyd–Steinberg" },
  { value: "atkinson", label: "Atkinson" },
  { value: "ordered", label: "Ordered (Bayer)" },
  { value: "none", label: "None (threshold)" },
];

export const CHARSETS = [
  { value: "standard", label: "Standard" },
  { value: "detailed", label: "Detailed" },
  { value: "simple", label: "Simple" },
  { value: "custom", label: "Custom" },
];

export const LIMITS = {
  width: { min: 10, max: 400, step: 1 },
  brightness: { min: -100, max: 100, step: 1 },
  contrast: { min: -100, max: 100, step: 1 },
  gamma: { min: 0.2, max: 5, step: 0.05 },
  sharpen: { min: 0, max: 3, step: 0.1 },
  edges: { min: 0, max: 1, step: 0.05 },
  threshold: { min: 0, max: 255, step: 1 },
  charAspect: { min: 0.3, max: 1, step: 0.05 },
  customCharset: { min: 2, max: 128 },
};

/** The server caps output height; wider requests are narrowed to keep the aspect ratio. */
export const MAX_ROWS = 400;

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
/** Uploads are downscaled in the browser to this longest side (the server works at ≤ 2048 px anyway). */
export const MAX_UPLOAD_SIDE = 2048;
export const MAX_URL_LENGTH = 2048;

export const GENERATE_DEBOUNCE_MS = 300;

export const DEFAULT_SETTINGS = Object.freeze({
  mode: "braille",
  width: 100,
  dither: "floyd_steinberg",
  thresholdAuto: true,
  threshold: 128,
  invert: false,
  brightness: 0,
  contrast: 0,
  gamma: 1,
  sharpen: 0,
  edges: 0,
  equalize: false,
  normalize: true,
  charset: "standard",
  customCharset: "",
  charAspect: 0.5,
});

/**
 * Presets are patches over DEFAULT_SETTINGS; applying one keeps the current
 * width, invert flag and custom characters.
 */
export const PRESETS = [
  {
    id: "balanced",
    name: "Balanced",
    description: "Braille · diffusion",
    settings: { mode: "braille", dither: "floyd_steinberg" },
  },
  {
    id: "crisp-lines",
    name: "Crisp lines",
    description: "Braille · edges",
    settings: { mode: "braille", dither: "none", thresholdAuto: true, edges: 0.6, sharpen: 1 },
  },
  {
    id: "soft-shading",
    name: "Soft shading",
    description: "Braille · Atkinson",
    settings: { mode: "braille", dither: "atkinson", contrast: 15 },
  },
  {
    id: "classic-ascii",
    name: "Classic ASCII",
    description: "10-step ramp",
    settings: { mode: "ascii", charset: "standard", dither: "none" },
  },
  {
    id: "detailed-ascii",
    name: "Detailed ASCII",
    description: "70-step ramp",
    settings: { mode: "ascii", charset: "detailed", contrast: 20 },
  },
  {
    id: "retro-blocks",
    name: "Retro blocks",
    description: "Blocks · Bayer",
    settings: { mode: "blocks", dither: "ordered" },
  },
];

export const BACKGROUNDS = ["dark", "light"];

export const PREVIEW_THEMES = {
  dark: { background: "#09090b", foreground: "#f4f4f5" },
  light: { background: "#ffffff", foreground: "#18181b" },
};

export const DEFAULT_VIEW = Object.freeze({ background: "dark", color: false, zoom: "fit" });

/** Font sizes (px) the zoom buttons step through. */
export const ZOOM_STEPS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24];
/** Bounds for the font size chosen by "Fit". */
export const FIT_FONT_SIZE = { min: 2, max: 16 };

export const ART_FONT_FAMILY =
  '"DejaVu Sans Mono", Menlo, Consolas, "Liberation Mono", "Segoe UI Symbol", "Noto Sans Symbols 2", monospace';

/** A representative glyph per mode, used to measure the rendered character advance. */
export const MODE_GLYPHS = { braille: "⣿", ascii: "@", blocks: "█" };

export const SAMPLES = [
  { id: "spheres", label: "Spheres", description: "Studio-lit 3-D" },
  { id: "mountains", label: "Mountains", description: "Sunset ridgelines" },
  { id: "waves", label: "Waves", description: "Interference rings" },
];

export const STORAGE_KEYS = {
  settings: "dot-art:settings:v2",
  view: "dot-art:view:v2",
};
