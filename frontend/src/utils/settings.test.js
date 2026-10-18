import { DEFAULT_SETTINGS, PRESETS } from "../constants";
import {
  applyPreset,
  buildRequestOptions,
  estimateGrid,
  findActivePreset,
  getSettingsIssue,
  sanitizeSettings,
  sanitizeView,
} from "./settings";

const DARK = { background: "dark", color: false, includeProcessed: false };

test("invert is relative to the preview background", () => {
  expect(buildRequestOptions(DEFAULT_SETTINGS, DARK).invert).toBe(true);
  expect(buildRequestOptions(DEFAULT_SETTINGS, { ...DARK, background: "light" }).invert).toBe(false);
  expect(buildRequestOptions({ ...DEFAULT_SETTINGS, invert: true }, DARK).invert).toBe(false);
});

test("only mode-relevant fields are sent", () => {
  const braille = buildRequestOptions(DEFAULT_SETTINGS, DARK);
  expect(braille).not.toHaveProperty("threshold");
  expect(braille).not.toHaveProperty("charset");
  expect(braille).not.toHaveProperty("char_aspect");

  const thresholded = buildRequestOptions({ ...DEFAULT_SETTINGS, dither: "none" }, DARK);
  expect(thresholded.threshold).toBe("auto");
  const manual = buildRequestOptions({ ...DEFAULT_SETTINGS, dither: "none", thresholdAuto: false, threshold: 90 }, DARK);
  expect(manual.threshold).toBe(90);

  const ascii = buildRequestOptions({ ...DEFAULT_SETTINGS, mode: "ascii", charset: "custom", customCharset: "#. " }, DARK);
  expect(ascii).toMatchObject({ charset: "custom", custom_charset: "#. ", char_aspect: 0.5 });

  const blocks = buildRequestOptions({ ...DEFAULT_SETTINGS, mode: "blocks", charset: "detailed" }, DARK);
  expect(blocks).not.toHaveProperty("charset");
  expect(blocks.char_aspect).toBe(0.5);
});

test("color and processed-image flags map to API fields", () => {
  const options = buildRequestOptions(DEFAULT_SETTINGS, { background: "dark", color: true, includeProcessed: true });
  expect(options.color).toBe(true);
  expect(options.include_processed).toBe(true);
});

test("sanitizes untrusted stored settings", () => {
  expect(sanitizeSettings(undefined)).toEqual(DEFAULT_SETTINGS);
  expect(sanitizeSettings("garbage")).toEqual(DEFAULT_SETTINGS);
  const settings = sanitizeSettings({
    mode: "hologram",
    width: 9000,
    gamma: "2.5",
    brightness: -12.7,
    invert: "yes",
    customCharset: "ab\ncd\t",
  });
  expect(settings.mode).toBe("braille");
  expect(settings.width).toBe(400);
  expect(settings.gamma).toBe(2.5);
  expect(settings.brightness).toBe(-13);
  expect(settings.invert).toBe(false);
  expect(settings.customCharset).toBe("abcd");
  expect(sanitizeView({ background: "neon", zoom: 13 })).toEqual({ background: "dark", color: false, zoom: "fit" });
});

test("presets keep width and invert, and are recognized once applied", () => {
  const current = { ...DEFAULT_SETTINGS, width: 180, invert: true, contrast: 40 };
  const crisp = PRESETS.find((preset) => preset.id === "crisp-lines");
  const applied = applyPreset(current, crisp);
  expect(applied).toMatchObject({ width: 180, invert: true, contrast: 0, dither: "none", edges: 0.6, sharpen: 1 });
  expect(findActivePreset(applied)).toBe(crisp);
  expect(findActivePreset({ ...applied, gamma: 1.5 })).toBeNull();
  expect(findActivePreset(DEFAULT_SETTINGS).id).toBe("balanced");
});

test("an incomplete custom charset blocks generation with a hint", () => {
  const custom = { ...DEFAULT_SETTINGS, mode: "ascii", charset: "custom", customCharset: "#" };
  expect(getSettingsIssue(custom)).toMatch(/at least 2/);
  expect(getSettingsIssue({ ...custom, customCharset: "#." })).toBeNull();
  expect(getSettingsIssue({ ...custom, mode: "braille" })).toBeNull();
});

test("estimates the output grid like the server, including the row cap", () => {
  expect(estimateGrid({ mode: "braille", width: 100, charAspect: 0.5 }, 768 / 1024)).toEqual({ columns: 100, rows: 38 });
  expect(estimateGrid({ mode: "ascii", width: 100, charAspect: 0.5 }, 1)).toEqual({ columns: 100, rows: 50 });
  expect(estimateGrid({ mode: "ascii", width: 400, charAspect: 1 }, 4)).toEqual({ columns: 100, rows: 400 });
  expect(estimateGrid({ mode: "braille", width: 100, charAspect: 0.5 }, null)).toBeNull();
});
