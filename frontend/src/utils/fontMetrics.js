import { ART_FONT_FAMILY } from "../constants";

/** Typical monospace advance; used when layout can't be measured (jsdom, SSR). */
const FALLBACK_ADVANCE = 0.6;
const PROBE_SIZE = 100;
const PROBE_REPEAT = 64;
const cache = new Map();

/**
 * Horizontal advance of `char` in the art font, as a fraction of the font
 * size. Braille glyphs often come from a fallback font whose advance differs
 * from the primary monospace font, so this is measured rather than assumed.
 */
export function measureAdvance(char) {
  if (cache.has(char)) return cache.get(char);
  let advance = FALLBACK_ADVANCE;
  if (typeof document !== "undefined" && document.body) {
    const probe = document.createElement("span");
    probe.textContent = char.repeat(PROBE_REPEAT);
    Object.assign(probe.style, {
      position: "absolute",
      left: "-10000px",
      top: "0",
      visibility: "hidden",
      whiteSpace: "pre",
      fontFamily: ART_FONT_FAMILY,
      fontSize: `${PROBE_SIZE}px`,
      lineHeight: "1",
    });
    document.body.appendChild(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    if (width > 0) advance = width / PROBE_REPEAT / PROBE_SIZE;
  }
  cache.set(char, advance);
  return advance;
}
