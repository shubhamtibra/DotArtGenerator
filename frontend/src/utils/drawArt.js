import { ART_FONT_FAMILY } from "../constants";
import { brailleBits, BRAILLE_DOTS } from "./braille";

/** Ink coverage of each shade glyph, drawn as translucent cells. */
const BLOCK_COVERAGE = { "█": 1, "▓": 0.75, "▒": 0.5, "░": 0.25 };
/** Below this radius a square is indistinguishable from a circle and much cheaper. */
const MIN_ARC_RADIUS = 1.25;
/** Dot radius as a fraction of the dot pitch: distinct dots, like a well-made Braille font. */
const DOT_RADIUS = 0.34;

let canvasSupport;

/** Whether this browser can draw on a 2D canvas (false in jsdom). */
export function canDrawOnCanvas() {
  if (canvasSupport === undefined) {
    try {
      canvasSupport = Boolean(document.createElement("canvas").getContext("2d"));
    } catch {
      canvasSupport = false;
    }
  }
  return canvasSupport;
}

/**
 * Cell geometry for a mode: braille cells are twice as tall as wide (2×4
 * square dots); ascii/blocks cells use the character aspect the server
 * sampled with.
 */
export function cellHeightFor(mode, cellWidth, charAspect) {
  return mode === "braille" ? cellWidth * 2 : cellWidth / charAspect;
}

/** 5 bits per channel: visually lossless here, and it lets dots share paths. */
const quantize = (value) => (value & 0xf8) | 0x04;

function colorAt(colors, columns, x, y) {
  const i = (y * columns + x) * 4;
  return `rgb(${quantize(colors[i])},${quantize(colors[i + 1])},${quantize(colors[i + 2])})`;
}

/**
 * Draw art onto a 2D context on an explicit cell grid, so alignment never
 * depends on font metrics. Braille is drawn as real dots, blocks as
 * translucent cells, ASCII as centred glyphs. `colors` (RGBA per character,
 * row-major) overrides `foreground` when given.
 */
export function drawArt(context, options) {
  const { art, mode, columns, cellWidth, cellHeight, foreground, colors, originX = 0, originY = 0 } = options;
  const lines = art.split("\n");
  const fill = (x, y) => (colors ? colorAt(colors, columns, x, y) : foreground);

  if (mode === "braille") {
    drawBraille(context, lines, fill, { cellWidth, cellHeight, originX, originY });
  } else if (mode === "blocks") {
    drawBlocks(context, lines, fill, { cellWidth, cellHeight, originX, originY });
  } else {
    drawGlyphs(context, lines, fill, { cellWidth, cellHeight, originX, originY });
  }
}

function drawBraille(context, lines, fill, { cellWidth, cellHeight, originX, originY }) {
  const stepX = cellWidth / 2;
  const stepY = cellHeight / 4;
  const radius = Math.min(stepX, stepY) * DOT_RADIUS;
  const useArcs = radius >= MIN_ARC_RADIUS;
  // Batch dots by colour: one path per distinct colour keeps fills cheap.
  const paths = new Map();

  lines.forEach((line, y) => {
    for (let x = 0; x < line.length; x += 1) {
      const bits = brailleBits(line[x]);
      if (!bits) continue;
      const color = fill(x, y);
      let dots = paths.get(color);
      if (!dots) {
        dots = [];
        paths.set(color, dots);
      }
      for (const [bit, dx, dy] of BRAILLE_DOTS) {
        if (bits & bit) {
          dots.push(originX + x * cellWidth + (dx + 0.5) * stepX, originY + y * cellHeight + (dy + 0.5) * stepY);
        }
      }
    }
  });

  for (const [color, dots] of paths) {
    context.fillStyle = color;
    context.beginPath();
    for (let i = 0; i < dots.length; i += 2) {
      const cx = dots[i];
      const cy = dots[i + 1];
      if (useArcs) {
        context.moveTo(cx + radius, cy);
        context.arc(cx, cy, radius, 0, Math.PI * 2);
      } else {
        context.rect(cx - radius, cy - radius, radius * 2, radius * 2);
      }
    }
    context.fill();
  }
}

function drawBlocks(context, lines, fill, { cellWidth, cellHeight, originX, originY }) {
  lines.forEach((line, y) => {
    const top = Math.round(originY + y * cellHeight);
    const bottom = Math.round(originY + (y + 1) * cellHeight);
    for (let x = 0; x < line.length; x += 1) {
      const coverage = BLOCK_COVERAGE[line[x]];
      if (!coverage) continue;
      const left = Math.round(originX + x * cellWidth);
      const right = Math.round(originX + (x + 1) * cellWidth);
      context.globalAlpha = coverage;
      context.fillStyle = fill(x, y);
      context.fillRect(left, top, right - left, bottom - top);
    }
  });
  context.globalAlpha = 1;
}

function drawGlyphs(context, lines, fill, { cellWidth, cellHeight, originX, originY }) {
  const fontSize = Math.min(cellHeight, cellWidth / 0.6);
  context.font = `${fontSize}px ${ART_FONT_FAMILY}`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  let current = null;
  lines.forEach((line, y) => {
    const cy = originY + (y + 0.5) * cellHeight;
    for (let x = 0; x < line.length; x += 1) {
      const char = line[x];
      if (char === " ") continue;
      const color = fill(x, y);
      if (color !== current) {
        context.fillStyle = color;
        current = color;
      }
      context.fillText(char, originX + (x + 0.5) * cellWidth, cy);
    }
  });
}
