import { cellHeightFor, drawArt } from "./drawArt";
import { clamp } from "./format";

const PNG_MAX_SIDE = 4096;
/** Exports aim for roughly this width so small and large grids both come out crisp. */
const PNG_TARGET_WIDTH = 2400;
const PNG_PADDING_CELLS = 2;

/** "my-photo-braille-100x42.txt" */
export function artFilename(sourceName, { mode, columns, rows }, extension) {
  const base =
    String(sourceName || "dot-art")
      .replace(/\.[^.]+$/, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "dot-art";
  return `${base}-${mode}-${columns}x${rows}.${extension}`;
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoke on the next tick: some browsers start the download asynchronously.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadText(text, filename) {
  downloadBlob(new Blob([text], { type: "text/plain;charset=utf-8" }), filename);
}

/** Pixel layout of a PNG export: cell size, padding and canvas size, capped per side. */
export function pngLayout({ mode, columns, rows, charAspect }) {
  const baseCell = clamp(Math.round(PNG_TARGET_WIDTH / columns), 4, 24);
  const baseHeight = cellHeightFor(mode, baseCell, charAspect);
  const padding = baseCell * PNG_PADDING_CELLS;
  const fullWidth = columns * baseCell + padding * 2;
  const fullHeight = rows * baseHeight + padding * 2;
  const scale = Math.min(1, PNG_MAX_SIDE / fullWidth, PNG_MAX_SIDE / fullHeight);
  return {
    cellWidth: baseCell * scale,
    cellHeight: baseHeight * scale,
    padding: padding * scale,
    width: Math.max(1, Math.floor(fullWidth * scale)),
    height: Math.max(1, Math.floor(fullHeight * scale)),
  };
}

/**
 * Render the art to a PNG blob using the preview colours (or per-character
 * colours). Resolves to null when the browser has no canvas support.
 */
export function renderArtPng({ art, mode, columns, rows, charAspect, colors, theme }) {
  const layout = pngLayout({ mode, columns, rows, charAspect });
  const canvas = document.createElement("canvas");
  canvas.width = layout.width;
  canvas.height = layout.height;
  const context = canvas.getContext("2d");
  if (!context) return Promise.resolve(null);

  context.fillStyle = theme.background;
  context.fillRect(0, 0, layout.width, layout.height);
  drawArt(context, {
    art,
    mode,
    columns,
    cellWidth: layout.cellWidth,
    cellHeight: layout.cellHeight,
    foreground: theme.foreground,
    colors,
    originX: layout.padding,
    originY: layout.padding,
  });
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}
