import { useEffect, useRef } from "react";
import { drawArt } from "../utils/drawArt";

/** Keep the backing store within what browsers will reliably allocate. */
const MAX_BACKING_SIDE = 8192;
const MAX_BACKING_PIXELS = 16_777_216;

function backingScale(width, height) {
  const deviceScale = window.devicePixelRatio || 1;
  return Math.min(
    deviceScale,
    MAX_BACKING_SIDE / width,
    MAX_BACKING_SIDE / height,
    Math.sqrt(MAX_BACKING_PIXELS / (width * height))
  );
}

/**
 * Art drawn on an explicit cell grid (see drawArt). Used for colour, where
 * tens of thousands of individually styled spans would be far too slow, and
 * for Braille dots. Decorative: the selectable text sits on top of it.
 */
export function ArtCanvas({ art, mode, columns, rows, cellWidth, cellHeight, colors, foreground }) {
  const canvasRef = useRef(null);
  const width = columns * cellWidth;
  const height = rows * cellHeight;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!context || !width || !height) return;
    const scale = backingScale(width, height);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    context.setTransform(scale, 0, 0, scale, 0, 0);
    context.clearRect(0, 0, width, height);
    drawArt(context, { art, mode, columns, cellWidth, cellHeight, foreground, colors });
  }, [art, mode, columns, cellWidth, cellHeight, width, height, colors, foreground]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="block"
      style={{ width: `${width}px`, height: `${height}px` }}
    />
  );
}
