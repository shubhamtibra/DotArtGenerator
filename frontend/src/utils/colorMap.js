import { loadImage } from "./image";

/**
 * Decode the API's base64 `color_map` PNG (one pixel per character) into RGBA
 * bytes. Resolves to null when canvas isn't available or decoding fails.
 */
export async function decodeColorMap(encoded, columns, rows) {
  const canvas = document.createElement("canvas");
  canvas.width = columns;
  canvas.height = rows;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  try {
    const image = await loadImage(`data:image/png;base64,${encoded}`);
    context.drawImage(image, 0, 0, columns, rows);
    return context.getImageData(0, 0, columns, rows).data;
  } catch {
    return null;
  }
}
