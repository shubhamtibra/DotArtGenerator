import { MAX_UPLOAD_BYTES, MAX_UPLOAD_SIDE, MAX_URL_LENGTH } from "../constants";
import { formatBytes } from "./format";

const IMAGE_EXTENSION = /\.(png|jpe?g|gif|webp|bmp|tiff?|avif)$/i;
/** Sources that may carry transparency are re-encoded losslessly. */
const LOSSLESS_TYPES = new Set(["image/png", "image/gif", "image/webp"]);
const JPEG_QUALITY = 0.92;

export function isImageFile(file) {
  if (!file) return false;
  if (file.type) return file.type.startsWith("image/");
  return IMAGE_EXTENSION.test(file.name || "");
}

/** A user-facing reason the file can't be used, or null. */
export function validateImageFile(file) {
  if (!isImageFile(file)) {
    const name = file?.name ? `“${file.name}”` : "That file";
    return `${name} isn't an image. Choose a PNG, JPEG, GIF, WebP, BMP or TIFF file.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That image is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_UPLOAD_BYTES)}.`;
  }
  return null;
}

/** Validate a user-entered image URL. Returns { url } or { error }. */
export function validateImageUrl(raw) {
  const value = String(raw ?? "").trim();
  if (!value) return { error: "Enter an image URL." };
  if (value.length > MAX_URL_LENGTH) return { error: "That URL is too long." };
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return { error: "Enter a full URL starting with http:// or https://." };
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return { error: "Only http:// and https:// URLs are supported." };
  }
  return { url: parsed.href };
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.referrerPolicy = "no-referrer";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    image.src = src;
  });
}

/** Natural size of a remote image, or null if the browser can't display it. */
export async function probeImageSize(src) {
  try {
    const image = await loadImage(src);
    return { width: image.naturalWidth, height: image.naturalHeight };
  } catch {
    return null;
  }
}

async function decodeBitmap(blob) {
  if (typeof createImageBitmap !== "function") return null;
  try {
    return await createImageBitmap(blob);
  } catch {
    return null;
  }
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve) => {
    try {
      canvas.toBlob(resolve, type, quality);
    } catch {
      resolve(null);
    }
  });
}

/** Swap a file name's extension to match the encoded type. */
export function uploadFilename(name, type) {
  const base = (name || "image").replace(/\.[^./\\]+$/, "") || "image";
  const extension = { "image/png": "png", "image/jpeg": "jpg" }[type];
  return extension ? `${base}.${extension}` : name || "image";
}

/**
 * Prepare an image file for (repeated) upload. Reads its natural size and,
 * when the longest side exceeds `maxSide`, downscales it on a canvas so every
 * regeneration sends a small blob. Falls back to the original file whenever
 * the browser can't decode it or has no canvas (e.g. jsdom, TIFF in Chrome).
 *
 * Resolves to { blob, width, height } — width/height are null when unknown.
 */
export async function prepareImageFile(file, maxSide = MAX_UPLOAD_SIDE) {
  const bitmap = await decodeBitmap(file);
  if (!bitmap) return { blob: file, width: null, height: null };

  const { width, height } = bitmap;
  const original = { blob: file, width, height };
  try {
    const scale = maxSide / Math.max(width, height);
    if (scale >= 1) return original;

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const context = canvas.getContext("2d");
    if (!context) return original;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

    const type = LOSSLESS_TYPES.has(file.type) ? "image/png" : "image/jpeg";
    const blob = await canvasToBlob(canvas, type, JPEG_QUALITY);
    return blob ? { blob, width, height } : original;
  } catch {
    return original;
  } finally {
    bitmap.close?.();
  }
}
