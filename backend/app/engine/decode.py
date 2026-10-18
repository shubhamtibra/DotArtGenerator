"""Turn untrusted image bytes (or arrays) into bounded RGB/RGBA pixel arrays."""

import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .errors import EngineError, ImageTooLargeError, InvalidImageError
from .options import MAX_SOURCE_PIXELS, MAX_WORKING_SIDE

# Pillow's decompression-bomb guard backs up the explicit size check below; its
# warning (raised between 1x and 2x the limit) is promoted to an error so it
# surfaces as ``too_large`` instead of a log line.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)

_EXIF_ORIENTATION = 0x0112
_ORIENTATIONS_SWAPPING_AXES = {5, 6, 7, 8}
_HIGH_BIT_DEPTH_MODES = {"I;16", "I;16B", "I;16L", "I;16N", "I", "F"}
_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}

UNSUPPORTED_MESSAGE = (
    "Could not read that image. Supported formats include PNG, JPEG, GIF, WebP, BMP and TIFF."
)


@dataclass(frozen=True)
class SourceImage:
    """Pixels at working resolution plus facts about the original image."""

    pixels: np.ndarray  # uint8, H x W x 3 (RGB) or H x W x 4 (RGBA)
    width: int  # original width, after EXIF orientation is applied
    height: int
    format: str | None


def decode_image(
    data: bytes,
    *,
    max_pixels: int = MAX_SOURCE_PIXELS,
    max_side: int = MAX_WORKING_SIDE,
) -> SourceImage:
    """Decode the first frame of an image, upright, downscaled to ``max_side``."""
    if not data:
        raise InvalidImageError("The image is empty.")
    try:
        return _decode(data, max_pixels, max_side)
    except EngineError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError(_too_large_message(max_pixels)) from exc
    except Exception as exc:  # Pillow plugins raise many exception types on malformed input.
        raise InvalidImageError(UNSUPPORTED_MESSAGE) from exc


def image_from_array(pixels: np.ndarray, *, max_side: int = MAX_WORKING_SIDE) -> SourceImage:
    """Wrap an in-memory uint8 grayscale, RGB or RGBA array as a ``SourceImage``."""
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        raise ValueError("pixels must be a uint8 array")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or 0 in arr.shape:
        raise ValueError("pixels must be H x W, H x W x 3 or H x W x 4")
    height, width = arr.shape[:2]
    scale = max_side / max(width, height)
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return SourceImage(pixels=arr, width=width, height=height, format=None)


def _decode(data: bytes, max_pixels: int, max_side: int) -> SourceImage:
    # Image.open only reads the header; the pixel budget is enforced before decoding.
    # For animated GIF/WebP/APNG it is positioned on the first frame.
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        if width * height > max_pixels:
            raise ImageTooLargeError(_too_large_message(max_pixels))
        source_format = img.format
        if img.getexif().get(_EXIF_ORIENTATION, 1) in _ORIENTATIONS_SWAPPING_AXES:
            width, height = height, width

        if source_format == "JPEG":
            # Let libjpeg decode huge photos at 1/2, 1/4 or 1/8 scale directly.
            scale = max(img.size) / max_side
            if scale >= 2:
                img.draft(img.mode, (max(1, int(img.width / scale)), max(1, int(img.height / scale))))

        ImageOps.exif_transpose(img, in_place=True)
        img = _to_rgb_or_rgba(img)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        pixels = np.asarray(img)

    return SourceImage(pixels=pixels, width=width, height=height, format=source_format)


def _to_rgb_or_rgba(img: Image.Image) -> Image.Image:
    if img.mode in _HIGH_BIT_DEPTH_MODES:
        # convert("L") clips 16-bit samples instead of scaling them, which turns
        # 16-bit grayscale PNGs almost white.
        samples = np.asarray(img, dtype=np.float64)
        if img.mode.startswith("I;16") or samples.max() > 255:
            samples = samples / 257.0
        img = Image.fromarray(np.clip(samples, 0, 255).astype(np.uint8), "L")
    has_alpha = img.mode in _ALPHA_MODES or "transparency" in img.info
    return img.convert("RGBA" if has_alpha else "RGB")


def _too_large_message(max_pixels: int) -> str:
    return f"The image has too many pixels (limit is {max_pixels / 1_000_000:g} megapixels)."
