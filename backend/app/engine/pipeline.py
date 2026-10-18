"""The rendering pipeline: source image + options -> art and metadata."""

import base64
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import render
from .adjust import apply_adjustments, luminance, otsu_threshold, to_uint8
from .decode import SourceImage, decode_image, image_from_array
from .dither import quantize
from .options import MAX_ROWS, RenderOptions


@dataclass(frozen=True)
class Grid:
    columns: int
    rows: int
    px_width: int  # working resolution: dots for braille, cells otherwise
    px_height: int


def generate(source: bytes | np.ndarray | SourceImage, options: RenderOptions | None = None) -> dict:
    """Render ``source`` (encoded bytes, a uint8 array or a decoded image).

    Returns the API response body minus ``elapsed_ms``.
    """
    options = options or RenderOptions()
    image = _as_source_image(source)

    # Composite onto "paper" so transparent regions never produce ink.
    rgb = composite_on_paper(image.pixels, paper=0 if options.invert else 255)
    grid = grid_size(rgb.shape[1], rgb.shape[0], options)
    v = apply_adjustments(luminance(_resize(rgb, grid.px_width, grid.px_height)), options)

    threshold_auto = options.threshold is None
    threshold = otsu_threshold(v) if threshold_auto else options.threshold

    return {
        "art": _render(v, options, threshold),
        "mode": options.mode,
        "columns": grid.columns,
        "rows": grid.rows,
        "options": {**asdict(options), "threshold": threshold, "threshold_auto": threshold_auto},
        "source": {"width": image.width, "height": image.height, "format": image.format},
        "color_map": _png_base64(_resize(rgb, grid.columns, grid.rows)) if options.color else None,
        "processed_image": _png_base64(to_uint8(v)) if options.include_processed else None,
    }


def grid_size(width: int, height: int, options: RenderOptions) -> Grid:
    """Output grid for an image of ``width`` x ``height``, capped at MAX_ROWS."""
    columns = options.width
    if options.mode == "braille":
        # Each character is 2 x 4 dots; dots are treated as square.
        rows = max(1, round(columns * 2 * height / width / 4))
        if rows > MAX_ROWS:
            rows = MAX_ROWS
            columns = max(1, min(columns, round(rows * 4 * width / height / 2)))
        return Grid(columns, rows, columns * 2, rows * 4)

    aspect = options.char_aspect  # character cell width / height
    rows = max(1, round(columns * height / width * aspect))
    if rows > MAX_ROWS:
        rows = MAX_ROWS
        columns = max(1, min(columns, round(rows * width / height / aspect)))
    return Grid(columns, rows, columns, rows)


def composite_on_paper(pixels: np.ndarray, paper: int) -> np.ndarray:
    """Flatten RGBA onto a solid gray level; RGB passes through untouched."""
    if pixels.shape[2] == 3:
        return pixels
    rgb = pixels[..., :3].astype(np.float32)
    alpha = pixels[..., 3:].astype(np.float32) / 255.0
    return np.rint(rgb * alpha + paper * (1.0 - alpha)).astype(np.uint8)


def _as_source_image(source) -> SourceImage:
    if isinstance(source, SourceImage):
        return source
    if isinstance(source, np.ndarray):
        return image_from_array(source)
    return decode_image(bytes(source))


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _render(v: np.ndarray, options: RenderOptions, threshold: int) -> str:
    if options.mode == "braille" and options.dither == "none":
        scaled = v * 255.0
        dots = scaled > threshold if options.invert else scaled < threshold
        return render.braille(dots)

    ink = v if options.invert else 1.0 - v
    if options.mode == "braille":
        return render.braille(quantize(ink, 2, options.dither) == 1)

    ramp = options.ramp
    return render.ramp_text(quantize(ink, len(ramp), options.dither), ramp)


def _png_base64(pixels: np.ndarray) -> str:
    if pixels.ndim == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
