"""Image -> dot art rendering engine.

Pure Python/NumPy/OpenCV/Pillow with no Django dependency, so it can be unit
tested and reused on its own. The single entry point is :func:`generate`.
"""

from .decode import SourceImage, decode_image, image_from_array
from .errors import (
    EngineError,
    FetchError,
    ImageTooLargeError,
    InvalidImageError,
    UrlNotAllowedError,
)
from .fetch import fetch_image_bytes, validate_url
from .options import RenderOptions
from .pipeline import generate, grid_size

__all__ = [
    "EngineError",
    "FetchError",
    "ImageTooLargeError",
    "InvalidImageError",
    "RenderOptions",
    "SourceImage",
    "UrlNotAllowedError",
    "decode_image",
    "fetch_image_bytes",
    "generate",
    "grid_size",
    "image_from_array",
    "validate_url",
]
