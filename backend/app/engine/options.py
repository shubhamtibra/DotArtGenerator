"""Render options plus the enums and limits shared with the API layer."""

from dataclasses import dataclass

MODES = ("braille", "ascii", "blocks")
DITHERS = ("none", "floyd_steinberg", "atkinson", "ordered")

# Ramps run from densest ("most ink") to lightest ("paper").
ASCII_CHARSETS = {
    "standard": "@%#*+=-:. ",
    "detailed": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "simple": "#+-. ",
}
CHARSETS = (*ASCII_CHARSETS, "custom")
BLOCK_RAMP = "█▓▒░ "

MIN_COLUMNS, MAX_COLUMNS = 10, 400
MAX_ROWS = 400
CUSTOM_CHARSET_LENGTH = (2, 128)
THRESHOLD_RANGE = (0, 255)
BRIGHTNESS_RANGE = (-100, 100)
CONTRAST_RANGE = (-100, 100)
GAMMA_RANGE = (0.2, 5.0)
SHARPEN_RANGE = (0.0, 3.0)
EDGES_RANGE = (0.0, 1.0)
CHAR_ASPECT_RANGE = (0.3, 1.0)

MAX_SOURCE_PIXELS = 40_000_000
MAX_WORKING_SIDE = 2048


@dataclass(frozen=True)
class RenderOptions:
    """Validated rendering options. ``threshold=None`` means Otsu ("auto")."""

    mode: str = "braille"
    width: int = 100
    charset: str = "standard"
    custom_charset: str = ""
    dither: str = "floyd_steinberg"
    threshold: int | None = None
    invert: bool = False
    brightness: int = 0
    contrast: int = 0
    gamma: float = 1.0
    sharpen: float = 0.0
    edges: float = 0.0
    equalize: bool = False
    normalize: bool = True
    char_aspect: float = 0.5
    color: bool = False
    include_processed: bool = False

    @property
    def ramp(self) -> str | None:
        """Character ramp for ``ascii``/``blocks``; ``None`` for braille."""
        if self.mode == "blocks":
            return BLOCK_RAMP
        if self.mode == "ascii":
            if self.charset == "custom":
                return self.custom_charset
            return ASCII_CHARSETS[self.charset]
        return None
