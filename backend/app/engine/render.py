"""Turn dot masks and level grids into text.

Every line is exactly ``columns`` characters, lines are joined with ``\\n`` and
there is no trailing newline. Nothing is stripped: braille blanks use U+2800 so
pasted art never collapses.
"""

import numpy as np

BRAILLE_BASE = 0x2800
BRAILLE_BLANK = chr(BRAILLE_BASE)

# Bit for the dot at (x, y) inside a 2-wide, 4-tall braille cell, indexed [y][x].
_BRAILLE_BITS = np.array(
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint32,
)


def braille(dots: np.ndarray) -> str:
    """Render a boolean dot mask of shape (rows * 4, columns * 2)."""
    height, width = dots.shape
    if height % 4 or width % 2:
        raise ValueError("dot mask must be a multiple of 4 rows by 2 columns")
    cells = dots.reshape(height // 4, 4, width // 2, 2).astype(np.uint32)
    bits = np.einsum("ryxc,yc->rx", cells, _BRAILLE_BITS)
    return _codepoints_to_text(BRAILLE_BASE + bits)


def ramp_text(levels: np.ndarray, ramp: str) -> str:
    """Render ink levels (0 = paper) with a densest-to-lightest character ramp."""
    if int(levels.max(initial=0)) >= len(ramp):
        raise ValueError("level out of range for ramp")
    lightest_first = np.array([ord(ch) for ch in reversed(ramp)], dtype=np.uint32)
    return _codepoints_to_text(lightest_first[levels])


def _codepoints_to_text(codepoints: np.ndarray) -> str:
    rows, columns = codepoints.shape
    buffer = np.empty((rows, columns + 1), dtype="<u4")
    buffer[:, :columns] = codepoints
    buffer[:, columns] = ord("\n")
    return buffer.tobytes().decode("utf-32-le")[:-1]
