"""Quantise an ink image (0 = paper, 1 = full ink) to integer levels 0..L-1.

Error diffusion is inherently sequential along a row, so the inner loops run on
plain Python floats (much faster than indexing NumPy scalars) while the
diffusion into the rows below is applied to whole rows with NumPy.
"""

import numpy as np


def _bayer_matrix(size: int) -> np.ndarray:
    m = np.array([[0, 2], [3, 1]])
    while m.shape[0] < size:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


# Thresholds strictly inside (0, 1), centred in each of the 64 buckets.
_BAYER_8 = ((_bayer_matrix(8) + 0.5) / 64).astype(np.float32)


def quantize(ink: np.ndarray, levels: int, method: str) -> np.ndarray:
    """Return a uint8 array of levels in ``0..levels-1`` with the ink's shape."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    if method == "none":
        return _nearest(ink, levels)
    if method == "ordered":
        return _ordered(ink, levels)
    if method == "floyd_steinberg":
        return _floyd_steinberg(ink, levels)
    if method == "atkinson":
        return _atkinson(ink, levels)
    raise ValueError(f"unknown dither method: {method!r}")


def _nearest(ink: np.ndarray, levels: int) -> np.ndarray:
    top = levels - 1
    return np.clip(np.rint(ink * top), 0, top).astype(np.uint8)


def _ordered(ink: np.ndarray, levels: int) -> np.ndarray:
    top = levels - 1
    height, width = ink.shape
    bayer = _BAYER_8[np.arange(height)[:, None] % 8, np.arange(width)[None, :] % 8]
    return np.clip(np.floor(ink * top + bayer), 0, top).astype(np.uint8)


def _floyd_steinberg(ink: np.ndarray, levels: int) -> np.ndarray:
    """Serpentine Floyd-Steinberg: 7/16 right, 3/16 down-left, 5/16 down, 1/16 down-right."""
    height, width = ink.shape
    top = levels - 1
    step = 1.0 / top
    work = ink.astype(np.float64)  # copy: accumulates diffused error
    out = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        # Odd rows run right-to-left; reversed views keep the kernel code identical.
        backwards = y % 2 == 1
        row = work[y, ::-1] if backwards else work[y]
        quantized = []
        errors = []
        carry = 0.0
        for value in row.tolist():
            value += carry
            level = int(value * top + 0.5)
            if level < 0:
                level = 0
            elif level > top:
                level = top
            error = value - level * step
            quantized.append(level)
            errors.append(error)
            carry = error * 0.4375
        (out[y, ::-1] if backwards else out[y])[:] = quantized

        if y + 1 < height:
            below = work[y + 1, ::-1] if backwards else work[y + 1]
            err = np.asarray(errors)
            below[:-1] += err[1:] * 0.1875
            below += err * 0.3125
            below[1:] += err[:-1] * 0.0625
    return out


def _atkinson(ink: np.ndarray, levels: int) -> np.ndarray:
    """Atkinson: 1/8 of the error to six neighbours (x+1, x+2, three below, two below)."""
    height, width = ink.shape
    top = levels - 1
    step = 1.0 / top
    work = ink.astype(np.float64)
    out = np.empty((height, width), dtype=np.uint8)

    for y in range(height):
        quantized = []
        shares = []
        carry_next = carry_after = 0.0  # error owed to x+1 and x+2
        for value in work[y].tolist():
            value += carry_next
            level = int(value * top + 0.5)
            if level < 0:
                level = 0
            elif level > top:
                level = top
            share = (value - level * step) * 0.125
            quantized.append(level)
            shares.append(share)
            carry_next = carry_after + share
            carry_after = share
        out[y] = quantized

        err = np.asarray(shares)
        if y + 1 < height:
            below = work[y + 1]
            below[:-1] += err[1:]
            below += err
            below[1:] += err[:-1]
        if y + 2 < height:
            work[y + 2] += err
    return out
