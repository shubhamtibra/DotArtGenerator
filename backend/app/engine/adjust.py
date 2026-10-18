"""Luminance extraction and tone/detail adjustments.

All functions work on float32 luminance images with values in [0, 1].
"""

import cv2
import numpy as np

from .options import RenderOptions

_REC601 = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a uint8 RGB image as float32 in [0, 1]."""
    return (rgb.astype(np.float32) @ _REC601) / 255.0


def to_uint8(v: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(v * 255.0), 0, 255).astype(np.uint8)


def otsu_threshold(v: np.ndarray) -> int:
    threshold, _ = cv2.threshold(to_uint8(v), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return int(threshold)


def apply_adjustments(v: np.ndarray, options: RenderOptions) -> np.ndarray:
    """Apply the optional adjustments in their fixed order."""
    if options.normalize:
        v = _percentile_stretch(v)
    if options.equalize:
        v = _clahe(v)
    if options.brightness:
        v = _clip(v + options.brightness / 100 * 0.5)
    if options.contrast:
        factor = 4 ** (options.contrast / 100)
        v = _clip((v - 0.5) * factor + 0.5)
    if options.gamma != 1.0:
        v = _clip(v ** (1.0 / options.gamma))
    if options.sharpen:
        blurred = cv2.GaussianBlur(v, (0, 0), sigmaX=1.0)
        v = _clip(v + options.sharpen * (v - blurred))
    if options.edges:
        v = _clip(v * (1.0 - options.edges * _edge_mask(v)))
    return v.astype(np.float32, copy=False)


def _clip(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.0, 1.0)


def _percentile_stretch(v: np.ndarray) -> np.ndarray:
    low, high = (float(p) for p in np.percentile(v, (1, 99)))
    if high - low < 1e-3:
        return v
    return _clip((v - low) / (high - low))


def _clahe(v: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(to_uint8(v)).astype(np.float32) / 255.0


def _edge_mask(v: np.ndarray) -> np.ndarray:
    """Canny edges (1.0 on edges) with thresholds derived from the median."""
    gray = to_uint8(v)
    median = float(np.median(gray))
    lower = max(0.0, 0.66 * median)
    upper = min(255.0, 1.33 * median)
    return (cv2.Canny(gray, lower, upper) > 0).astype(np.float32)
