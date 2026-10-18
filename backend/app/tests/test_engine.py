"""Unit tests for the Django-free rendering engine."""

import base64
import io
import time
from unittest import TestCase

import numpy as np
from PIL import Image

from app.engine import (
    ImageTooLargeError,
    InvalidImageError,
    RenderOptions,
    decode_image,
    generate,
    grid_size,
)
from app.engine.adjust import apply_adjustments
from app.engine.dither import quantize
from app.engine.options import ASCII_CHARSETS, BLOCK_RAMP, DITHERS, MAX_ROWS, MODES
from app.engine.render import BRAILLE_BLANK

from .support import encode, gradient, noise, png, solid

BRAILLE_BIT = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (1, 0): 0x08,
    (1, 1): 0x10, (1, 2): 0x20, (0, 3): 0x40, (1, 3): 0x80,
}
EXACT = {"dither": "none", "threshold": 128, "normalize": False}


def lines_of(result):
    return result["art"].split("\n")


def decode_png_b64(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


class GridTests(TestCase):
    def test_every_mode_and_dither_has_exact_dimensions(self):
        image = gradient(300, 200)
        for mode in MODES:
            for dither in DITHERS:
                with self.subTest(mode=mode, dither=dither):
                    result = generate(image, RenderOptions(mode=mode, dither=dither, width=60))
                    self.assertEqual(result["columns"], 60)
                    self.assertEqual(result["rows"], 20)
                    lines = lines_of(result)
                    self.assertEqual(len(lines), 20)
                    self.assertTrue(all(len(line) == 60 for line in lines))
                    self.assertFalse(result["art"].endswith("\n"))

    def test_braille_rows_follow_square_dots(self):
        # px_w = 2 * cols; rows = round(px_w * H / W / 4)
        grid = grid_size(400, 300, RenderOptions(mode="braille", width=100))
        self.assertEqual((grid.columns, grid.rows, grid.px_width, grid.px_height), (100, 38, 200, 152))

    def test_ascii_rows_use_char_aspect(self):
        for aspect, rows in ((0.5, 20), (1.0, 40), (0.3, 12)):
            with self.subTest(aspect=aspect):
                grid = grid_size(200, 100, RenderOptions(mode="ascii", width=80, char_aspect=aspect))
                self.assertEqual((grid.columns, grid.rows), (80, rows))
                self.assertEqual((grid.px_width, grid.px_height), (80, rows))

    def test_rows_are_capped_and_columns_shrink_to_keep_aspect(self):
        tall = solid(10, 1000, 128)
        braille = generate(tall, RenderOptions(mode="braille", width=400))
        self.assertEqual((braille["columns"], braille["rows"]), (8, MAX_ROWS))
        ascii_art = generate(tall, RenderOptions(mode="ascii", width=400))
        self.assertEqual((ascii_art["columns"], ascii_art["rows"]), (8, MAX_ROWS))
        self.assertTrue(all(len(line) == 8 for line in lines_of(ascii_art)))

    def test_extreme_aspect_ratios_never_produce_empty_grids(self):
        for width, height in ((1, 1), (2000, 1), (1, 2000)):
            with self.subTest(size=(width, height)):
                result = generate(solid(width, height, 0), RenderOptions(mode="braille", width=10))
                self.assertGreaterEqual(result["columns"], 1)
                self.assertGreaterEqual(result["rows"], 1)


class BrailleTests(TestCase):
    def test_bit_layout_for_every_dot(self):
        for (x, y), bit in BRAILLE_BIT.items():
            with self.subTest(dot=(x, y)):
                image = solid(2, 4, 255)
                image[y, x] = 0
                result = generate(image, RenderOptions(width=1, **EXACT))
                self.assertEqual(result["art"], chr(0x2800 + bit))

    def test_single_dot_at_bottom_right(self):
        image = solid(2, 4, 255)
        image[3, 1] = 0
        self.assertEqual(generate(image, RenderOptions(width=1, **EXACT))["art"], "⢀")

    def test_blank_cells_use_u2800_never_spaces(self):
        result = generate(solid(40, 40, 255), RenderOptions(width=20))
        self.assertNotIn(" ", result["art"])
        self.assertEqual(set(result["art"].replace("\n", "")), {BRAILLE_BLANK})
        self.assertTrue(all(len(line) == 20 for line in lines_of(result)))

    def test_black_image_is_fully_inked(self):
        result = generate(solid(40, 40, 0), RenderOptions(width=20))
        self.assertEqual(set(result["art"].replace("\n", "")), {"⣿"})

    def test_invert_flips_every_dot(self):
        binary = (np.random.default_rng(1).random((40, 20)) > 0.5).astype(np.uint8) * 255
        image = np.dstack([binary] * 3)
        plain = generate(image, RenderOptions(width=10, **EXACT))["art"]
        inverted = generate(image, RenderOptions(width=10, invert=True, **EXACT))["art"]
        flipped = "".join(
            ch if ch == "\n" else chr(0x2800 + (0xFF ^ (ord(ch) - 0x2800))) for ch in plain
        )
        self.assertEqual(inverted, flipped)


class RampTests(TestCase):
    def test_paper_renders_as_lightest_character_without_stripping(self):
        result = generate(solid(100, 50, 255), RenderOptions(mode="ascii", width=40))
        self.assertTrue(all(line == " " * 40 for line in lines_of(result)))

    def test_ink_renders_as_densest_character(self):
        cases = (("ascii", "standard", "@"), ("ascii", "detailed", "$"), ("ascii", "simple", "#"),
                 ("blocks", "standard", "█"))
        for mode, charset, densest in cases:
            with self.subTest(mode=mode, charset=charset):
                result = generate(solid(50, 50, 0), RenderOptions(mode=mode, charset=charset, width=20))
                self.assertEqual(set(result["art"].replace("\n", "")), {densest})

    def test_custom_charset_is_used_densest_first(self):
        options = dict(mode="ascii", charset="custom", custom_charset="XO", **EXACT)
        black = generate(solid(20, 20, 0), RenderOptions(width=10, **options))["art"]
        white = generate(solid(20, 20, 255), RenderOptions(width=10, **options))["art"]
        self.assertEqual(set(black.replace("\n", "")), {"X"})
        self.assertEqual(set(white.replace("\n", "")), {"O"})

    def test_gradient_uses_the_whole_ramp(self):
        for mode, ramp in (("ascii", ASCII_CHARSETS["standard"]), ("blocks", BLOCK_RAMP)):
            with self.subTest(mode=mode):
                result = generate(gradient(400, 100), RenderOptions(mode=mode, width=200, dither="none"))
                self.assertEqual(set(result["art"].replace("\n", "")), set(ramp))


class DecodeTests(TestCase):
    def test_transparent_pixels_composite_to_paper(self):
        transparent_black = solid(40, 40, (0, 0, 0, 0))
        for invert in (False, True):
            with self.subTest(invert=invert):
                art = generate(png(transparent_black), RenderOptions(invert=invert, width=20))["art"]
                self.assertEqual(set(art.replace("\n", "")), {BRAILLE_BLANK})
                ascii_art = generate(
                    png(transparent_black), RenderOptions(mode="ascii", invert=invert, width=20)
                )["art"]
                self.assertEqual(set(ascii_art.replace("\n", "")), {" "})

    def test_opaque_region_inside_transparency_still_inks(self):
        image = solid(40, 40, (0, 0, 0, 0))
        image[10:30, 10:30] = (0, 0, 0, 255)
        lines = lines_of(generate(png(image), RenderOptions(width=20, **EXACT)))
        self.assertEqual(lines[0], BRAILLE_BLANK * 20)
        self.assertIn("⣿", lines[len(lines) // 2])

    def test_exif_rotated_jpeg_is_upright(self):
        stored = solid(40, 20, 255)
        stored[:, :20] = 0  # left half black as stored
        exif = Image.Exif()
        exif[0x0112] = 6  # display = stored rotated 90 degrees clockwise
        data = encode(Image.fromarray(stored), "JPEG", exif=exif, quality=95)

        image = decode_image(data)
        self.assertEqual(image.pixels.shape[:2], (40, 20))
        self.assertEqual((image.width, image.height, image.format), (20, 40, "JPEG"))
        self.assertLess(image.pixels[:20].mean(), 40)  # the stored left half is now on top
        self.assertGreater(image.pixels[20:].mean(), 215)

    def test_animated_gif_uses_first_frame(self):
        frames = [Image.new("RGB", (32, 32), "black"), Image.new("RGB", (32, 32), "white")]
        data = encode(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100)
        image = decode_image(data)
        self.assertEqual(image.format, "GIF")
        self.assertLess(image.pixels.mean(), 10)
        result = generate(data, RenderOptions(width=16))
        self.assertEqual(set(result["art"].replace("\n", "")), {"⣿"})

    def test_gif_transparency_becomes_paper(self):
        frame = Image.new("P", (16, 16), 0)
        frame.putpalette([0, 0, 0] * 256)
        data = encode(frame, "GIF", transparency=0)
        art = generate(data, RenderOptions(width=10))["art"]
        self.assertEqual(set(art.replace("\n", "")), {BRAILLE_BLANK})

    def test_sixteen_bit_grayscale_is_scaled_not_clipped(self):
        ramp = np.tile(np.linspace(0, 65535, 64, dtype=np.uint16), (16, 1))
        image = decode_image(encode(Image.fromarray(ramp)))
        self.assertAlmostEqual(float(image.pixels.mean()), 127.5, delta=3)

    def test_large_images_are_downscaled_but_report_original_size(self):
        image = decode_image(png(solid(3000, 300, 90)))
        self.assertEqual(max(image.pixels.shape[:2]), 2048)
        self.assertEqual((image.width, image.height), (3000, 300))

    def test_undecodable_data_is_invalid_image(self):
        truncated_png = png(gradient())[:60]
        for data in (b"", b"definitely not an image", b"\x89PNG\r\n\x1a\n" + b"\0" * 32, truncated_png):
            with self.subTest(data=data[:12]):
                with self.assertRaises(InvalidImageError):
                    decode_image(data)

    def test_pixel_budget_is_enforced_before_decoding(self):
        with self.assertRaises(ImageTooLargeError):
            decode_image(png(solid(100, 100, 0)), max_pixels=5_000)


class AdjustmentTests(TestCase):
    def setUp(self):
        self.v = np.linspace(0.25, 0.75, 64, dtype=np.float32).reshape(8, 8)

    def adjusted(self, **options):
        return apply_adjustments(self.v.copy(), RenderOptions(normalize=False, **options))

    def test_normalize_stretches_to_full_range(self):
        out = apply_adjustments(self.v.copy(), RenderOptions(normalize=True))
        self.assertAlmostEqual(float(out.min()), 0.0, places=5)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_normalize_skips_flat_images(self):
        flat = np.full((8, 8), 0.4, dtype=np.float32)
        self.assertTrue(np.allclose(apply_adjustments(flat, RenderOptions()), 0.4))

    def test_tone_controls(self):
        self.assertTrue(np.allclose(self.adjusted(brightness=50), np.clip(self.v + 0.25, 0, 1)))
        self.assertTrue(np.allclose(self.adjusted(contrast=100), np.clip((self.v - 0.5) * 4 + 0.5, 0, 1)))
        self.assertTrue(np.allclose(self.adjusted(gamma=2.0), self.v ** 0.5))
        self.assertGreater(self.adjusted(gamma=2.0).mean(), self.v.mean())

    def test_edges_only_darken(self):
        step = np.zeros((32, 32), dtype=np.float32)
        step[:, 16:] = 0.6
        step[:, :16] = 0.9
        out = apply_adjustments(step.copy(), RenderOptions(normalize=False, edges=1.0))
        self.assertTrue((out <= step + 1e-6).all())
        self.assertTrue((out < step - 0.5).any())

    def test_every_adjustment_keeps_values_in_range(self):
        options = RenderOptions(
            normalize=True, equalize=True, brightness=-40, contrast=60,
            gamma=0.3, sharpen=3.0, edges=1.0,
        )
        out = apply_adjustments(noise(64, 48)[..., 0].astype(np.float32) / 255, options)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_tiny_images_survive_every_option(self):
        options = dict(equalize=True, sharpen=2.0, edges=1.0, color=True, include_processed=True)
        for mode in MODES:
            with self.subTest(mode=mode):
                result = generate(solid(1, 1, 30), RenderOptions(mode=mode, width=10, **options))
                self.assertEqual(len(lines_of(result)), result["rows"])


class DitherTests(TestCase):
    def test_levels_stay_in_range(self):
        rng = np.random.default_rng(3)
        ink = rng.random((40, 50)).astype(np.float32)
        ink[0, :5] = (0.0, 1.0, 0.0, 1.0, 0.5)
        for method in DITHERS:
            for levels in (2, 5, 10, 70):
                with self.subTest(method=method, levels=levels):
                    q = quantize(ink, levels, method)
                    self.assertEqual(q.dtype, np.uint8)
                    self.assertEqual(q.shape, ink.shape)
                    self.assertLessEqual(int(q.max()), levels - 1)

    def test_extremes_map_to_extreme_levels(self):
        for method in DITHERS:
            with self.subTest(method=method):
                self.assertEqual(int(quantize(np.zeros((9, 9)), 5, method).max()), 0)
                self.assertEqual(int(quantize(np.ones((9, 9)), 5, method).min()), 4)

    def test_tone_is_preserved(self):
        ink = np.full((64, 64), 0.25, dtype=np.float32)
        self.assertEqual(float(quantize(ink, 2, "ordered").mean()), 0.25)
        self.assertAlmostEqual(float(quantize(ink, 2, "floyd_steinberg").mean()), 0.25, delta=0.02)
        self.assertEqual(int(quantize(ink, 2, "none").max()), 0)
        # Atkinson diffuses only 6/8 of the error: midtones hold, highlights/shadows push out.
        self.assertEqual(float(quantize(np.full((64, 64), 0.5), 2, "atkinson").mean()), 0.5)
        self.assertTrue(0.1 < float(quantize(ink, 2, "atkinson").mean()) < 0.25)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            quantize(np.zeros((2, 2)), 2, "blur")


class OutputTests(TestCase):
    def test_auto_threshold_is_resolved_and_reported(self):
        image = solid(40, 40, 50)
        image[:, 20:] = 200
        result = generate(image, RenderOptions(dither="none", normalize=False))
        self.assertTrue(result["options"]["threshold_auto"])
        self.assertIsInstance(result["options"]["threshold"], int)
        self.assertTrue(50 <= result["options"]["threshold"] < 200)

    def test_explicit_threshold_is_reported(self):
        result = generate(gradient(), RenderOptions(dither="none", threshold=77))
        self.assertEqual(result["options"]["threshold"], 77)
        self.assertFalse(result["options"]["threshold_auto"])

    def test_threshold_moves_the_cutoff(self):
        image = solid(20, 40, 100)
        dark = generate(image, RenderOptions(width=10, dither="none", threshold=101, normalize=False))
        light = generate(image, RenderOptions(width=10, dither="none", threshold=99, normalize=False))
        self.assertEqual(set(dark["art"].replace("\n", "")), {"⣿"})
        self.assertEqual(set(light["art"].replace("\n", "")), {BRAILLE_BLANK})

    def test_color_map_is_one_pixel_per_character(self):
        red = solid(60, 30, (255, 0, 0))
        for mode in MODES:
            with self.subTest(mode=mode):
                result = generate(red, RenderOptions(mode=mode, width=20, color=True))
                color_map = decode_png_b64(result["color_map"]).convert("RGB")
                self.assertEqual(color_map.size, (result["columns"], result["rows"]))
                self.assertEqual(color_map.getpixel((0, 0)), (255, 0, 0))

    def test_processed_image_is_grayscale_at_working_resolution(self):
        result = generate(gradient(300, 200), RenderOptions(width=60, include_processed=True))
        processed = decode_png_b64(result["processed_image"])
        self.assertEqual(processed.mode, "L")
        self.assertEqual(processed.size, (120, 80))

    def test_optional_outputs_default_to_none(self):
        result = generate(gradient(), RenderOptions())
        self.assertIsNone(result["color_map"])
        self.assertIsNone(result["processed_image"])

    def test_options_and_source_are_reported(self):
        result = generate(png(gradient(120, 80)), RenderOptions(mode="ascii", width=40, gamma=1.5))
        self.assertEqual(result["source"], {"width": 120, "height": 80, "format": "PNG"})
        self.assertEqual(result["mode"], "ascii")
        expected = {
            "mode", "width", "dither", "threshold", "threshold_auto", "invert", "brightness",
            "contrast", "gamma", "sharpen", "edges", "equalize", "normalize", "charset",
            "custom_charset", "char_aspect", "color", "include_processed",
        }
        self.assertEqual(set(result["options"]), expected)
        self.assertEqual(result["options"]["gamma"], 1.5)


class PerformanceTests(TestCase):
    def test_error_diffusion_on_800x800_is_fast(self):
        ink = np.random.default_rng(0).random((800, 800)).astype(np.float32)
        for method in ("floyd_steinberg", "atkinson"):
            with self.subTest(method=method):
                started = time.perf_counter()
                quantize(ink, 2, method)
                self.assertLess(time.perf_counter() - started, 1.0)

    def test_400_column_braille_render_under_two_seconds(self):
        data = png(gradient(1600, 1200))
        started = time.perf_counter()
        result = generate(data, RenderOptions(width=400, dither="floyd_steinberg"))
        elapsed = time.perf_counter() - started
        self.assertEqual(result["columns"], 400)
        self.assertLess(elapsed, 2.0)
