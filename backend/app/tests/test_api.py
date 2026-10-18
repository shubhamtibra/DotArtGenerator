"""HTTP-level tests for /api: success shapes, validation and the error contract."""

import base64
import io
import logging
from unittest import mock

import requests
from django.core.cache import caches
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from PIL import Image
from rest_framework.test import APITestCase
from rest_framework.throttling import AnonRateThrottle

from .support import PUBLIC_IP, FakeResponse, fake_dns, gradient, noise, png

GENERATE = "/api/generate/"
IMAGE_URL = "https://images.example/photo.png"
RESPONSE_KEYS = {"art", "mode", "columns", "rows", "options", "source", "color_map",
                 "processed_image", "elapsed_ms"}


def upload(data: bytes, name="image.png", content_type="image/png"):
    return SimpleUploadedFile(name, data, content_type=content_type)


class ApiTestCase(APITestCase):
    def setUp(self):
        for cache in caches.all():
            cache.clear()
        self.image = png(gradient(300, 200))
        # Django logs every 4xx/5xx response; these tests provoke them on purpose.
        request_logger = logging.getLogger("django.request")
        self.addCleanup(request_logger.setLevel, request_logger.level)
        request_logger.setLevel(logging.CRITICAL)

    def generate(self, path=GENERATE, **fields):
        fields.setdefault("file", upload(self.image))
        return self.client.post(path, fields, format="multipart")

    def assertError(self, response, status, code):
        self.assertEqual(response.status_code, status, response.content[:300])
        self.assertEqual(response["Content-Type"], "application/json")
        body = response.json()
        self.assertEqual(body["code"], code)
        self.assertIsInstance(body["error"], str)
        self.assertTrue(body["error"])
        self.assertLessEqual(set(body), {"error", "code", "details"})
        return body


class GenerateTests(ApiTestCase):
    def test_each_mode_renders_with_the_documented_shape(self):
        for mode in ("braille", "ascii", "blocks"):
            with self.subTest(mode=mode):
                response = self.generate(mode=mode, width="60")
                self.assertEqual(response.status_code, 200, response.content[:300])
                body = response.json()
                self.assertEqual(set(body), RESPONSE_KEYS)
                self.assertEqual((body["mode"], body["columns"], body["rows"]), (mode, 60, 20))
                lines = body["art"].split("\n")
                self.assertEqual(len(lines), 20)
                self.assertTrue(all(len(line) == 60 for line in lines))
                self.assertEqual(body["source"], {"width": 300, "height": 200, "format": "PNG"})
                self.assertIsNone(body["color_map"])
                self.assertIsNone(body["processed_image"])
                self.assertIsInstance(body["elapsed_ms"], int)

    def test_defaults_apply_to_multipart_requests(self):
        options = self.generate().json()["options"]
        self.assertEqual(options["mode"], "braille")
        self.assertEqual(options["width"], 100)
        self.assertEqual(options["dither"], "floyd_steinberg")
        self.assertTrue(options["normalize"])  # omitted booleans must not become False
        self.assertFalse(options["invert"])
        self.assertTrue(options["threshold_auto"])
        self.assertEqual(options["gamma"], 1.0)
        self.assertEqual(options["char_aspect"], 0.5)

    def test_string_booleans_and_numbers(self):
        body = self.generate(invert="true", equalize="on", normalize="0", color="1",
                             include_processed="True", width="80", gamma="1.5",
                             brightness="-20", sharpen="0.5").json()
        options = body["options"]
        self.assertTrue(options["invert"])
        self.assertTrue(options["equalize"])
        self.assertFalse(options["normalize"])
        self.assertEqual((options["width"], options["gamma"], options["brightness"]), (80, 1.5, -20))
        color_map = Image.open(io.BytesIO(base64.b64decode(body["color_map"])))
        self.assertEqual(color_map.size, (body["columns"], body["rows"]))
        self.assertIsNotNone(body["processed_image"])

    def test_threshold_accepts_auto_blank_and_integers(self):
        for raw, auto in (("auto", True), ("", True), ("AUTO", True), ("120", False), ("0", False)):
            with self.subTest(threshold=raw):
                options = self.generate(dither="none", threshold=raw).json()["options"]
                self.assertEqual(options["threshold_auto"], auto)
                if not auto:
                    self.assertEqual(options["threshold"], int(raw))
                self.assertIsInstance(options["threshold"], int)

    def test_custom_charset_keeps_whitespace(self):
        body = self.generate(mode="ascii", charset="custom", custom_charset="#o. ").json()
        self.assertEqual(body["options"]["custom_charset"], "#o. ")
        self.assertLessEqual(set(body["art"].replace("\n", "")), set("#o. "))

    def test_custom_charset_is_ignored_outside_ascii(self):
        response = self.generate(mode="braille", charset="custom", custom_charset="")
        self.assertEqual(response.status_code, 200)

    def test_upload_alias_and_optional_trailing_slash(self):
        for path in ("/api/upload/", "/api/generate", "/api/upload"):
            with self.subTest(path=path):
                self.assertEqual(self.generate(path=path).status_code, 200)

    def test_garbage_bytes_are_invalid_image(self):
        for data in (b"this is not an image", b""):
            with self.subTest(data=data):
                self.assertError(self.generate(file=upload(data)), 400, "invalid_image")


class ValidationTests(ApiTestCase):
    def test_invalid_options_are_reported_per_field(self):
        cases = [
            ({"width": "5"}, "width"),
            ({"width": "401"}, "width"),
            ({"width": "wide"}, "width"),
            ({"mode": "sketch"}, "mode"),
            ({"dither": "blur"}, "dither"),
            ({"charset": "fancy"}, "charset"),
            ({"threshold": "300"}, "threshold"),
            ({"threshold": "dark"}, "threshold"),
            ({"brightness": "101"}, "brightness"),
            ({"contrast": "-101"}, "contrast"),
            ({"gamma": "0.1"}, "gamma"),
            ({"gamma": "nan"}, "gamma"),
            ({"sharpen": "4"}, "sharpen"),
            ({"edges": "1.5"}, "edges"),
            ({"char_aspect": "0.1"}, "char_aspect"),
            ({"invert": "maybe"}, "invert"),
            ({"mode": "ascii", "charset": "custom"}, "custom_charset"),
            ({"mode": "ascii", "charset": "custom", "custom_charset": "x"}, "custom_charset"),
            ({"mode": "ascii", "charset": "custom", "custom_charset": "a\nb"}, "custom_charset"),
            ({"mode": "ascii", "charset": "custom", "custom_charset": "a\tb"}, "custom_charset"),
            ({"mode": "ascii", "charset": "custom", "custom_charset": "x" * 129}, "custom_charset"),
        ]
        for fields, field in cases:
            with self.subTest(fields=fields):
                body = self.assertError(self.generate(**fields), 400, "invalid_options")
                self.assertIn(field, body["details"])
                self.assertTrue(body["error"].startswith(f"{field}: "), body["error"])

    def test_missing_source(self):
        response = self.client.post(GENERATE, {"mode": "ascii"}, format="multipart")
        body = self.assertError(response, 400, "missing_source")
        self.assertNotIn("details", body)
        self.assertError(self.client.post(GENERATE, {}, format="json"), 400, "missing_source")

    def test_file_and_url_together_are_rejected(self):
        body = self.assertError(self.generate(url=IMAGE_URL), 400, "invalid_options")
        self.assertIn("non_field_errors", body["details"])

    def test_url_length_is_limited(self):
        long_url = "https://images.example/" + "a" * 2100
        body = self.assertError(self.client.post(GENERATE, {"url": long_url}, format="json"),
                                400, "invalid_options")
        self.assertIn("url", body["details"])

    def test_malformed_and_unsupported_bodies(self):
        self.assertError(self.client.post(GENERATE, data="{oops", content_type="application/json"),
                         400, "invalid_options")
        self.assertError(self.client.post(GENERATE, data="hello", content_type="text/plain"),
                         415, "unsupported_media_type")


@mock.patch("socket.getaddrinfo", side_effect=fake_dns({"images.example": [PUBLIC_IP],
                                                       "intranet.example": ["10.0.0.9"]}))
class UrlSourceTests(ApiTestCase):
    def post_url(self, url=IMAGE_URL, **fields):
        return self.client.post(GENERATE, {"url": url, **fields}, format="json")

    def test_fetches_url_images_and_caches_them(self, _dns):
        with mock.patch.object(requests.Session, "get", return_value=FakeResponse(body=self.image)) as get:
            first = self.post_url(mode="ascii", width=50, invert=True)
            second = self.post_url(mode="braille", width=40)
        self.assertEqual(first.status_code, 200, first.content[:300])
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["source"]["format"], "PNG")
        self.assertTrue(first.json()["options"]["invert"])
        self.assertEqual(get.call_count, 1)

    def test_form_encoded_url_requests_work(self, _dns):
        with mock.patch.object(requests.Session, "get", return_value=FakeResponse(body=self.image)):
            response = self.client.post(GENERATE, {"url": IMAGE_URL, "mode": "blocks"})
        self.assertEqual(response.status_code, 200, response.content[:300])

    def test_unsafe_urls_are_not_allowed(self, _dns):
        with mock.patch.object(requests.Session, "get") as get:
            for url in ("http://127.0.0.1/", "http://10.0.0.8/a.png", "http://169.254.169.254/latest/",
                        "file:///etc/passwd", "http://intranet.example/a.png", "ftp://images.example/a"):
                with self.subTest(url=url):
                    self.assertError(self.post_url(url), 400, "url_not_allowed")
        get.assert_not_called()

    @override_settings(ALLOW_PRIVATE_URLS=True)
    def test_private_urls_can_be_enabled_for_development(self, _dns):
        with mock.patch.object(requests.Session, "get", return_value=FakeResponse(body=self.image)):
            self.assertEqual(self.post_url("http://127.0.0.1:9000/a.png").status_code, 200)

    def test_redirect_to_private_address_is_blocked(self, _dns):
        redirect = FakeResponse(302, headers={"Location": "http://10.0.0.8/secret.png"})
        with mock.patch.object(requests.Session, "get", return_value=redirect) as get:
            self.assertError(self.post_url(), 400, "url_not_allowed")
        self.assertEqual(get.call_count, 1)

    def test_oversized_download_is_too_large(self, _dns):
        response = FakeResponse(body=b"x", headers={"Content-Length": str(500 * 1024 * 1024)})
        with mock.patch.object(requests.Session, "get", return_value=response):
            self.assertError(self.post_url(), 413, "too_large")

    def test_network_failures_are_bad_gateway(self, _dns):
        for outcome in (requests.ConnectionError("refused"), FakeResponse(404)):
            kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with self.subTest(outcome=outcome), mock.patch.object(requests.Session, "get", **kwargs):
                self.assertError(self.post_url(), 502, "fetch_failed")
        self.assertError(self.post_url("https://unknown-host.example/a.png"), 502, "fetch_failed")

    def test_non_image_download_is_invalid_image(self, _dns):
        page = FakeResponse(body=b"<html>not an image</html>", headers={"Content-Type": "text/html"})
        with mock.patch.object(requests.Session, "get", return_value=page):
            self.assertError(self.post_url(), 400, "invalid_image")


class SizeLimitTests(ApiTestCase):
    @override_settings(MAX_UPLOAD_BYTES=4 * 1024)
    def test_file_over_the_upload_limit(self):
        big_png = png(noise(64, 64))
        self.assertGreater(len(big_png), 4 * 1024)
        body = self.assertError(self.generate(file=upload(big_png)), 413, "too_large")
        self.assertIn("MB", body["error"])

    @override_settings(MAX_UPLOAD_BYTES=1024)
    def test_declared_body_size_is_rejected_before_parsing(self):
        payload = b"\0" * (1024 * 1024 + 4096)
        self.assertError(self.generate(file=upload(payload, "big.bin")), 413, "too_large")

    @override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=256)
    def test_oversized_form_fields_are_too_large(self):
        response = self.generate(mode="ascii", charset="custom", custom_charset="x" * 1000)
        self.assertError(response, 413, "too_large")


class ErrorContractTests(ApiTestCase):
    def test_health(self):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "version": "2.0.0"})

    def test_wrong_method_is_json_405(self):
        self.assertError(self.client.get(GENERATE), 405, "method_not_allowed")
        self.assertError(self.client.put(GENERATE, {}), 405, "method_not_allowed")
        self.assertError(self.client.post("/api/health/"), 405, "method_not_allowed")

    def test_unknown_api_paths_are_json_404(self):
        self.assertError(self.client.get("/api/nope/"), 404, "not_found")
        self.assertError(self.client.post("/api/v3/generate/", {}), 404, "not_found")

    def test_throttling_is_json_429(self):
        with mock.patch.object(AnonRateThrottle, "rate", "2/min", create=True):
            for _ in range(2):
                self.assertEqual(self.generate().status_code, 200)
            response = self.generate()
        body = self.assertError(response, 429, "throttled")
        self.assertIn("Retry-After", response)
        self.assertGreater(body["details"]["retry_after"], 0)

    def test_unexpected_errors_are_json_500_without_internals(self):
        with mock.patch("app.views.generate", side_effect=RuntimeError("secret internals")), \
                self.assertLogs("app.exceptions", "ERROR"):
            response = self.generate()
        self.assertError(response, 500, "server_error")
        self.assertNotIn(b"secret internals", response.content)
        self.assertNotIn(b"Traceback", response.content)

    @override_settings(CORS_ALLOW_ALL_ORIGINS=False)
    def test_cors_allows_configured_origins_and_vercel_deployments(self):
        preflight = self.client.options(GENERATE, HTTP_ORIGIN="http://localhost:3000",
                                        HTTP_ACCESS_CONTROL_REQUEST_METHOD="POST")
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight["Access-Control-Allow-Origin"], "http://localhost:3000")
        vercel = "https://dot-art-git-main.vercel.app"
        allowed = self.client.get("/api/health/", HTTP_ORIGIN=vercel)
        self.assertEqual(allowed["Access-Control-Allow-Origin"], vercel)
        denied = self.client.get("/api/health/", HTTP_ORIGIN="https://evil.example")
        self.assertNotIn("Access-Control-Allow-Origin", denied)
