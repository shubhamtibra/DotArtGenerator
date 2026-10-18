"""SSRF guard and download behaviour of the URL fetcher. No real network access."""

from unittest import TestCase, mock

import requests

from app.engine import FetchError, ImageTooLargeError, UrlNotAllowedError
from app.engine.fetch import USER_AGENT, cache_key, fetch_image_bytes, validate_url

from .support import PUBLIC_IP, DictCache, FakeResponse, FakeSession, fake_dns, gradient, png

IMAGE_URL = "https://images.example/cat.png"
DNS = {
    "images.example": [PUBLIC_IP],
    "v6.example": ["2606:4700::1111"],
    "localhost": ["127.0.0.1", "::1"],
    "internal.example": ["10.0.0.7"],
    "mixed.example": [PUBLIC_IP, "127.0.0.1"],  # any private answer is enough to refuse
}


class FetchTestCase(TestCase):
    def setUp(self):
        patcher = mock.patch("socket.getaddrinfo", side_effect=fake_dns(DNS))
        self.getaddrinfo = patcher.start()
        self.addCleanup(patcher.stop)
        self.body = png(gradient(16, 16))

    def fetch(self, session, url=IMAGE_URL, **kwargs):
        kwargs.setdefault("max_bytes", 1024 * 1024)
        return fetch_image_bytes(url, session=session, **kwargs)


class ValidateUrlTests(FetchTestCase):
    def test_rejects_non_http_schemes(self):
        for url in ("file:///etc/passwd", "ftp://images.example/a.png", "gopher://images.example/",
                    "data:image/png;base64,AAAA", "javascript:alert(1)", "images.example/a.png"):
            with self.subTest(url=url), self.assertRaises(UrlNotAllowedError):
                validate_url(url)

    def test_rejects_credentials_and_missing_host(self):
        for url in ("http://user:secret@images.example/a.png", "http://user@images.example/",
                    "http:///a.png", "http://images.example:99999/"):
            with self.subTest(url=url), self.assertRaises(UrlNotAllowedError):
                validate_url(url)

    def test_rejects_private_loopback_and_metadata_addresses(self):
        for url in ("http://127.0.0.1/", "http://localhost:8000/", "http://10.1.2.3/a.png",
                    "http://169.254.169.254/latest/meta-data/", "http://192.168.1.1/",
                    "http://172.16.0.1/", "http://100.64.0.1/", "http://0.0.0.0/", "http://224.0.0.1/",
                    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://[fe80::1]/",
                    "http://internal.example/a.png", "http://mixed.example/a.png"):
            with self.subTest(url=url), self.assertRaises(UrlNotAllowedError):
                validate_url(url)

    def test_accepts_public_hosts(self):
        for url in (IMAGE_URL, "http://images.example:8080/a.png", f"http://{PUBLIC_IP}/", "https://v6.example/"):
            with self.subTest(url=url):
                validate_url(url)

    def test_allow_private_skips_address_checks_only(self):
        validate_url("http://127.0.0.1:8000/a.png", allow_private=True)
        with self.assertRaises(UrlNotAllowedError):
            validate_url("file:///etc/passwd", allow_private=True)

    def test_unresolvable_host_is_a_fetch_failure(self):
        with self.assertRaises(FetchError):
            validate_url("https://no-such-host.example/")


class DownloadTests(FetchTestCase):
    def test_downloads_with_safe_request_settings(self):
        session = FakeSession({IMAGE_URL: FakeResponse(body=self.body)})
        self.assertEqual(self.fetch(session), self.body)
        (url, kwargs), = session.calls
        self.assertEqual(url, IMAGE_URL)
        self.assertTrue(kwargs["stream"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (5, 10))
        self.assertEqual(kwargs["headers"]["User-Agent"], USER_AGENT)

    def test_cache_prevents_refetching(self):
        cache = DictCache()
        session = FakeSession({IMAGE_URL: FakeResponse(body=self.body)})
        self.assertEqual(self.fetch(session, cache=cache), self.body)
        self.assertEqual(self.fetch(session, cache=cache), self.body)
        self.assertEqual(len(session.calls), 1)
        self.assertIn(cache_key(IMAGE_URL), cache.store)

    def test_declared_oversize_is_rejected_before_reading(self):
        response = FakeResponse(body=b"x" * 10, headers={"Content-Length": str(50 * 1024 * 1024)})
        with self.assertRaises(ImageTooLargeError):
            self.fetch(FakeSession({IMAGE_URL: response}))

    def test_streamed_oversize_is_rejected(self):
        response = FakeResponse(body=b"x" * 5000, chunk_size=1000)
        with self.assertRaises(ImageTooLargeError):
            self.fetch(FakeSession({IMAGE_URL: response}), max_bytes=2500)

    def test_follows_relative_redirects(self):
        target = "https://images.example/real/cat.png"
        session = FakeSession({
            IMAGE_URL: FakeResponse(302, headers={"Location": "/real/cat.png"}),
            target: FakeResponse(body=self.body),
        })
        self.assertEqual(self.fetch(session), self.body)
        self.assertEqual([url for url, _ in session.calls], [IMAGE_URL, target])

    def test_redirect_to_private_address_is_blocked(self):
        session = FakeSession({
            IMAGE_URL: FakeResponse(301, headers={"Location": "http://169.254.169.254/latest/meta-data/"}),
        })
        with self.assertRaises(UrlNotAllowedError):
            self.fetch(session)
        self.assertEqual(len(session.calls), 1)

    def test_redirect_chain_is_limited(self):
        hops = [f"https://images.example/{i}" for i in range(5)]
        responses = {url: FakeResponse(302, headers={"Location": nxt}) for url, nxt in zip(hops, hops[1:])}
        with self.assertRaises(FetchError):
            self.fetch(FakeSession(responses), url=hops[0])

    def test_http_errors_and_network_failures(self):
        for outcome in (FakeResponse(404), FakeResponse(500), requests.ConnectionError("refused"),
                        requests.Timeout("slow")):
            with self.subTest(outcome=outcome), self.assertRaises(FetchError):
                self.fetch(FakeSession({IMAGE_URL: outcome}))

    def test_failed_downloads_are_not_cached(self):
        cache = DictCache()
        with self.assertRaises(FetchError):
            self.fetch(FakeSession({IMAGE_URL: FakeResponse(503)}), cache=cache)
        self.assertEqual(cache.store, {})
