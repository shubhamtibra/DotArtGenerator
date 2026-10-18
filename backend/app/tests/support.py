"""Synthetic images and network fakes shared by the test modules."""

import io
import ipaddress
import socket

import numpy as np
from PIL import Image

PUBLIC_IP = "93.184.216.34"


def encode(image: Image.Image, fmt: str = "PNG", **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **params)
    return buffer.getvalue()


def png(pixels: np.ndarray) -> bytes:
    return encode(Image.fromarray(pixels))


def solid(width: int, height: int, value) -> np.ndarray:
    channels = len(value) if isinstance(value, tuple) else 3
    return np.full((height, width, channels), value, dtype=np.uint8)


def gradient(width: int = 120, height: int = 80) -> np.ndarray:
    """Diagonal RGB gradient with some colour, so every pipeline stage has work to do."""
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    red = 255 * x * np.ones_like(y)
    green = 255 * y * np.ones_like(x)
    blue = 255 * (x + y) / 2
    return np.dstack([red, green, blue]).astype(np.uint8)


def noise(width: int, height: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def fake_dns(mapping: dict[str, list[str]] | None = None):
    """A ``socket.getaddrinfo`` stand-in: IP literals resolve to themselves."""
    mapping = mapping or {}

    def getaddrinfo(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
            addresses = [host]
        except ValueError:
            if host not in mapping:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            addresses = mapping[host]
        results = []
        for address in addresses:
            if ":" in address:
                results.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0)))
            else:
                results.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)))
        return results

    return getaddrinfo


class FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None, chunk_size=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self._chunk_size = chunk_size
        self.closed = False

    def iter_content(self, chunk_size):
        size = self._chunk_size or chunk_size
        for start in range(0, len(self._body), size):
            yield self._body[start:start + size]

    def close(self):
        self.closed = True


class FakeSession:
    """Serves canned responses (or raises canned exceptions) per URL."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.responses[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class DictCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, timeout=None):
        self.store[key] = value
