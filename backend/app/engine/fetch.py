"""SSRF-safe download of images from user-supplied URLs.

Every hop (the original URL and each redirect target) must use http(s), carry
no credentials and resolve only to globally routable addresses. Redirects are
followed manually so each target is re-validated, and the body is streamed with
a hard byte cap and an overall deadline.
"""

import hashlib
import ipaddress
import socket
import time
from contextlib import closing
from urllib.parse import urljoin, urlsplit

import requests

from .errors import FetchError, ImageTooLargeError, UrlNotAllowedError

USER_AGENT = "DotArtGenerator/2.0"
TIMEOUT = (5, 10)  # connect, read (per socket operation)
TOTAL_DEADLINE_SECONDS = 20
MAX_REDIRECTS = 3
DEFAULT_CACHE_SECONDS = 600
_CHUNK_SIZE = 64 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def fetch_image_bytes(
    url: str,
    *,
    max_bytes: int,
    allow_private: bool = False,
    cache=None,
    cache_seconds: int = DEFAULT_CACHE_SECONDS,
    session: requests.Session | None = None,
) -> bytes:
    """Download ``url`` and return its body.

    ``cache`` is any object with Django-cache-style ``get(key)`` /
    ``set(key, value, timeout)``; repeated renders of the same URL then skip
    the network.
    """
    key = cache_key(url)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    if session is None:
        with _new_session() as own_session:
            data = _download(url, own_session, max_bytes, allow_private)
    else:
        data = _download(url, session, max_bytes, allow_private)

    if cache is not None:
        cache.set(key, data, cache_seconds)
    return data


def cache_key(url: str) -> str:
    return "dotart:url:" + hashlib.sha256(url.encode("utf-8")).hexdigest()


def validate_url(url: str, *, allow_private: bool = False) -> None:
    """Raise ``UrlNotAllowedError`` unless ``url`` is safe to fetch."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UrlNotAllowedError("That doesn't look like a valid URL.") from exc

    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise UrlNotAllowedError("Only http:// and https:// image URLs are supported.")
    if not parts.hostname:
        raise UrlNotAllowedError("The URL must include a host name.")
    if parts.username is not None or parts.password is not None:
        raise UrlNotAllowedError("URLs containing credentials are not allowed.")
    if allow_private:
        return

    for address in _resolve(parts.hostname, port or _DEFAULT_PORTS[scheme]):
        if not _is_public_address(address):
            raise UrlNotAllowedError(
                "That URL points to a private or local network address, which is not allowed."
            )


def _new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # ignore proxy env vars and ~/.netrc credentials
    return session


def _download(url: str, session: requests.Session, max_bytes: int, allow_private: bool) -> bytes:
    deadline = time.monotonic() + TOTAL_DEADLINE_SECONDS
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validate_url(current, allow_private=allow_private)
        with closing(_get(session, current)) as response:
            location = response.headers.get("Location")
            if response.status_code in _REDIRECT_STATUSES and location:
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise FetchError(f"The image server responded with HTTP {response.status_code}.")
            return _read_body(response, max_bytes, deadline)
    raise FetchError(f"The URL redirected more than {MAX_REDIRECTS} times.")


def _get(session: requests.Session, url: str) -> requests.Response:
    try:
        return session.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
        )
    except requests.Timeout as exc:
        raise FetchError("Timed out while connecting to the image server.") from exc
    except requests.RequestException as exc:
        raise FetchError("Could not download the image from that URL.") from exc


def _read_body(response: requests.Response, max_bytes: int, deadline: float) -> bytes:
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ImageTooLargeError(_too_large_message(max_bytes))

    chunks = []
    received = 0
    try:
        for chunk in response.iter_content(_CHUNK_SIZE):
            received += len(chunk)
            if received > max_bytes:
                raise ImageTooLargeError(_too_large_message(max_bytes))
            if time.monotonic() > deadline:
                raise FetchError("Timed out while downloading the image.")
            chunks.append(chunk)
    except requests.RequestException as exc:
        raise FetchError("The connection failed while downloading the image.") from exc
    return b"".join(chunks)


def _resolve(host: str, port: int) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise FetchError(f"Could not resolve the host {host!r}.") from exc
    return {info[4][0] for info in infos}


def _is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop any IPv6 zone id
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def _too_large_message(max_bytes: int) -> str:
    return f"The image is larger than the {max_bytes / (1024 * 1024):g} MB limit."
