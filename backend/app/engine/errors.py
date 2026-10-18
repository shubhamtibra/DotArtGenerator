"""Errors raised by the rendering engine.

Each error carries a stable, machine-readable ``code``. The web layer maps codes
to HTTP statuses, which keeps the engine itself transport-agnostic.
"""


class EngineError(Exception):
    code = "engine_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(EngineError):
    code = "invalid_image"


class ImageTooLargeError(EngineError):
    code = "too_large"


class UrlNotAllowedError(EngineError):
    code = "url_not_allowed"


class FetchError(EngineError):
    code = "fetch_failed"
