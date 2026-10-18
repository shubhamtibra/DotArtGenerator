"""API error types and the DRF exception handler.

Every error response under /api has the shape
``{"error": "<message>", "code": "<code>", "details": {...}}`` (``details`` only
when there is something structured to report).
"""

import logging

from django.core.exceptions import PermissionDenied, RequestDataTooBig, SuspiciousOperation
from django.http import Http404
from rest_framework import exceptions, status
from rest_framework.response import Response
from rest_framework.settings import api_settings
from rest_framework.views import exception_handler as drf_exception_handler

from .engine import EngineError

logger = logging.getLogger(__name__)

SERVER_ERROR_MESSAGE = "Something went wrong on our side. Please try again."


class MissingSource(exceptions.APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "missing_source"
    default_detail = "Provide an image file or an image URL."


class TooLarge(exceptions.APIException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    default_code = "too_large"
    default_detail = "The request is too large."


_ENGINE_STATUS = {
    "invalid_image": status.HTTP_400_BAD_REQUEST,
    "url_not_allowed": status.HTTP_400_BAD_REQUEST,
    "fetch_failed": status.HTTP_502_BAD_GATEWAY,
    "too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
}

# DRF codes that the public API reports under a different name.
_CODE_ALIASES = {"invalid": "invalid_options", "parse_error": "invalid_options"}


def error_payload(code: str, message: str, details=None) -> dict:
    payload = {"error": message, "code": code}
    if details:
        payload["details"] = details
    return payload


def api_exception_handler(exc, context):
    """``REST_FRAMEWORK["EXCEPTION_HANDLER"]``: render any exception as an API error."""
    if isinstance(exc, EngineError):
        return Response(
            error_payload(exc.code, exc.message),
            status=_ENGINE_STATUS.get(exc.code, status.HTTP_400_BAD_REQUEST),
        )

    api_exc = _as_api_exception(exc)
    if api_exc is None:
        request = context.get("request")
        logger.error("Unhandled error on %s", getattr(request, "path", "?"), exc_info=exc)
        return Response(
            error_payload("server_error", SERVER_ERROR_MESSAGE),
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # The default handler sets headers such as Retry-After; only the body is replaced.
    response = drf_exception_handler(api_exc, context)
    response.data = _payload_for(api_exc)
    return response


def _as_api_exception(exc):
    if isinstance(exc, exceptions.APIException):
        return exc
    if isinstance(exc, RequestDataTooBig):
        return TooLarge("The request body is too large.")
    if isinstance(exc, SuspiciousOperation):
        return exceptions.ParseError("The request could not be parsed.")
    if isinstance(exc, Http404):
        return exceptions.NotFound()
    if isinstance(exc, PermissionDenied):
        return exceptions.PermissionDenied()
    return None


def _payload_for(exc: exceptions.APIException) -> dict:
    if isinstance(exc, exceptions.ValidationError):
        details = exc.detail
        if not isinstance(details, dict):
            details = {api_settings.NON_FIELD_ERRORS_KEY: details}
        return error_payload("invalid_options", _first_message(details), details)

    if isinstance(exc, exceptions.Throttled):
        wait = round(exc.wait) if exc.wait else None
        if wait:
            message = f"Too many requests. Please try again in {wait} seconds."
        else:
            message = "Too many requests. Please slow down."
        return error_payload("throttled", message, {"retry_after": wait} if wait else None)

    code = getattr(exc.detail, "code", None) or exc.default_code
    return error_payload(_CODE_ALIASES.get(code, code), str(exc.detail))


def _first_message(details: dict) -> str:
    """One-line summary of the first error, e.g. ``"width: Ensure this value is ..."``."""
    field, errors = next(iter(details.items()), (None, None))
    while isinstance(errors, (list, dict)) and errors:
        errors = errors[0] if isinstance(errors, list) else next(iter(errors.values()))
    message = str(errors) if errors else "Invalid request."
    if field in (None, api_settings.NON_FIELD_ERRORS_KEY):
        return message
    return f"{field}: {message}"
