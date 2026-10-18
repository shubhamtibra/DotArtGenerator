"""HTTP endpoints. Validation lives in ``serializers``, image work in ``engine``."""

import time

from django.conf import settings
from django.core.cache import caches
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.views import APIView

from . import __version__
from .engine import fetch_image_bytes, generate
from .exceptions import SERVER_ERROR_MESSAGE, TooLarge, error_payload
from .serializers import GenerateRequestSerializer

# Allowance for multipart framing and option fields on top of the image itself.
_BODY_OVERHEAD_BYTES = 1024 * 1024


class GenerateView(APIView):
    """``POST /api/generate/``: render an uploaded or linked image as text art."""

    def post(self, request):
        started = time.perf_counter()
        _reject_oversized_body(request)

        serializer = GenerateRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        result = generate(_source_bytes(serializer.validated_data), serializer.render_options())

        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        return Response(result)


class HealthView(APIView):
    throttle_classes = ()

    def get(self, request):
        return Response({"status": "ok", "version": __version__})


def not_found(request, exception=None):
    """JSON 404 for unknown /api/ paths (and site-wide when DEBUG is off)."""
    return JsonResponse(error_payload("not_found", "Not found."), status=404)


def server_error(request):
    return JsonResponse(error_payload("server_error", SERVER_ERROR_MESSAGE), status=500)


def _reject_oversized_body(request):
    """Refuse oversized bodies from their declared length, before reading them."""
    try:
        declared = int(request.META.get("CONTENT_LENGTH") or 0)
    except ValueError:
        return
    if declared > settings.MAX_UPLOAD_BYTES + _BODY_OVERHEAD_BYTES:
        limit_mb = settings.MAX_UPLOAD_BYTES / (1024 * 1024)
        raise TooLarge(f"The upload is larger than the {limit_mb:g} MB limit.")


def _source_bytes(data) -> bytes:
    upload = data.get("file")
    if upload is not None:
        return upload.read()
    return fetch_image_bytes(
        data["url"],
        max_bytes=settings.MAX_UPLOAD_BYTES,
        allow_private=settings.ALLOW_PRIVATE_URLS,
        cache=caches[settings.URL_CACHE_ALIAS],
        cache_seconds=settings.URL_CACHE_SECONDS,
    )
