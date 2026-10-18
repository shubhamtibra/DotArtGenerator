"""Request validation for the generate endpoint."""

import math
import unicodedata

from django.conf import settings
from rest_framework import serializers
from rest_framework.fields import empty

from .engine import RenderOptions
from .engine.options import (
    BRIGHTNESS_RANGE,
    CHAR_ASPECT_RANGE,
    CHARSETS,
    CONTRAST_RANGE,
    CUSTOM_CHARSET_LENGTH,
    DITHERS,
    EDGES_RANGE,
    GAMMA_RANGE,
    MAX_COLUMNS,
    MIN_COLUMNS,
    MODES,
    SHARPEN_RANGE,
    THRESHOLD_RANGE,
)
from .exceptions import MissingSource, TooLarge

MAX_URL_LENGTH = 2048


class OptionalBooleanField(serializers.BooleanField):
    """BooleanField whose default also applies to multipart/form requests.

    DRF treats an omitted boolean in HTML-style input as ``False`` (checkbox
    semantics), which would silently turn off ``normalize`` for every multipart
    request that doesn't send it.
    """

    default_empty_html = empty


class FiniteFloatField(serializers.FloatField):
    """FloatField that rejects NaN (which slips past min/max validators)."""

    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        if not math.isfinite(value):
            self.fail("invalid")
        return value


class ThresholdField(serializers.IntegerField):
    """An integer in 0..255, or ``"auto"`` / blank / null for Otsu (``None``)."""

    default_error_messages = {
        "invalid": 'Must be an integer from 0 to 255, or "auto".',
    }

    def __init__(self, **kwargs):
        low, high = THRESHOLD_RANGE
        super().__init__(
            min_value=low, max_value=high, required=False, allow_null=True, default=None, **kwargs
        )

    def to_internal_value(self, data):
        if isinstance(data, str) and data.strip().lower() in ("", "auto"):
            return None
        return super().to_internal_value(data)

    def run_validators(self, value):
        if value is not None:
            super().run_validators(value)


def _float(default, bounds):
    low, high = bounds
    return FiniteFloatField(default=default, min_value=low, max_value=high)


def _integer(default, bounds):
    low, high = bounds
    return serializers.IntegerField(default=default, min_value=low, max_value=high)


class GenerateRequestSerializer(serializers.Serializer):
    file = serializers.FileField(required=False, allow_empty_file=True)
    url = serializers.CharField(required=False, allow_blank=True, max_length=MAX_URL_LENGTH)

    mode = serializers.ChoiceField(choices=MODES, default="braille")
    width = _integer(100, (MIN_COLUMNS, MAX_COLUMNS))
    charset = serializers.ChoiceField(choices=CHARSETS, default="standard")
    custom_charset = serializers.CharField(
        required=False, default="", allow_blank=True, trim_whitespace=False
    )
    dither = serializers.ChoiceField(choices=DITHERS, default="floyd_steinberg")
    threshold = ThresholdField()
    invert = OptionalBooleanField(default=False)
    brightness = _integer(0, BRIGHTNESS_RANGE)
    contrast = _integer(0, CONTRAST_RANGE)
    gamma = _float(1.0, GAMMA_RANGE)
    sharpen = _float(0.0, SHARPEN_RANGE)
    edges = _float(0.0, EDGES_RANGE)
    equalize = OptionalBooleanField(default=False)
    normalize = OptionalBooleanField(default=True)
    char_aspect = _float(0.5, CHAR_ASPECT_RANGE)
    color = OptionalBooleanField(default=False)
    include_processed = OptionalBooleanField(default=False)

    def validate_file(self, upload):
        limit = settings.MAX_UPLOAD_BYTES
        if upload.size > limit:
            raise TooLarge(f"The image is larger than the {limit / (1024 * 1024):g} MB upload limit.")
        return upload

    def validate(self, attrs):
        has_file = attrs.get("file") is not None
        has_url = bool(attrs.get("url"))
        if has_file and has_url:
            raise serializers.ValidationError("Provide either an image file or a URL, not both.")
        if not has_file and not has_url:
            # Reported with its own code rather than as a field error.
            raise MissingSource()

        if attrs["mode"] == "ascii" and attrs["charset"] == "custom":
            problem = _custom_charset_problem(attrs["custom_charset"])
            if problem:
                raise serializers.ValidationError({"custom_charset": [problem]})
        return attrs

    def render_options(self) -> RenderOptions:
        options = {k: v for k, v in self.validated_data.items() if k not in ("file", "url")}
        return RenderOptions(**options)


def _custom_charset_problem(chars: str) -> str | None:
    low, high = CUSTOM_CHARSET_LENGTH
    if not chars:
        return 'Provide the characters to use when charset is "custom".'
    if not low <= len(chars) <= high:
        return f"Must contain between {low} and {high} characters."
    # Cc covers newlines, tabs and other controls; Cs are lone surrogates.
    if any(unicodedata.category(ch) in ("Cc", "Cs") for ch in chars):
        return "Must not contain newlines, tabs or other control characters."
    return None
