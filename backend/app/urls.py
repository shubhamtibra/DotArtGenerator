from django.urls import re_path

from . import views

# Trailing slashes are optional so POSTs never hit APPEND_SLASH redirects.
urlpatterns = [
    re_path(r"^generate/?$", views.GenerateView.as_view(), name="generate"),
    re_path(r"^upload/?$", views.GenerateView.as_view(), name="upload"),  # v1 alias
    re_path(r"^health/?$", views.HealthView.as_view(), name="health"),
    re_path(r"^.*$", views.not_found),
]
