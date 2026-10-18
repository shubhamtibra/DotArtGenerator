"""Root URL configuration: the whole service lives under /api/."""

from django.urls import include, path

urlpatterns = [
    path("api/", include("app.urls")),
]

# JSON instead of HTML error pages (Django only uses these when DEBUG is off).
handler404 = "app.views.not_found"
handler500 = "app.views.server_error"
