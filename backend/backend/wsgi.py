"""WSGI entry point.

Exposes ``application`` for WSGI servers and ``app`` for Vercel's Python runtime.
"""

import os
import sys
from pathlib import Path

from django.core.wsgi import get_wsgi_application

# Serverless runtimes import this file directly, possibly from the repository
# root; make sure the Django project directory is importable.
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

application = get_wsgi_application()
app = application
