"""WSGI entry point: `gunicorn app:app`, which is also Render's default start command for a
web service created by hand. Builds every index once at import (about 3 s)."""
from bioir.web import create_app

app = create_app()
