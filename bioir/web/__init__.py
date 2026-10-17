"""Read-only web interface: search, Zipf and embeddings pages. Needs flask (and numpy for embeddings)."""
from .app import create_app

__all__ = ["create_app"]
