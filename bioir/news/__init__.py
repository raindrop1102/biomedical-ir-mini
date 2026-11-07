"""News retrieval on Reuters-21578: TF-IDF versus word embeddings (homework 4)."""
from .queries import NewsQuery, coverage, judgments, load_queries
from .reuters import Article, load_reuters, parse

__all__ = ["Article", "NewsQuery", "coverage", "judgments", "load_queries", "load_reuters", "parse"]
