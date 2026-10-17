"""Biomedical abstract retrieval and analysis.

Shared foundation: text (tokenizer + normalization), porter, corpus, pubmed.
Subpackages depend only on that foundation, never on each other:
    search  positional index, Boolean queries, BM25, spelling
    zipf    rank-frequency statistics under preprocessing conditions A-D
"""
from .corpus import Document, load
from .search import Index, Results, Speller
from .text import highlight

__all__ = ["Document", "Index", "Results", "Speller", "highlight", "load"]
