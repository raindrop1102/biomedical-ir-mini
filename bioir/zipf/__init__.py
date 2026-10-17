"""Zipf analysis under preprocessing conditions A-D (statistics: standard library only)."""
from .stats import (CONDITIONS, ConditionStats, Fit, Region, Term, analyze_corpus, compare_terms,
                    preprocess, write_csv)

__all__ = ["CONDITIONS", "ConditionStats", "Fit", "Region", "Term", "analyze_corpus", "compare_terms",
           "preprocess", "write_csv"]
