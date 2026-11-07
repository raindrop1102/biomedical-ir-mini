"""Rank the ModApte test stories for each query and score the rankings.

A method is any function from query text to a ranked list of newids, so TF-IDF
(method A) and the word-embedding methods (method B, step 6) are evaluated by the
same code on the same queries and judgments.

Method A is homework 3's vector space, unchanged: the test stories become Documents
(title + body) and VectorSpace ranks them by SMART-weighted cosine.
"""
import statistics
from collections.abc import Callable

from ..vsm import VectorSpace
from ..vsm.evaluate import score
from .queries import judgments

Ranker = Callable[[str], list[str]]
METRICS = ["P@10", "R@100", "MRR", "MAP", "nDCG@10"]


def test_space(articles) -> VectorSpace:
    """The retrieval collection: every ModApte test story, as a Document."""
    return VectorSpace([a.document for a in articles if a.split == "test"])


def tfidf_ranker(space: VectorSpace, scheme: str = "ltc.ltc") -> Ranker:
    """Method A: TF-IDF vectors for documents and query, ranked by cosine."""
    return lambda text: [hit.pmid for hit in space.search(text, scheme).hits]


def evaluate(ranker: Ranker, articles, queries) -> list[dict]:
    """One row per query: its metrics, how many stories it scored above 0, and how many were relevant."""
    rows = []
    for query in queries:
        grades = judgments(articles, query)
        ranking = ranker(query.text)
        metrics = score(ranking, grades)
        found = sum(1 for newid in ranking if newid in grades)
        rows.append({"query": query.id, "topic": query.topic, "kind": query.kind, "relevant": len(grades),
                     "retrieved": len(ranking), "relevant_retrieved": found,
                     **{m: round(metrics.values[m], 4) for m in METRICS}})
    return rows


def mean(rows: list[dict], kind: str | None = None) -> dict:
    """Mean of every metric over the rows, optionally only one kind of query."""
    rows = [r for r in rows if kind is None or r["kind"] == kind]
    return {m: round(statistics.mean(r[m] for r in rows), 4) for m in METRICS}
