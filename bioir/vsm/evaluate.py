"""Relevance judgments from MeSH, ranking metrics and a paired significance test.

Judgments: a query names MeSH descriptors; a document is relevant when the NLM
indexers assigned all of mesh_all and at least one of mesh_any (when given).
Grade 2 when one of those descriptors is a major topic of the article, else 1.
MeSH describes what an article is about, so this is topical relevance assigned
independently of any ranking — but indexers can miss a topic, and an article can
mention a topic it was not indexed under.
"""
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    mesh_all: tuple[str, ...] = ()
    mesh_any: tuple[str, ...] = ()

    def rule(self) -> str:
        parts = [" AND ".join(self.mesh_all)] if self.mesh_all else []
        if self.mesh_any:
            parts.append("(" + " OR ".join(self.mesh_any) + ")")
        return " AND ".join(parts)


def load_queries(path) -> list[Query]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = [Query(q["id"], q["query"], tuple(q.get("mesh_all", ())), tuple(q.get("mesh_any", ())))
               for q in data["queries"]]
    if any(not (q.mesh_all or q.mesh_any) for q in queries):
        raise ValueError("Every query needs mesh_all or mesh_any")
    return queries


def judgments(documents, query: Query) -> dict[str, int]:
    """pmid → grade (1 or 2) for the relevant documents."""
    grades = {}
    for doc in documents:
        mesh = set(doc.mesh)
        if all(m in mesh for m in query.mesh_all) and (not query.mesh_any or mesh & set(query.mesh_any)):
            named = set(query.mesh_all) | (mesh & set(query.mesh_any))
            grades[doc.pmid] = 2 if named & set(doc.mesh_major) else 1
    return grades


@dataclass
class Metrics:
    relevant: int
    retrieved: int
    values: dict[str, float] = field(default_factory=dict)


METRICS = ["P@5", "P@10", "R@10", "R@100", "MRR", "MAP", "nDCG@10"]


def score(ranking: list[str], grades: dict[str, int]) -> Metrics:
    """Metrics for one ranked list of PMIDs; binary relevance except nDCG, which uses the grades."""
    total = len(grades)
    hits = [1 if pmid in grades else 0 for pmid in ranking]

    def precision(k):
        return sum(hits[:k]) / k

    def recall(k):
        return sum(hits[:k]) / total if total else 0.0

    first = next((i for i, h in enumerate(hits, 1) if h), None)
    average_precision = (sum(sum(hits[:i]) / i for i, h in enumerate(hits, 1) if h) / total) if total else 0.0
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 1) for i, p in enumerate(ranking[:10], 1))
    ideal = sum((2 ** g - 1) / math.log2(i + 1)
                for i, g in enumerate(sorted(grades.values(), reverse=True)[:10], 1))
    values = {"P@5": precision(5), "P@10": precision(10), "R@10": recall(10), "R@100": recall(100),
              "MRR": 1 / first if first else 0.0, "MAP": average_precision,
              "nDCG@10": dcg / ideal if ideal else 0.0}
    return Metrics(total, len(ranking), values)


def paired_permutation(a: list[float], b: list[float]) -> float:
    """Two-sided p-value that two systems' per-query scores have the same mean, by flipping
    the sign of every subset of the per-query differences (exact up to 16 queries)."""
    differences = [x - y for x, y in zip(a, b)]
    n = len(differences)
    observed = abs(sum(differences))
    if n == 0 or observed == 0:
        return 1.0
    if n > 16:
        raise ValueError("The exact test enumerates 2^n sign patterns; use at most 16 queries")
    extreme = sum(1 for signs in itertools.product((1, -1), repeat=n)
                  if abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-12)
    return extreme / 2 ** n
