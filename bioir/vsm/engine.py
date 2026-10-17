"""Vector-space retrieval: documents and queries as weighted term vectors, ranked by
their inner product — the cosine when both sides are L2-normalized.

Terms are the search index's (bioir.text.analyze: stop words removed, Porter stems)
from the title and abstract, as a bag of words: no phrases or Boolean operators.
Scoring is term-at-a-time over an inverted index, so only documents sharing a query
term are touched; document vector lengths are computed once per weighting.
"""
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..text import analyze
from .weighting import Scheme, Weighting

BM25_K1, BM25_B = 1.2, 0.75


@dataclass
class Hit:
    rank: int
    pmid: str
    score: float
    contributions: dict[str, float] = field(default_factory=dict)   # term → its share of the score


@dataclass
class Ranking:
    scheme: Scheme
    query_weights: dict[str, float]     # the query vector (terms absent from the corpus dropped)
    unknown: list[str]                  # query terms no document contains
    hits: list[Hit]                     # every document with a positive score, best first


class VectorSpace:
    def __init__(self, documents):
        self.documents = {d.pmid: d for d in documents}
        self.tf: dict[str, Counter] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)       # term → {pmid: tf}
        for pmid, doc in self.documents.items():
            counts = Counter(t for t, _ in analyze(doc.title)) + Counter(t for t, _ in analyze(doc.abstract))
            self.tf[pmid] = counts
            for term, count in counts.items():
                self.postings[term][pmid] = count
        self.max_tf = {p: max(c.values(), default=0) for p, c in self.tf.items()}
        self.lengths = {p: sum(c.values()) for p, c in self.tf.items()}
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self._norms: dict[str, dict[str, float]] = {}

    def __len__(self):
        return len(self.documents)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def weight(self, weighting: Weighting, term: str, pmid: str) -> float:
        tf = self.tf[pmid].get(term, 0)
        return weighting.tf_weight(tf, self.max_tf[pmid]) * weighting.idf_weight(self.df(term), len(self))

    def norms(self, weighting: Weighting) -> dict[str, float]:
        """L2 length of every document vector under a weighting (cached)."""
        if weighting.code not in self._norms:
            n = len(self)
            idf = {t: weighting.idf_weight(len(p), n) for t, p in self.postings.items()}
            self._norms[weighting.code] = {
                pmid: math.sqrt(sum((weighting.tf_weight(tf, self.max_tf[pmid]) * idf[t]) ** 2
                                    for t, tf in counts.items()))
                for pmid, counts in self.tf.items()}
        return self._norms[weighting.code]

    def query_vector(self, query: str, weighting: Weighting) -> tuple[dict[str, float], list[str]]:
        counts = Counter(t for t, _ in analyze(query))
        unknown = sorted(t for t in counts if t not in self.postings)
        counts = {t: c for t, c in counts.items() if t in self.postings}
        top = max(counts.values(), default=1)
        vector = {t: weighting.tf_weight(c, top) * weighting.idf_weight(self.df(t), len(self)) for t, c in counts.items()}
        if weighting.norm == "c":
            length = math.sqrt(sum(w * w for w in vector.values()))
            vector = {t: w / length for t, w in vector.items()} if length else vector
        return {t: w for t, w in vector.items() if w}, unknown

    def _bm25(self, term: str, pmid: str) -> float:
        df, tf = self.df(term), self.tf[pmid][term]
        idf = math.log(1 + (len(self) - df + 0.5) / (df + 0.5))
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[pmid] / self.average_length)
        return idf * tf * (BM25_K1 + 1) / (tf + norm)

    def search(self, query: str, scheme="lnc.ltc", k: int | None = None) -> Ranking:
        """Rank documents for a free-text query; k=None keeps every document that scores above 0."""
        scheme = Scheme.parse(scheme) if isinstance(scheme, str) else scheme
        vector, unknown = self.query_vector(query, scheme.query)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        for term, query_weight in vector.items():
            for pmid in self.postings[term]:
                document_weight = (self._bm25(term, pmid) if scheme.document is None
                                   else self.weight(scheme.document, term, pmid))
                if document_weight:
                    contributions[pmid][term] = query_weight * document_weight
        if scheme.document is not None and scheme.document.norm == "c":
            norms = self.norms(scheme.document)
            contributions = {p: {t: v / norms[p] for t, v in parts.items()} for p, parts in contributions.items()}
        scored = sorted(((sum(parts.values()), pmid) for pmid, parts in contributions.items()),
                        key=lambda item: (-item[0], item[1]))
        hits = [Hit(rank, pmid, score, contributions[pmid])
                for rank, (score, pmid) in enumerate(scored[:k] if k else scored, 1) if score > 0]
        return Ranking(scheme, vector, unknown, hits)
