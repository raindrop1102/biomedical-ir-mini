"""Vector-space retrieval: documents and queries as weighted term vectors, ranked by
their inner product — the cosine when both sides are L2-normalized.

Terms are the search index's (bioir.text.analyze: stop words removed, Porter stems)
from the title and abstract, as a bag of words: no phrases or Boolean operators.
Scoring is term-at-a-time over an inverted index, so only documents sharing a query
term are touched; document vector lengths are computed once per weighting.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..text import analyze
from .weighting import Scheme, Weighting


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


@dataclass
class TermWeight:
    term: str
    tf: int
    tf_weight: float                    # the TF letter applied to tf
    idf: float                          # the IDF letter (1 for n)
    weight: float                       # tf_weight × idf ÷ the side's divisor


@dataclass
class Side:
    """How one side — the query or one document — turns raw counts into its vector's
    entries for the query terms, letter by letter."""
    weighting: Weighting
    terms: list[TermWeight]
    divisor: float                      # what the normalization letter divides by (1 for n)
    max_tf: int                         # inputs of the a and L TF weights
    average_tf: float
    distinct: int                       # inputs of the u and b normalizations
    characters: int


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
        self.average_tf = {p: self.lengths[p] / len(c) if c else 1.0 for p, c in self.tf.items()}
        self.characters = {p: len(d.title) + 1 + len(d.abstract) for p, d in self.documents.items()}
        self.pivot = sum(len(c) for c in self.tf.values()) / max(len(self.tf), 1)   # mean distinct terms
        self._norms: dict[Weighting, dict[str, float]] = {}

    def __len__(self):
        return len(self.documents)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def weight(self, weighting: Weighting, term: str, pmid: str) -> float:
        tf = self.tf[pmid].get(term, 0)
        return (weighting.tf_weight(tf, self.max_tf[pmid], self.average_tf[pmid])
                * weighting.idf_weight(self.df(term), len(self)))

    def norms(self, weighting: Weighting) -> dict[str, float]:
        """Normalization divisor of every document under a weighting (cached): for "c" the
        L2 length of its vector, which needs every term, not just the query's."""
        if weighting not in self._norms:
            n = len(self)
            idf = {t: weighting.idf_weight(len(p), n) for t, p in self.postings.items()} if weighting.norm == "c" else {}
            self._norms[weighting] = {
                pmid: weighting.divisor(
                    (weighting.tf_weight(tf, self.max_tf[pmid], self.average_tf[pmid]) * idf[t]
                     for t, tf in counts.items()) if idf else (),
                    len(counts), self.characters[pmid], self.pivot)
                for pmid, counts in self.tf.items()}
        return self._norms[weighting]

    def query_vector(self, query: str, weighting: Weighting) -> tuple[dict[str, float], list[str]]:
        counts = Counter(t for t, _ in analyze(query))
        unknown = sorted(t for t in counts if t not in self.postings)
        counts = {t: c for t, c in counts.items() if t in self.postings}
        top = max(counts.values(), default=1)
        average_tf = sum(counts.values()) / len(counts) if counts else 1.0
        vector = {t: weighting.tf_weight(c, top, average_tf) * weighting.idf_weight(self.df(t), len(self))
                  for t, c in counts.items()}
        divisor = weighting.divisor(vector.values(), len(counts), len(query), self.pivot)
        if divisor:
            vector = {t: w / divisor for t, w in vector.items()}
        return {t: w for t, w in vector.items() if w}, unknown

    def search(self, query: str, scheme="lnc.ltc", k: int | None = None) -> Ranking:
        """Rank documents for a free-text query; k=None keeps every document that scores above 0."""
        scheme = Scheme.parse(scheme) if isinstance(scheme, str) else scheme
        vector, unknown = self.query_vector(query, scheme.query)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        for term, query_weight in vector.items():
            for pmid in self.postings[term]:
                document_weight = self.weight(scheme.document, term, pmid)
                if document_weight:
                    contributions[pmid][term] = query_weight * document_weight
        if scheme.document.norm != "n":
            norms = self.norms(scheme.document)
            contributions = {p: {t: v / norms[p] for t, v in parts.items()} for p, parts in contributions.items()}
        scored = sorted(((sum(parts.values()), pmid) for pmid, parts in contributions.items()),
                        key=lambda item: (-item[0], item[1]))
        hits = [Hit(rank, pmid, score, contributions[pmid])
                for rank, (score, pmid) in enumerate(scored[:k] if k else scored, 1) if score > 0]
        return Ranking(scheme, vector, unknown, hits)

    def breakdown(self, query: str, scheme, pmid: str) -> tuple[Side, Side]:
        """The query's and one document's weights for every query term the corpus knows,
        step by step; the score is the sum over terms of the two sides' weights multiplied."""
        scheme = Scheme.parse(scheme) if isinstance(scheme, str) else scheme
        n = len(self)
        counts = Counter(t for t, _ in analyze(query))
        counts = {t: c for t, c in counts.items() if t in self.postings}
        sides = []
        for is_document, (weighting, tf, max_tf, average_tf, distinct, characters) in enumerate((
                (scheme.query, counts, max(counts.values(), default=1),
                 sum(counts.values()) / len(counts) if counts else 1.0, len(counts), len(query)),
                (scheme.document, self.tf[pmid], self.max_tf[pmid], self.average_tf[pmid],
                 len(self.tf[pmid]), self.characters[pmid]))):
            raw = [(t, tf.get(t, 0), weighting.tf_weight(tf.get(t, 0), max_tf, average_tf),
                    weighting.idf_weight(self.df(t), n)) for t in counts]
            if is_document:                 # the cached divisor: c needs every term of the document
                divisor = self.norms(weighting)[pmid] if weighting.norm != "n" else 1.0
            else:
                divisor = weighting.divisor((w * i for _, _, w, i in raw), distinct, characters, self.pivot)
            divisor = divisor or 1.0
            sides.append(Side(weighting, [TermWeight(t, c, w, i, w * i / divisor) for t, c, w, i in raw],
                              divisor, max_tf, average_tf, distinct, characters))
        return sides[0], sides[1]
