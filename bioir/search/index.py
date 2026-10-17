"""Positional inverted index with Boolean matching and BM25 ranking."""
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..corpus import Document
from ..text import analyze, analyze_spans
from .query import parse, positive_phrases, positive_terms


@dataclass
class Results:
    total: int                               # documents matching the Boolean query
    hits: list[tuple[float, Document]]       # top-k by BM25, best first
    terms: set[str]                          # analyzed terms used for ranking
    # pmid → {"title": [(start, end), ...], "abstract": [...]}: where each hit's query phrases occur
    spans: dict[str, dict[str, list[tuple[int, int]]]] = field(default_factory=dict)


def phrase_spans(text: str, phrases) -> list[tuple[int, int]]:
    """Character span of every occurrence of every phrase, first to last word."""
    terms, places = {}, {}
    for term, position, start, end in analyze_spans(text):
        terms[position], places[position] = term, (start, end)
    spans = set()
    for pairs in phrases:
        (first, _), last = pairs[0], pairs[-1][1]
        for position, term in terms.items():
            if term == first and all(terms.get(position + offset) == t for t, offset in pairs[1:]):
                spans.add((places[position][0], places[position + last][1]))
    return sorted(spans)


class Index:
    def __init__(self, documents, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.documents = {d.pmid: d for d in documents}
        self.postings = defaultdict(dict)    # term → {pmid: [positions]}
        self.lengths = {}
        for pmid, doc in self.documents.items():
            pairs = analyze(doc.title)
            # Start the abstract two positions after the title so phrases cannot span the fields.
            pairs += analyze(doc.abstract, start=pairs[-1][1] + 2 if pairs else 0)
            self.lengths[pmid] = len(pairs)
            for term, position in pairs:
                self.postings[term].setdefault(pmid, []).append(position)
        self.average_length = sum(self.lengths.values()) / max(len(self.lengths), 1)

    def __len__(self):
        return len(self.documents)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        # BM25 idf with +1 inside the log (as in Lucene), so it is never negative.
        df = self.df(term)
        return math.log(1 + (len(self) - df + 0.5) / (df + 0.5))

    # ---------- Boolean matching ----------

    def phrase(self, pairs) -> set[str]:
        """Documents containing every term at its relative offset."""
        lists = [self.postings.get(term) for term, _ in pairs]
        if not all(lists):
            return set()
        docs = set(min(lists, key=len))
        for postings in lists:
            docs &= postings.keys()
        if len(pairs) == 1:
            return docs
        (first, base), rest = pairs[0], pairs[1:]
        return {pmid for pmid in docs
                if self._adjacent(self.postings[first][pmid],
                                  [(set(self.postings[t][pmid]), offset - base) for t, offset in rest])}

    @staticmethod
    def _adjacent(starts, others) -> bool:
        return any(all(p + offset in positions for positions, offset in others) for p in starts)

    def match(self, node) -> set[str]:
        kind, value = node
        if kind == "phrase":
            return self.phrase(value)
        if kind == "or":
            return set().union(*map(self.match, value))
        if kind == "not":
            return self.documents.keys() - self.match(value)
        # AND: intersect the positive parts smallest-first, then subtract the NOT parts.
        include = sorted((self.match(n) for n in value if n[0] != "not"), key=len)
        result = include[0].intersection(*include[1:]) if include else set(self.documents)
        for n in value:
            if n[0] == "not" and result:
                result -= self.match(n[1])
        return result

    # ---------- Ranking ----------

    def search(self, query: str, k: int = 10) -> Results:
        tree = parse(query)
        if tree is None:
            return Results(0, [], set())
        matched = self.match(tree)
        terms = positive_terms(tree)
        weights = {t: self.idf(t) for t in terms if t in self.postings}

        def bm25(pmid):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[pmid] / self.average_length)
            score = 0.0
            for term, idf in weights.items():
                tf = len(self.postings[term].get(pmid, ()))
                score += idf * tf * (self.k1 + 1) / (tf + norm)
            return score

        # nlargest is stable, so iterating PMIDs in order breaks score ties deterministically.
        top = heapq.nlargest(k, ((bm25(p), p) for p in sorted(matched)), key=lambda hit: hit[0])
        hits = [(s, self.documents[p]) for s, p in top]
        phrases = positive_phrases(tree)
        spans = {doc.pmid: {"title": phrase_spans(doc.title, phrases),
                            "abstract": phrase_spans(doc.abstract, phrases)} for _, doc in hits}
        return Results(len(matched), hits, terms, spans)
