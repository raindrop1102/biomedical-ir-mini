"""Method B: documents and queries as TF-IDF-weighted averages of word vectors.

    doc vector  d = Σ_t w(t, d) · v_t / Σ_t w(t, d),   w(t, d) = (1 + log10 tf) · log10(N / df)

over the words of the story that the embedding model knows; the query vector is
built the same way, and stories are ranked by cosine(d, q). Words are the embedding
model's own (lower case, compounds kept, numbers dropped, no stemming); stop words
are dropped — their IDF is near 0 anyway. N and df come from the test collection.

Unlike TF-IDF, every story gets a score, including stories that share no word with
the query: that is the point — and the risk — of the method.

weighting="mean" averages the vectors without TF-IDF (the ablation that shows what
the weights contribute). hybrid() mixes method A's and B's cosines with a fixed α.
Needs numpy.
"""
import math
from collections import Counter

import numpy as np

from ..text import STOP_WORDS, is_number, normalize, tokens


def words(text: str, model) -> list[str]:
    """The text's words that the model has vectors for."""
    compound = model.metadata.get("config", {}).get("compound", True)
    found = (normalize(w) for w in tokens(text, compound) if not is_number(w))
    return [w for w in found if w not in STOP_WORDS and w in model.index]


class EmbeddingSpace:
    def __init__(self, articles, model, weighting: str = "tfidf"):
        if weighting not in ("tfidf", "mean"):
            raise ValueError("weighting must be 'tfidf' or 'mean'")
        self.model, self.weighting = model, weighting
        stories = [a for a in articles if a.split == "test"]
        self.ids = [a.newid for a in stories]
        counts = [Counter(words(f"{a.title} . {a.body}", model)) for a in stories]
        self.n = len(stories)
        self.df = Counter(w for c in counts for w in c)
        self.matrix = np.vstack([self.vector(c) for c in counts])
        self.empty = int((np.linalg.norm(self.matrix, axis=1) == 0).sum())   # stories with no known word

    def idf(self, word: str) -> float:
        return math.log10(self.n / max(self.df.get(word, 0), 1))           # unseen query words count as rare

    def vector(self, counts: Counter) -> np.ndarray:
        """Unit-length weighted average of the counted words' vectors (zeros if none has weight)."""
        if not counts:
            return np.zeros(self.model.vectors.shape[1])
        if self.weighting == "mean":
            weights = {w: float(c) for w, c in counts.items()}
        else:
            weights = {w: (1 + math.log10(c)) * self.idf(w) for w, c in counts.items()}
        total = sum(weights.values())
        if total <= 0:
            return np.zeros(self.model.vectors.shape[1])
        rows = [self.model.index[w] for w in weights]
        average = (np.array(list(weights.values()))[:, None] * self.model.unit[rows]).sum(0) / total
        norm = np.linalg.norm(average)
        return average / norm if norm else average

    def scores(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Cosine of every test story with the query, and the query words that had vectors."""
        query_words = words(text, self.model)
        q = self.vector(Counter(query_words))
        return self.matrix @ q, query_words

    def ranker(self):
        def rank(text: str) -> list[str]:
            scores, query_words = self.scores(text)
            if not query_words:
                return []
            order = sorted(range(self.n), key=lambda i: (-scores[i], self.ids[i]))
            return [self.ids[i] for i in order]
        return rank


def hybrid(space_a, scheme: str, space_b: EmbeddingSpace, alpha: float = 0.5):
    """α · TF-IDF cosine + (1 − α) · embedding cosine, over every test story."""
    def rank(text: str) -> list[str]:
        b, _ = space_b.scores(text)
        combined = dict(zip(space_b.ids, (1 - alpha) * b))
        for hit in space_a.search(text, scheme).hits:
            combined[hit.pmid] = combined.get(hit.pmid, 0.0) + alpha * hit.score
        return [newid for newid, _ in sorted(combined.items(), key=lambda item: (-item[1], item[0]))]
    return rank
