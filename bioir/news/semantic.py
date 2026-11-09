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
        self.counts = dict(zip(self.ids, counts))
        self.n = len(stories)
        self.df = Counter(w for c in counts for w in c)
        self.matrix = np.vstack([self.vector(c) for c in counts])
        self.empty = int((np.linalg.norm(self.matrix, axis=1) == 0).sum())   # stories with no known word

    def idf(self, word: str) -> float:
        return math.log10(self.n / max(self.df.get(word, 0), 1))           # unseen query words count as rare

    def weights(self, counts: Counter) -> dict[str, float]:
        """w(t) for each counted word: tf for "mean", (1 + log10 tf) · idf otherwise."""
        if self.weighting == "mean":
            return {w: float(c) for w, c in counts.items()}
        return {w: (1 + math.log10(c)) * self.idf(w) for w, c in counts.items()}

    def vector(self, counts: Counter) -> np.ndarray:
        """Unit-length weighted average of the counted words' vectors (zeros if none has weight)."""
        if not counts:
            return np.zeros(self.model.vectors.shape[1])
        weights = self.weights(counts)
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

    def explain(self, text: str, newid: str) -> dict | None:
        """One story's score split exactly into its words' contributions.

        The story vector is d = Σ w_t·v_t / L with v_t unit vectors and L = ‖Σ w_t·v_t‖ (dividing
        by Σ w first does not change the direction), so
            cosine(q, d) = Σ_t (w_t / L) · cos(v_t, q)
        — each word's share of the story times how close the word is to the query vector.
        Words are sorted by contribution; None when no query word has a vector."""
        query_counts = Counter(words(text, self.model))
        q = self.vector(query_counts)
        if not q.any():
            return None
        query_weights = self.weights(query_counts)
        query_total = sum(query_weights.values())
        query = [{"word": w, "tf": c, "tf_weight": 1 + math.log10(c), "df": self.df.get(w, 0), "idf": self.idf(w),
                  "weight": query_weights[w],
                  "share": query_weights[w] / query_total if query_total else 0.0}
                 for w, c in query_counts.items()]
        counts = self.counts[newid]
        weights = self.weights(counts)
        rows = []
        if counts:
            unit = self.model.unit[[self.model.index[w] for w in weights]]
            length = float(np.linalg.norm((np.array(list(weights.values()))[:, None] * unit).sum(0)))
            for (w, weight), cosine in zip(weights.items(), (unit @ q).tolist()):
                share = weight / length if length else 0.0
                rows.append({"word": w, "tf": counts[w], "df": self.df[w], "idf": self.idf(w), "weight": weight,
                             "share": share, "cosine": cosine, "contribution": share * cosine,
                             "in_query": w in query_counts})
        else:
            length = 0.0
        rows.sort(key=lambda r: (-r["contribution"], r["word"]))
        return {"query": query, "rows": rows, "length": length,
                "score": sum(r["contribution"] for r in rows), "weighting": self.weighting}

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
