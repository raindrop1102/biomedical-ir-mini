"""Trained word vectors: lookup, cosine neighbours, and a single-file .npz format."""
import json
from pathlib import Path

import numpy as np

from ..text import is_number, normalize, tokens


class Embeddings:
    def __init__(self, words, counts, vectors, metadata: dict):
        self.words = list(words)
        self.counts = np.asarray(counts)
        self.vectors = np.asarray(vectors)
        self.metadata = metadata
        self.index = {w: i for i, w in enumerate(self.words)}
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.unit = self.vectors / np.maximum(norms, 1e-12)

    def __len__(self):
        return len(self.words)

    def term(self, word: str) -> str | None:
        """The vocabulary form of a query word, preprocessed as in training; None if it is not one token."""
        compound = self.metadata.get("config", {}).get("compound", True)
        parts = tokens(word, compound)
        return normalize(parts[0]) if len(parts) == 1 and not is_number(parts[0]) else None

    def _id(self, word: str) -> int:
        term = self.term(word)
        if term is None:
            raise ValueError(f"「{word}」不是單一詞")
        if term not in self.index:
            raise KeyError(f"「{term}」不在詞彙表中（可能低於 min_count）")
        return self.index[term]

    def similarity(self, a: str, b: str) -> float:
        return float(self.unit[self._id(a)] @ self.unit[self._id(b)])

    def similar(self, word: str, k: int = 10, min_count: int = 1) -> list[tuple[str, float, int]]:
        """(term, cosine, corpus count) of the k nearest words, skipping words seen fewer than min_count times."""
        i = self._id(word)
        scores = self.unit @ self.unit[i]
        ranked = (j for j in np.argsort(-scores, kind="stable") if j != i and self.counts[j] >= min_count)
        return [(self.words[j], float(scores[j]), int(self.counts[j])) for j in list(ranked)[:k]]

    def save(self, path) -> Path:
        path = Path(path)
        path = path if path.suffix == ".npz" else path.with_name(path.name + ".npz")   # numpy adds it anyway
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, words=np.array(self.words), counts=self.counts, vectors=self.vectors,
                            metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)))
        return path

    @classmethod
    def load(cls, path) -> "Embeddings":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["words"].tolist(), data["counts"], data["vectors"], json.loads(str(data["metadata"])))
