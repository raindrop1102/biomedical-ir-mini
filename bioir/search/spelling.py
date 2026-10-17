"""Edit-distance "did you mean" for query words the index has never seen."""
import re
from collections import Counter, defaultdict

from ..porter import stem
from ..text import STOP_WORDS, normalize, words
from .query import OPERATORS


def edit_distance(a: str, b: str, limit: int) -> int:
    """Optimal-string-alignment distance: Levenshtein plus adjacent transposition ("weigth" → 1).

    Row-by-row DP keeping two previous rows; returns limit + 1 as soon as the
    distance must exceed limit (row minima never decrease).
    """
    before, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        row = [i]
        for j in range(1, len(b) + 1):
            cost = min(row[j - 1] + 1, previous[j] + 1, previous[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cost = min(cost, before[j - 2] + 1)
            row.append(cost)
        if min(row) > limit:
            return limit + 1
        before, previous = previous, row
    return previous[-1]


class Speller:
    def __init__(self, index):
        self.index = index
        self.counts = Counter(w for d in index.documents.values() for w in words(f"{d.title} {d.abstract}"))
        self.by_length = defaultdict(list)
        for word in self.counts:
            self.by_length[len(word)].append(word)

    def correct_word(self, word: str) -> str | None:
        limit = 1 if len(word) < 5 else 2
        best = None
        for length in range(len(word) - limit, len(word) + limit + 1):
            for candidate in self.by_length.get(length, ()):
                distance = edit_distance(word, candidate, limit)
                if distance <= limit:
                    key = (distance, -self.counts[candidate], candidate)
                    best = min(best, key) if best else key
        return best[2] if best else None

    def suggest(self, query: str) -> str | None:
        """The query with each unknown word replaced by its closest frequent word, or None."""
        def fix(match):
            token = match.group()
            word = normalize(token)
            if (token in OPERATORS or len(word) < 3 or word in STOP_WORDS
                    or stem(word) in self.index.postings):
                return token
            return self.correct_word(word) or token

        corrected = re.sub(r"[^\W\d_]+", fix, query)
        return corrected if corrected != query else None
