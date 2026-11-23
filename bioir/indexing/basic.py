"""Part 1: the basic inverted index, and the Boolean retrieval every index shares.

    dictionary: term → df (and, on disk, where its postings start)
    postings:   term → ascending docIDs of the documents that contain it

Built the direct way, in memory, one document after another: this is the reference that the BSBI
and SPIMI indexes must equal exactly.
"""
from .collection import terms as analyze_terms


class InvertedIndex:
    def __init__(self, documents):
        """documents: (docID, terms) in ascending docID order."""
        self.postings: dict[str, list[int]] = {}
        self.documents = 0
        for doc_id, terms in documents:
            self.documents += 1
            for term in terms:
                postings = self.postings.setdefault(term, [])
                if not postings or postings[-1] != doc_id:      # once per document
                    postings.append(doc_id)
        self.postings = dict(sorted(self.postings.items()))

    @property
    def vocabulary(self) -> list[str]:
        return list(self.postings)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def lookup(self, term: str) -> list[int]:
        return self.postings.get(term, [])

    def __len__(self):
        return len(self.postings)


def intersect(a: list[int], b: list[int]) -> list[int]:
    """Docs in both sorted lists: walk them together, O(len a + len b)."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return result


def union(a: list[int], b: list[int]) -> list[int]:
    """Docs in either sorted list, sorted, each once."""
    result, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    return result + a[i:] + b[j:]


def query_terms(text: str) -> list[str]:
    """The query's terms after the documents' preprocessing, each once, in order."""
    return list(dict.fromkeys(analyze_terms(text)))


def postings_of(index, term: str) -> list[int]:
    """Postings from an in-memory InvertedIndex or an on-disk IndexReader."""
    return index.lookup(term) if hasattr(index, "lookup") else index.postings(term)


def boolean(index, text: str, operator: str) -> list[int]:
    """AND / OR of the query's terms. AND starts from the rarest term, so lists only shrink."""
    lists = [postings_of(index, t) for t in query_terms(text)]
    if not lists:
        return []
    if operator == "AND":
        lists.sort(key=len)
        result = lists[0]
        for other in lists[1:]:
            result = intersect(result, other)
        return result
    if operator == "OR":
        result = []
        for other in lists:
            result = union(result, other)
        return result
    raise ValueError("operator must be AND or OR")
