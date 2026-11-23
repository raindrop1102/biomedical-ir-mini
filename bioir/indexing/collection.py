"""The documents both indexers read: all 21,578 Reuters-21578 stories, analyzed once.

Every <REUTERS> record counts, in or out of ModApte, since indexing needs no topics. docID is the
story's NEWID (1 … 21,578, already in file order). Preprocessing is the search engine's own
(bioir.text.analyze): letters/digits tokens, case folding, stop words dropped, Porter stemming —
so BSBI and SPIMI see exactly the same term stream and only the indexing differs.
"""
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..news.reuters import ATTRIBUTE, STORY, _body, _clean, _field
from ..text import analyze


@dataclass(frozen=True)
class Story:
    doc_id: int
    title: str
    body: str


def load_stories(folder) -> list[Story]:
    """Every story in reut2-000.sgm … reut2-021.sgm, in NEWID order."""
    files = sorted(Path(folder).glob("reut2-*.sgm"))
    if not files:
        raise FileNotFoundError(f"No reut2-*.sgm files in {folder}; see README for the download")
    stories = []
    for path in files:
        for story in STORY.finditer(path.read_text(encoding="latin-1")):
            newid = int(dict(ATTRIBUTE.findall(story["attributes"]))["NEWID"])
            content = story["content"]
            stories.append(Story(newid, _clean(_field(content, "TITLE")), _clean(_body(content))))
    return sorted(stories, key=lambda s: s.doc_id)


def write_titles(stories, path: Path) -> None:
    """docID, tab, title per line, so query results show without the .sgm files (no title: body start)."""
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        for s in stories:
            title = " ".join((s.title or s.body[:80]).split())
            file.write(f"{s.doc_id}\t{title}\n")


def read_titles(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as file:
        return {int(doc): title for doc, title in (line.rstrip("\n").split("\t", 1) for line in file)}


def terms(text: str) -> list[str]:
    """Tokenize, lower-case, drop stop words, stem — every occurrence, in order."""
    return [term for term, _ in analyze(text)]


def analyze_stories(stories) -> list[tuple[int, list[str]]]:
    """(docID, terms of title + body) for each story."""
    return [(s.doc_id, terms(f"{s.title} . {s.body}")) for s in stories]


def scaled(documents: list[tuple[int, list[str]]], n: int) -> Iterator[tuple[int, list[str]]]:
    """n documents for the scaling runs: the collection repeated, renumbered 1 … n.

    Copies share the vocabulary, so this grows postings and pairs but not the dictionary — real
    collections keep adding terms (Heaps' law), so dictionary-bound costs are underestimated."""
    for i in range(n):
        yield i + 1, documents[i % len(documents)][1]
