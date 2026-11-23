"""On-disk formats: the partial indexes each method writes, and the final index both produce.

Partial index (one file per block), records sorted by key:
    BSBI   termID uint32 · df uint32 · df × docID uint32
    SPIMI  len uint16 · term (UTF-8) · df uint32 · df × docID uint32
BSBI's blocks hold termIDs, so it also writes the termID → term table (terms.tsv) once.

Final index (the same format from both methods):
    dictionary.tsv   term \t df \t byte offset into postings.bin, sorted by term
    postings.bin     each term's docIDs, ascending, uint32 little-endian
Only the dictionary is loaded into memory; a query reads just its terms' postings from disk.
"""
import struct
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path

assert array("I").itemsize == 4
TERM_ID = struct.Struct("<II")        # termID, df
LENGTH = struct.Struct("<H")
COUNT = struct.Struct("<I")


def _ids(doc_ids) -> bytes:
    return array("I", doc_ids).tobytes()


def _read_ids(file, df: int) -> list[int]:
    ids = array("I")
    ids.frombytes(file.read(4 * df))
    return ids.tolist()


def write_bsbi_block(path: Path, postings: Iterable[tuple[int, list[int]]]) -> int:
    """(termID, docIDs) in termID order → file; returns its size in bytes."""
    with open(path, "wb") as file:
        for term_id, doc_ids in postings:
            file.write(TERM_ID.pack(term_id, len(doc_ids)))
            file.write(_ids(doc_ids))
    return path.stat().st_size


def read_bsbi_block(path: Path) -> Iterator[tuple[int, list[int]]]:
    with open(path, "rb") as file:
        while header := file.read(TERM_ID.size):
            term_id, df = TERM_ID.unpack(header)
            yield term_id, _read_ids(file, df)


def write_spimi_block(path: Path, postings: Iterable[tuple[str, list[int]]]) -> int:
    """(term, docIDs) in term order → file; returns its size in bytes."""
    with open(path, "wb") as file:
        for term, doc_ids in postings:
            encoded = term.encode("utf-8")
            file.write(LENGTH.pack(len(encoded)) + encoded + COUNT.pack(len(doc_ids)))
            file.write(_ids(doc_ids))
    return path.stat().st_size


def read_spimi_block(path: Path) -> Iterator[tuple[str, list[int]]]:
    with open(path, "rb") as file:
        while header := file.read(LENGTH.size):
            term = file.read(LENGTH.unpack(header)[0]).decode("utf-8")
            (df,) = COUNT.unpack(file.read(COUNT.size))
            yield term, _read_ids(file, df)


def write_terms(path: Path, terms: list[str]) -> int:
    """BSBI's termID → term table: line i holds termID i."""
    path.write_text("".join(f"{t}\n" for t in terms), encoding="utf-8")
    return path.stat().st_size


def read_terms(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


class FinalWriter:
    """Appends merged postings lists to postings.bin, then writes the sorted dictionary."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = open(self.folder / "postings.bin", "wb")
        self.entries: list[tuple[str, int, int]] = []

    def add(self, term: str, doc_ids: list[int]) -> None:
        self.entries.append((term, len(doc_ids), self.file.tell()))
        self.file.write(_ids(doc_ids))

    def close(self) -> None:
        self.file.close()
        self.entries.sort()
        with open(self.folder / "dictionary.tsv", "w", encoding="utf-8", newline="\n") as file:
            file.writelines(f"{t}\t{df}\t{offset}\n" for t, df, offset in self.entries)


def index_size(folder: Path) -> dict[str, int]:
    folder = Path(folder)
    sizes = {name: (folder / name).stat().st_size for name in ("dictionary.tsv", "postings.bin")}
    sizes["total"] = sum(sizes.values())
    return sizes


class IndexReader:
    """A final index on disk: dictionary in memory, postings read on demand."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.dictionary: dict[str, tuple[int, int]] = {}
        with open(self.folder / "dictionary.tsv", encoding="utf-8") as file:
            for line in file:
                term, df, offset = line.rstrip("\n").split("\t")
                self.dictionary[term] = (int(df), int(offset))

    @property
    def vocabulary(self) -> list[str]:
        return list(self.dictionary)

    def df(self, term: str) -> int:
        return self.dictionary.get(term, (0, 0))[0]

    def postings(self, term: str) -> list[int]:
        if term not in self.dictionary:
            return []
        df, offset = self.dictionary[term]
        with open(self.folder / "postings.bin", "rb") as file:
            file.seek(offset)
            return _read_ids(file, df)

    def __len__(self):
        return len(self.dictionary)
