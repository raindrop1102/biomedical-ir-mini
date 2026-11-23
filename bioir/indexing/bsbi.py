"""Part 2: Blocked Sort-Based Indexing (Manning et al., Introduction to IR, §4.2).

    1. divide the documents into blocks of `block_docs` documents
    2. turn every term occurrence of a block into a (termID, docID) pair
    3. sort the block's pairs by termID, then docID
    4. collapse the sorted pairs into the block's partial index; write it to disk
    5. merge all partial indexes at once (k-way, by termID) into the final index

termIDs come from one term → termID table shared by every block, so termID 17 is the same term in
all partial indexes — which is what lets step 5 merge on termID alone. The table stays in memory
for the whole run (and is written to terms.tsv for the merge).
"""
import heapq
import time
from collections.abc import Iterable
from itertools import groupby, islice
from pathlib import Path

from .stats import Stats
from .storage import FinalWriter, index_size, read_bsbi_block, read_terms, write_bsbi_block, write_terms

SAMPLE = 12        # pairs / entries kept from the first block for the demo


def _blocks(documents: Iterable, size: int):
    iterator = iter(documents)
    while block := list(islice(iterator, size)):
        yield block


def invert_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Sorted (termID, docID) pairs → [(termID, ascending unique docIDs)]. The sort already put each
    term's docIDs in order, so a repeated pair is always next to its twin."""
    partial = []
    for term_id, group in groupby(pairs, key=lambda p: p[0]):
        doc_ids = []
        for _, doc in group:
            if not doc_ids or doc_ids[-1] != doc:
                doc_ids.append(doc)
        partial.append((term_id, doc_ids))
    return partial


def bsbi(documents: Iterable, output: Path, work: Path, block_docs: int = 2000, trace: dict | None = None) -> Stats:
    """Index (docID, terms) documents into `output`; partial indexes go to `work`."""
    if block_docs < 1:
        raise ValueError("block_docs must be at least 1")
    output, work = Path(output), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stats = Stats("BSBI", setting=f"{block_docs:,} 篇／block")
    term_ids: dict[str, int] = {}
    terms: list[str] = []
    started = time.perf_counter()
    blocks = []
    for number, block in enumerate(_blocks(documents, block_docs), 1):
        clock = time.perf_counter()
        pairs = []                                            # step 2
        for doc_id, doc_terms in block:
            for term in doc_terms:
                term_id = term_ids.get(term)
                if term_id is None:
                    term_id = term_ids[term] = len(terms)
                    terms.append(term)
                pairs.append((term_id, doc_id))
        stats.add_time("產生 pairs", clock)
        if trace is not None and number == 1:
            trace["pairs"] = [(t, d, terms[t]) for t, d in pairs[:SAMPLE]]
        clock = time.perf_counter()
        pairs.sort()                                          # step 3
        stats.add_time("排序 pairs", clock)
        if trace is not None and number == 1:
            trace["sorted"] = [(t, d, terms[t], sum(1 for _ in g)) for (t, d), g in islice(groupby(pairs), SAMPLE)]
        clock = time.perf_counter()
        partial = invert_pairs(pairs)                         # step 4
        stats.add_time("建局部索引", clock)
        clock = time.perf_counter()
        path = work / f"bsbi_block_{number:03d}.bin"
        size = write_bsbi_block(path, partial)
        stats.add_time("寫入磁碟", clock)
        stats.documents += len(block)
        stats.pairs += len(pairs)
        stats.peak_items = max(stats.peak_items, len(pairs))
        blocks.append(path)
        stats.blocks_info.append({"block": number, "first": block[0][0], "last": block[-1][0], "documents": len(block),
                                  "pairs": len(pairs), "terms": len(partial),
                                  "postings": sum(len(d) for _, d in partial), "bytes": size})
        if trace is not None and number == 1:
            trace["partial"] = [(t, terms[t], d[:8], len(d)) for t, d in partial[:SAMPLE]]
        del pairs, partial
    clock = time.perf_counter()
    terms_size = write_terms(work / "terms.tsv", terms)
    stats.add_time("寫入磁碟", clock)
    stats.intermediate_files = len(blocks) + 1
    stats.intermediate_bytes = sum(b["bytes"] for b in stats.blocks_info) + terms_size
    del term_ids, terms

    stats.mark_memory("建立局部索引")
    clock = time.perf_counter()                               # step 5
    terms = read_terms(work / "terms.tsv")
    writer = FinalWriter(output)
    merged = heapq.merge(*(read_bsbi_block(path) for path in blocks), key=lambda entry: entry[0])
    for term_id, group in groupby(merged, key=lambda entry: entry[0]):
        doc_ids = [doc for _, docs in group for doc in docs]  # blocks hold increasing docIDs
        writer.add(terms[term_id], doc_ids)
        stats.postings += len(doc_ids)
        if trace is not None and len(trace.setdefault("merged", [])) < SAMPLE and len(doc_ids) > 1:
            trace["merged"].append((term_id, terms[term_id], doc_ids[:10], len(doc_ids)))
    writer.close()
    stats.mark_memory("合併")
    stats.add_time("合併", clock)
    stats.seconds = time.perf_counter() - started
    stats.terms = len(terms)
    stats.blocks = len(blocks)
    stats.final = index_size(output)
    return stats
