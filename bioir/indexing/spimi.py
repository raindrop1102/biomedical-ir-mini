"""Part 3: Single-Pass In-Memory Indexing (Manning et al., Introduction to IR, §4.3).

    1. read the documents one after another
    2. for every term: a new term gets a dictionary entry; its postings list gets the current
       docID unless it already ends with it (a term repeated in a document is posted once)
    3. when the postings in memory reach `capacity`, sort the dictionary's terms, write
       dictionary + postings to disk as a partial index, empty memory and go on
    4. merge all partial indexes at once (k-way, by term) into the final index

No (termID, docID) pairs and no termIDs: postings grow directly under the term itself, and the
only sort is of each block's distinct terms. The memory check runs after each document, so a
document never straddles two blocks.
"""
import heapq
import time
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path

from .stats import Stats
from .storage import FinalWriter, index_size, read_spimi_block, write_spimi_block

SAMPLE = 12


def spimi(documents: Iterable, output: Path, work: Path, capacity: int = 200_000, trace: dict | None = None) -> Stats:
    """Index (docID, terms) documents into `output`; capacity is postings held in memory per block."""
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    output, work = Path(output), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    stats = Stats("SPIMI", setting=f"{capacity:,} postings／block")
    blocks = []
    dictionary: dict[str, list[int]] = {}
    held, first, documents_in_block = 0, None, 0
    started = time.perf_counter()
    clock = time.perf_counter()

    def flush(last_doc: int):
        nonlocal dictionary, held, first, documents_in_block, clock
        stats.add_time("建 dictionary", clock)
        clock = time.perf_counter()
        ordered = sorted(dictionary)                          # step 3: sort the terms only
        stats.add_time("排序 terms", clock)
        number = len(blocks) + 1
        if trace is not None and number == 1:
            trace["dictionary"] = [(t, dictionary[t][:8], len(dictionary[t])) for t in list(dictionary)[:SAMPLE]]
            trace["sorted"] = [(t, dictionary[t][:8], len(dictionary[t])) for t in ordered[:SAMPLE]]
        clock = time.perf_counter()
        path = work / f"spimi_block_{number:03d}.bin"
        size = write_spimi_block(path, ((t, dictionary[t]) for t in ordered))
        stats.add_time("寫入磁碟", clock)
        blocks.append(path)
        stats.blocks_info.append({"block": number, "first": first, "last": last_doc, "documents": documents_in_block,
                                  "terms": len(dictionary), "postings": held, "bytes": size})
        stats.peak_items = max(stats.peak_items, held)
        dictionary, held, first, documents_in_block = {}, 0, None, 0   # empty memory
        clock = time.perf_counter()

    doc_id = None
    for doc_id, doc_terms in documents:                       # step 1
        if first is None:
            first = doc_id
        documents_in_block += 1
        stats.documents += 1
        for term in doc_terms:                                # step 2
            postings = dictionary.get(term)
            if postings is None:
                postings = dictionary[term] = []
            if not postings or postings[-1] != doc_id:
                postings.append(doc_id)
                held += 1
        if held >= capacity:
            flush(doc_id)
    if dictionary:
        flush(doc_id)
    stats.intermediate_files = len(blocks)
    stats.intermediate_bytes = sum(b["bytes"] for b in stats.blocks_info)

    stats.mark_memory("建立局部索引")
    clock = time.perf_counter()                               # step 4
    writer = FinalWriter(output)
    merged = heapq.merge(*(read_spimi_block(path) for path in blocks), key=lambda entry: entry[0])
    for term, group in groupby(merged, key=lambda entry: entry[0]):
        doc_ids = [doc for _, docs in group for doc in docs]
        writer.add(term, doc_ids)
        stats.postings += len(doc_ids)
        stats.terms += 1
        if trace is not None and len(trace.setdefault("merged", [])) < SAMPLE and len(doc_ids) > 1:
            trace["merged"].append((term, doc_ids[:10], len(doc_ids)))
    writer.close()
    stats.mark_memory("合併")
    stats.add_time("合併", clock)
    stats.seconds = time.perf_counter() - started
    stats.blocks = len(blocks)
    stats.final = index_size(output)
    return stats
