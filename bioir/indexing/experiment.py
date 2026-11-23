"""Homework 5's experiment: BSBI against SPIMI on the same documents and preprocessing.

    main     the full collection (21,578 stories), BSBI 2,000 documents per block against SPIMI
             200,000 postings per block; time is the median of `repeats` runs, memory a separate
             run under tracemalloc (tracing slows Python down, so it never touches the timings)
    blocks   block size swept at the full collection: what more, smaller blocks cost
    scaling  10,000 … 1,000,000 documents (the collection repeated, see collection.scaled)
    queries  single-term, AND and OR queries on the basic, BSBI and SPIMI indexes, compared

Preprocessing is done once and shared, so every timing is indexing alone. Writes CSVs, JSON with
the intermediate results, a chart, and keeps the two final indexes of the main run for the web page.
"""
import csv
import json
import shutil
import statistics
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from .basic import InvertedIndex, boolean, query_terms
from .bsbi import bsbi
from .collection import analyze_stories, load_stories, scaled, write_titles
from .spimi import spimi
from .storage import IndexReader

BLOCK_DOCS = 2_000
CAPACITY = 200_000
BLOCK_SWEEP = {"BSBI": [500, 1_000, 2_000, 5_000, 25_000], "SPIMI": [50_000, 100_000, 200_000, 500_000, 2_000_000]}
SIZES = [10_000, 21_578, 50_000, 100_000, 215_780, 1_000_000]
QUERIES = ["information retrieval", "artificial intelligence", "financial market", "medical research",
           "computer science", "crude oil prices", "interest rates", "trade deficit"]


def _run(method: str, documents, setting: int, output: Path, work: Path, trace=None):
    if method == "BSBI":
        return bsbi(documents, output, work, block_docs=setting, trace=trace)
    return spimi(documents, output, work, capacity=setting, trace=trace)


def measure(method: str, make_documents, setting: int, scratch: Path, repeats: int = 1, keep: Path | None = None,
            trace=None) -> dict:
    """Time (median of `repeats`) and traced memory of one method at one setting."""
    times, stats = [], None
    for attempt in range(repeats):
        output, work = scratch / f"{method}_out", scratch / f"{method}_work"
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
        stats = _run(method, make_documents(), setting, output, work, trace if attempt == 0 else None)
        times.append(stats.seconds)
        if attempt < repeats - 1:
            shutil.rmtree(work)
    stats.seconds = statistics.median(times)
    if keep is not None:
        shutil.rmtree(keep, ignore_errors=True)
        shutil.copytree(scratch / f"{method}_out", keep)
    shutil.rmtree(scratch / f"{method}_out")
    shutil.rmtree(scratch / f"{method}_work")
    output, work = scratch / f"{method}_mem_out", scratch / f"{method}_mem_work"
    tracemalloc.start()
    traced = _run(method, make_documents(), setting, output, work)
    tracemalloc.stop()
    shutil.rmtree(output)
    shutil.rmtree(work)
    return {"stats": stats, "times": times, "memory": traced.memory, "peak": max(traced.memory.values())}


def _row(result: dict, **extra) -> dict:
    s = result["stats"]
    return {**extra, "method": s.method, "setting": s.setting, "documents": s.documents, "seconds": round(s.seconds, 3),
            "peak_mb": round(result["peak"] / 2**20, 2),
            "build_peak_mb": round(result["memory"].get("建立局部索引", 0) / 2**20, 2),
            "merge_peak_mb": round(result["memory"].get("合併", 0) / 2**20, 2),
            "blocks": s.blocks, "intermediate_files": s.intermediate_files,
            "intermediate_mb": round(s.intermediate_bytes / 2**20, 2), "final_mb": round(s.final["total"] / 2**20, 2),
            "pairs": s.pairs, "postings": s.postings, "terms": s.terms, "peak_items": s.peak_items}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_queries(basic: InvertedIndex, indexes: dict, queries=QUERIES) -> list[dict]:
    """Every query's single-term, AND and OR results on each index, and whether they agree."""
    report = []
    for text in queries:
        terms = query_terms(text)
        entry = {"query": text, "terms": [], "operators": {}}
        for term in terms:
            lists = {name: index.postings(term) for name, index in indexes.items()}
            lists["基本索引"] = basic.lookup(term)
            entry["terms"].append({"term": term, "df": len(lists["基本索引"]), "first": lists["基本索引"][:10],
                                   "same": len({tuple(v) for v in lists.values()}) == 1})
        for operator in ("AND", "OR"):
            lists = {name: boolean(index, text, operator) for name, index in indexes.items()}
            lists["基本索引"] = boolean(basic, text, operator)
            entry["operators"][operator] = {"count": len(lists["基本索引"]), "first": lists["基本索引"][:10],
                                            "same": len({tuple(v) for v in lists.values()}) == 1}
        report.append(entry)
    return report


def run(data: Path, results: Path, sizes=SIZES, repeats: int = 3, log=print) -> dict:
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stories = load_stories(data)
    documents = analyze_stories(stories)
    preprocessing = time.perf_counter() - started
    tokens = sum(len(t) for _, t in documents)
    log(f"{len(documents):,} 篇、{tokens:,} 個 term（前處理 {preprocessing:.1f} s，兩種方法共用）")
    basic = InvertedIndex(documents)
    write_titles(stories, results / "titles.tsv")
    settings = {"BSBI": BLOCK_DOCS, "SPIMI": CAPACITY}

    with tempfile.TemporaryDirectory(prefix="bioir_index_") as folder:
        scratch = Path(folder)
        main, traces = {}, {}
        for method, setting in settings.items():
            traces[method] = {}
            main[method] = measure(method, lambda: documents, setting, scratch, repeats,
                                   keep=results / "index" / method.lower(), trace=traces[method])
            log(f"主實驗 {method}: {main[method]['stats'].seconds:.2f} s, {main[method]['peak'] / 2**20:.1f} MB")
        _write_csv(results / "main.csv", [_row(main[m]) for m in settings])

        sweep = []
        for method, values in BLOCK_SWEEP.items():
            for value in values:
                result = measure(method, lambda: documents, value, scratch, repeats)
                sweep.append(_row(result))
                log(f"block 大小 {method} {value:,}: {result['stats'].blocks} blocks, {result['stats'].seconds:.2f} s")
        _write_csv(results / "blocks.csv", sweep)

        scaling = []
        for n in sizes:
            for method, setting in settings.items():
                result = measure(method, lambda: scaled(documents, n), setting, scratch, repeats if n <= 50_000 else 1)
                scaling.append(_row(result, n=n))
                log(f"規模 {n:,} {method}: {result['stats'].seconds:.1f} s, {result['peak'] / 2**20:.1f} MB, "
                    f"{result['stats'].blocks} blocks")
        _write_csv(results / "scaling.csv", scaling)

    indexes = {name: IndexReader(results / "index" / name.lower()) for name in settings}
    queries = compare_queries(basic, indexes)
    equal = all(indexes[m].postings(t) == basic.lookup(t) for m in indexes for t in basic.vocabulary) and \
        all(len(indexes[m]) == len(basic) for m in indexes)
    log(f"BSBI、SPIMI 與基本索引的每個 postings list {'完全相同' if equal else '不一致！'}")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "collection": "Reuters-21578, all 21,578 stories (docID = NEWID)",
        "preprocessing": "letters/digits tokens, case folding, stop words removed, Porter stemming (bioir.text.analyze)",
        "documents": len(documents), "tokens": tokens, "preprocessing_seconds": round(preprocessing, 2),
        "vocabulary": len(basic), "postings": sum(len(p) for p in basic.postings.values()),
        "settings": settings, "repeats": repeats, "equal": equal,
        "main": {m: {"row": _row(main[m]), "times": [round(t, 3) for t in main[m]["times"]],
                     "phases": {k: round(v, 3) for k, v in main[m]["stats"].phases.items()},
                     "memory_mb": {k: round(v / 2**20, 2) for k, v in main[m]["memory"].items()},
                     "blocks": main[m]["stats"].blocks_info, "final": main[m]["stats"].final}
                 for m in settings},
        "trace": traces, "queries": queries,
        "basic_sample": [(t, basic.df(t), basic.lookup(t)[:10]) for t in
                         ("inform", "retriev", "oil", "crude", "opec", "wheat", "tariff", "yen")],
    }
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        from .plot import plot
        plot(results)
    except ImportError:
        log("沒有 matplotlib，略過圖表")
    log(f"完成（{time.perf_counter() - started:.0f} s），結果在 {results}")
    return summary
