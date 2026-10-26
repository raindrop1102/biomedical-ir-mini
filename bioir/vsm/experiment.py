"""Controlled term-weighting experiments: each comparison changes one SMART letter.

Writes, into one folder:
  summary.csv        mean metrics per scheme over all queries
  per_query.csv      every metric for every query and scheme
  comparisons.csv    the controlled pairs: mean difference and exact paired permutation p-value
  judgments.csv      the MeSH relevance judgments (query, pmid, grade)
  idf.csv            df and IDF of every query term and of the most common terms (RQ2)
  tf_example.csv     raw vs log TF components of one long document's vector (RQ1)
  length_bias.csv    length of the top-10 documents per scheme vs the corpus (RQ4)
  manifest.json      corpus fingerprint, queries, schemes
  metrics.png        (with plot=True and matplotlib) the summary as a chart
"""
import csv
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ..text import analyze
from .engine import VectorSpace
from .evaluate import METRICS, judgments, load_queries, paired_permutation, score
from .weighting import Scheme

# "x.x" schemes weight both sides alike, so IDF enters the score twice (IDF²);
# "lnc.ltc" and "bnc.btc" apply it once, on the query side only.
SCHEMES = {
    "nnn.nnn": "raw TF only, no normalization",
    "nnc.nnc": "raw TF only, cosine",
    "lnc.lnc": "log TF only, cosine",
    "bnc.bnc": "binary, cosine",
    "btc.btc": "IDF only (binary × IDF) on both sides, cosine",
    "bnc.btc": "IDF only, applied once (query side), cosine",
    "ntn.ntn": "raw TF-IDF, no normalization",
    "ntc.ntc": "raw TF-IDF, cosine",
    "ltn.ltn": "log TF-IDF, no normalization",
    "ltc.ltc": "log TF-IDF, cosine",
    "atc.atc": "augmented TF-IDF, cosine",
    "lpc.lpc": "log TF × probabilistic IDF, cosine",
    "lnc.ltc": "SMART classic: documents log TF cosine, queries log TF-IDF cosine",
    "Lnc.ltc": "lnc.ltc with log average TF on documents (cosine cancels L: same ranking)",
    "lnu.ltc": "lnc.ltc with pivoted unique normalization on documents",
    "Lnu.ltc": "log average TF, pivoted unique (Singhal et al.'s Lnu)",
    "lnb.ltc": "lnc.ltc with byte-size normalization on documents",
}

# (research question, what changes, baseline, variant)
COMPARISONS = [
    ("RQ1", "raw → log TF (no IDF)", "nnc.nnc", "lnc.lnc"),
    ("RQ1", "raw → log TF (with IDF)", "ntc.ntc", "ltc.ltc"),
    # Under cosine L only rescales a document by a constant, so Lnc.ltc ranks exactly like
    # lnc.ltc; the comparison is meaningful with a normalization that ignores the weights.
    ("RQ1", "log → log average TF (pivoted unique)", "lnu.ltc", "Lnu.ltc"),
    ("RQ2", "add IDF once (query side)", "lnc.lnc", "lnc.ltc"),
    ("RQ2", "IDF once → IDF on both sides (IDF²)", "lnc.ltc", "ltc.ltc"),
    ("RQ2", "add IDF to log TF on both sides", "lnc.lnc", "ltc.ltc"),
    ("RQ2", "add IDF to raw TF", "nnc.nnc", "ntc.ntc"),
    ("RQ3", "TF only → TF-IDF (IDF once)", "lnc.lnc", "lnc.ltc"),
    ("RQ3", "IDF only → TF-IDF (IDF once)", "bnc.btc", "lnc.ltc"),
    ("RQ3", "TF only → TF-IDF (IDF²)", "lnc.lnc", "ltc.ltc"),
    ("RQ3", "IDF only → TF-IDF", "btc.btc", "ltc.ltc"),
    ("RQ3", "TF only → IDF only", "lnc.lnc", "btc.btc"),
    ("RQ4", "add L2 normalization (log TF-IDF)", "ltn.ltn", "ltc.ltc"),
    ("RQ4", "add L2 normalization (raw TF-IDF)", "ntn.ntn", "ntc.ntc"),
    ("RQ4", "add L2 normalization (raw TF only)", "nnn.nnn", "nnc.nnc"),
    ("RQ4", "cosine → pivoted unique normalization", "lnc.ltc", "lnu.ltc"),
    ("RQ4", "cosine → pivoted unique, with log average TF", "Lnc.ltc", "Lnu.ltc"),
    ("RQ4", "cosine → byte-size normalization", "lnc.ltc", "lnb.ltc"),
]

COMPARED_METRICS = ["P@10", "MAP", "nDCG@10", "MRR"]


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _tf_example(space: VectorSpace) -> list[dict]:
    """Ten heaviest terms of the document that repeats one word most often — a word, not a number
    or a term found in half the corpus — and each term's share of the L2-normalized vector under
    raw and log TF."""
    def repeated_word(pmid):
        return max((tf for t, tf in space.tf[pmid].items() if t.isalpha() and space.df(t) < len(space) / 2),
                   default=0)
    pmid = max(sorted(space.tf), key=repeated_word)
    counts = space.tf[pmid]
    raw = {t: float(c) for t, c in counts.items()}
    log = {t: space.weight(Scheme.parse("lnn.lnn").document, t, pmid) for t in counts}
    raw_norm, log_norm = (sum(v * v for v in w.values()) ** 0.5 for w in (raw, log))
    return [{"pmid": pmid, "term": t, "tf": counts[t], "raw_weight": raw[t], "log_weight": round(log[t], 4),
             "raw_share": round(raw[t] / raw_norm, 4), "log_share": round(log[t] / log_norm, 4)}
            for t, _ in counts.most_common(10)]


def _idf_table(space: VectorSpace, queries) -> list[dict]:
    query_terms = {t for q in queries for t, _ in analyze(q.text)}
    common = sorted(space.postings, key=lambda t: (-space.df(t), t))[:15]
    rows = []
    for term in list(dict.fromkeys(common + sorted(query_terms, key=lambda t: space.df(t)))):
        df = space.df(term)
        if not df:
            continue
        t = Scheme.parse("ntn.ntn").document.idf_weight(df, len(space))
        p = Scheme.parse("npn.npn").document.idf_weight(df, len(space))
        rows.append({"term": term, "group": "query term" if term in query_terms else "most common",
                     "df": df, "df_share": round(df / len(space), 4), "idf": round(t, 4), "probabilistic_idf": round(p, 4)})
    return rows


def run(documents, queries_path, folder, schemes=None, plot: bool = False, corpus=None, progress=None) -> list[dict]:
    """Evaluate every scheme on every query and write the files listed above; returns the summary rows."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    space = VectorSpace(documents)
    queries = load_queries(queries_path)
    schemes = schemes or list(SCHEMES)
    relevant = {q.id: judgments(documents, q) for q in queries}
    per_query, rankings = [], {}
    for code in schemes:
        if progress:
            progress(code)
        for query in queries:
            ranking = [h.pmid for h in space.search(query.text, code).hits]
            rankings[code, query.id] = ranking
            metrics = score(ranking, relevant[query.id])
            per_query.append({"scheme": code, "query": query.id, "relevant": metrics.relevant,
                              "retrieved": metrics.retrieved, **{m: round(v, 4) for m, v in metrics.values.items()}})
    summary = []
    for code in schemes:
        rows = [r for r in per_query if r["scheme"] == code]
        summary.append({"scheme": code, "description": SCHEMES.get(code, Scheme.parse(code).describe()),
                        **{m: round(statistics.mean(r[m] for r in rows), 4) for m in METRICS}})
    comparisons = []
    for question, change, base, variant in COMPARISONS:
        if base not in schemes or variant not in schemes:
            continue
        for metric in COMPARED_METRICS:
            a = [r[metric] for r in per_query if r["scheme"] == variant]
            b = [r[metric] for r in per_query if r["scheme"] == base]
            comparisons.append({"question": question, "change": change, "baseline": base, "variant": variant,
                                "metric": metric, "baseline_mean": round(statistics.mean(b), 4),
                                "variant_mean": round(statistics.mean(a), 4),
                                "difference": round(statistics.mean(a) - statistics.mean(b), 4),
                                "better": sum(x > y for x, y in zip(a, b)), "worse": sum(x < y for x, y in zip(a, b)),
                                "p_value": round(paired_permutation(a, b), 4)})
    length_bias = []
    mean_length = statistics.mean(space.lengths.values())
    longest = sorted(space.lengths.values())[int(0.75 * len(space))]
    for code in schemes:
        tops = [p for q in queries for p in rankings[code, q.id][:10]]
        length_bias.append({"scheme": code, "corpus_mean_terms": round(mean_length, 1),
                            "top10_mean_terms": round(statistics.mean(space.lengths[p] for p in tops), 1),
                            "top10_share_in_longest_quarter": round(sum(space.lengths[p] >= longest for p in tops)
                                                                    / len(tops), 3)})
    judged = [{"query": q.id, "text": q.text, "rule": q.rule(), "pmid": p, "grade": g}
              for q in queries for p, g in sorted(relevant[q.id].items())]
    for name, rows in (("summary", summary), ("per_query", per_query), ("comparisons", comparisons),
                       ("judgments", judged), ("idf", _idf_table(space, queries)),
                       ("tf_example", _tf_example(space)), ("length_bias", length_bias)):
        _write(folder / f"{name}.csv", rows)
    text = json.dumps([[d.pmid, d.title, d.abstract] for d in documents], ensure_ascii=False)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "corpus": {"file": Path(corpus).name if corpus else None,
                           "file_sha256": hashlib.sha256(Path(corpus).read_bytes()).hexdigest() if corpus else None,
                           "documents": len(documents), "terms": len(space.postings),
                           "texts_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
                "queries": [{"id": q.id, "query": q.text, "rule": q.rule(), "relevant": len(relevant[q.id]),
                             "grade_2": sum(g == 2 for g in relevant[q.id].values())} for q in queries],
                "schemes": {code: SCHEMES.get(code, "") for code in schemes},
                "metrics": METRICS, "significance": "exact two-sided paired permutation (sign-flip) test over queries",
                "text_fields": "title + abstract; stop words removed; Porter stems; bag of words",
                "python": platform.python_version()}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if plot:
        from .plot import plot_summary
        plot_summary(summary, folder / "metrics.png", len(queries))
    return summary
