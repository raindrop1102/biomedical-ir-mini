"""Homework 4's experiment in one call: method A against method B on the same queries.

Methods (every one ranks the ModApte test stories; models come from `news train`):
    A           TF-IDF vectors, cosine (homework 3's ltc.ltc)
    B-<model>   TF-IDF-weighted average of word vectors, cosine        ← the core comparison
    mean-<m>    plain average of word vectors (ablation: what TF-IDF weighting adds)
    A+B-<m>     0.5 · A's cosine + 0.5 · B's cosine (α fixed beforehand, not tuned on these queries)

Writes summary.csv (all / keyword / paraphrase queries), per_query.csv, comparisons.csv
(each method against A, with an exact paired permutation test), coverage.csv,
zero_overlap.csv (top-10 stories sharing no query word), neighbors.csv, pairs.csv,
projections (PCA, t-SNE), manifest.json and, with plot=True, metrics.png.
"""
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from ..embed import Embeddings
from ..embed.evaluate import write_projections
from ..vsm.evaluate import paired_permutation
from .queries import coverage, judgments, load_queries
from .retrieval import METRICS, evaluate, mean, test_space, tfidf_ranker
from .reuters import load_reuters
from .semantic import EmbeddingSpace, hybrid, words
from .words import PROJECTION_GROUPS, word_report

MODELS = ("cbow", "skipgram")
TFIDF_SCHEME = "ltc.ltc"
ALPHA = 0.5


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def methods(articles, models: dict) -> tuple[dict, dict]:
    """name → (description, ranker), and name → EmbeddingSpace for the B methods."""
    space = test_space(articles)
    found = {"A": (f"TF-IDF vectors ({TFIDF_SCHEME}), cosine", tfidf_ranker(space, TFIDF_SCHEME))}
    spaces = {}
    for name, model in models.items():
        weighted = EmbeddingSpace(articles, model)
        spaces[f"B-{name}"] = weighted
        found[f"B-{name}"] = (f"{name}: TF-IDF-weighted average of word vectors, cosine", weighted.ranker())
        found[f"mean-{name}"] = (f"{name}: plain average of word vectors, cosine",
                                 EmbeddingSpace(articles, model, "mean").ranker())
        found[f"A+B-{name}"] = (f"{ALPHA} × A + {1 - ALPHA} × B-{name}", hybrid(space, TFIDF_SCHEME, weighted, ALPHA))
    return found, spaces


def run(data, queries_path, models_dir, folder, plot: bool = False, progress=None) -> list[dict]:
    """Evaluate every method and write the files listed in the module docstring; returns summary rows."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    articles = load_reuters(data)
    queries = load_queries(queries_path)
    models = {name: Embeddings.load(Path(models_dir) / f"{name}.npz") for name in MODELS
              if (Path(models_dir) / f"{name}.npz").exists()}
    if not models:
        raise FileNotFoundError(f"No cbow.npz or skipgram.npz in {models_dir}; run `python -m bioir news train` first")
    rankers, spaces = methods(articles, models)
    per_query, summary = [], []
    for name, (description, ranker) in rankers.items():
        if progress:
            progress(name)
        rows = evaluate(ranker, articles, queries)
        per_query += [{"method": name, **row} for row in rows]
        for scope in ("all", "keyword", "paraphrase"):
            summary.append({"method": name, "description": description, "queries": scope,
                            **mean(rows, None if scope == "all" else scope)})
    comparisons = []
    for name in rankers:
        if name == "A":
            continue
        for metric in ("nDCG@10", "P@10", "MRR"):
            a = [r[metric] for r in per_query if r["method"] == "A"]
            b = [r[metric] for r in per_query if r["method"] == name]
            comparisons.append({"baseline": "A", "method": name, "metric": metric,
                                "baseline_mean": round(sum(a) / len(a), 4), "method_mean": round(sum(b) / len(b), 4),
                                "better": sum(y > x for x, y in zip(a, b)), "worse": sum(y < x for x, y in zip(a, b)),
                                "p_value": round(paired_permutation(b, a), 4)})
    texts = {a.newid: f"{a.title} . {a.body}" for a in articles if a.split == "test"}
    zero = []
    for name, space in spaces.items():
        ranker = rankers[name][1]
        for query in queries:
            relevant = judgments(articles, query)
            query_words = set(words(query.text, space.model))
            top = ranker(query.text)[:10]
            no_shared = [n for n in top if not query_words & set(words(texts[n], space.model))]
            zero.append({"method": name, "query": query.id, "kind": query.kind, "top10_no_query_word": len(no_shared),
                         "of_which_relevant": sum(n in relevant for n in no_shared)})
    neighbors, pairs = [], []
    for name, model in models.items():
        n, p = word_report(model)
        neighbors += [{"model": name, **row} for row in n]
        pairs += [{"model": name, **row} for row in p]
    for name, rows in (("summary", summary), ("per_query", per_query), ("comparisons", comparisons),
                       ("coverage", [{"query": q.id, "text": q.text, "topic": q.topic, "kind": q.kind,
                                      **coverage(articles, q)} for q in queries]),
                       ("zero_overlap", zero), ("neighbors", neighbors), ("pairs", pairs)):
        _write(folder / f"{name}.csv", rows)
    write_projections(models, folder, plot, PROJECTION_GROUPS)
    manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "collection": "Reuters-21578, ModApte split",
                "train_stories": sum(a.split == "train" for a in articles),
                "test_stories": sum(a.split == "test" for a in articles),
                "relevance": "a test story is relevant when the query's topic is one of its TOPICS",
                "queries": [{"id": q.id, "text": q.text, "topic": q.topic, "kind": q.kind,
                             "relevant": len(judgments(articles, q))} for q in queries],
                "methods": {name: description for name, (description, _) in rankers.items()},
                "models": {name: {"config": m.metadata.get("config"), "sanity": m.metadata.get("sanity"),
                                  "vocabulary": len(m)} for name, m in models.items()},
                "metrics": METRICS, "significance": "exact two-sided paired permutation test over the queries",
                "python": platform.python_version()}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if plot:
        from .plot import plot_summary
        plot_summary(summary, folder / "metrics.png")
    return summary
