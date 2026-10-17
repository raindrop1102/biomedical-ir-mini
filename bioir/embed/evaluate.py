"""Checks and comparisons for trained embeddings.

sanity(): the cosine of random word pairs. Without it a neighbour's cosine has no
scale: the original BiomedicalIR CBOW showed 0.97 for semaglutide/liraglutide,
but random pairs already averaged 0.93. Healthy word2vec vectors here average 0.2.

compare(): the experiment design of bioscope-ir-hw2 — query words, related pairs
and rare words fixed before training; CBOW and Skip-gram at window 2 and 5
(seed 42) and window 5 again with seed 7; overlap of the top-5 neighbours shows
how much the window and the seed change them.
"""
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SANITY_LIMIT = 0.5     # mean random-pair cosine above this = vectors share one direction

# Fixed before training, for the GLP-1 corpus (the same lists as bioscope-ir-hw2).
QUERY_WORDS = ["glp-1", "semaglutide", "liraglutide", "insulin", "glucose", "diabetes", "obesity",
               "weight", "receptor", "agonist"]
RELATED_PAIRS = [("semaglutide", "liraglutide"), ("insulin", "glucose"), ("diabetes", "obesity"),
                 ("obesity", "weight"), ("receptor", "agonist")]
RARE_WORDS = ["tumor", "diagnosis", "neuropathy", "retinopathy", "gastroparesis", "nephropathy", "cardiomyopathy"]
RUNS = [("cbow", 2, 42), ("cbow", 5, 42), ("cbow", 5, 7),
        ("skipgram", 2, 42), ("skipgram", 5, 42), ("skipgram", 5, 7)]
PROJECTED = ["cbow_w5_s42", "skipgram_w5_s42"]     # models drawn in the PCA / t-SNE figure


def sanity(model, top: int = 2000, samples: int = 20000, seed: int = 0) -> dict:
    """Cosine of random pairs among the `top` most frequent words."""
    unit = model.unit[:top]
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(unit), (2, samples))
    cosines = (unit[i[i != j]] * unit[j[i != j]]).sum(1)
    mean = float(cosines.mean())
    return {"random_pair_mean": round(mean, 4), "random_pair_std": round(float(cosines.std()), 4),
            "random_pair_p95": round(float(np.percentile(cosines, 95)), 4),
            "limit": SANITY_LIMIT, "passed": mean <= SANITY_LIMIT}


def neighbor_rows(model, words, k: int = 5) -> list[dict]:
    rows = []
    for word in dict.fromkeys(words):
        try:
            neighbors = model.similar(word, k)
        except (KeyError, ValueError):
            rows.append({"word": word, "rank": None, "neighbor": None, "cosine": None, "count": None})
            continue
        rows += [{"word": word, "rank": r, "neighbor": n, "cosine": round(c, 4), "count": count}
                 for r, (n, c, count) in enumerate(neighbors, 1)]
    return rows


def pair_rows(model, pairs) -> list[dict]:
    """Cosine of each pair and how many random-pair standard deviations it lies above their mean."""
    check = model.metadata.get("sanity") or sanity(model)
    rows = []
    for a, b in pairs:
        try:
            cosine = model.similarity(a, b)
        except (KeyError, ValueError):
            rows.append({"left": a, "right": b, "cosine": None, "z": None})
            continue
        z = (cosine - check["random_pair_mean"]) / check["random_pair_std"]
        rows.append({"left": a, "right": b, "cosine": round(cosine, 4), "z": round(z, 2)})
    return rows


def overlap(left: list[dict], right: list[dict], k: int = 5) -> list[dict]:
    """Per word, the share of top-k neighbours two models have in common."""
    def table(rows):
        result = {}
        for row in rows:
            result.setdefault(row["word"], set())
            if row["neighbor"] is not None and row["rank"] <= k:
                result[row["word"]].add(row["neighbor"])
        return result
    a, b = table(left), table(right)
    return [{"word": w, "overlap": len(a[w] & b[w]) / k if a[w] and b[w] else None,
             "shared": ", ".join(sorted(a[w] & b[w]))} for w in a if w in b]


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_projections(models: dict, folder, plot: bool = False, groups=None) -> list[Path]:
    """projections.csv (PCA and t-SNE points) and projection_quality.csv for each model,
    plus projections.png when plot is true (needs matplotlib). Models with fewer than
    three of the group words (another corpus) are skipped; nothing is written if all are."""
    from .projection import project
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in models.items():
        try:
            results[name] = project(model, groups)
        except ValueError:
            continue
    if not results:
        return []
    points = [{"model": name, **row} for name, (rows, _) in results.items() for row in rows]
    quality = [{"model": name, **row} for name, (_, rows) in results.items() for row in rows]
    paths = [_write(folder / "projections.csv", points), _write(folder / "projection_quality.csv", quality)]
    if plot:
        from .plot import plot_projections
        paths.append(plot_projections(results, folder / "projections.png"))
    return paths


def write_manifest(folder, config, documents, corpus=None) -> Path:
    """What produced a results folder: corpus fingerprint, settings, fixed word lists, versions."""
    import hashlib
    from dataclasses import asdict
    from .projection import PROJECTION_GROUPS
    settings = {k: v for k, v in asdict(config).items() if k not in ("architecture", "window", "seed")}
    text = json.dumps([[d.pmid, d.abstract] for d in documents], ensure_ascii=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": {"file": Path(corpus).name if corpus else None,
                   "file_sha256": hashlib.sha256(Path(corpus).read_bytes()).hexdigest() if corpus else None,
                   "documents": len(documents),
                   "abstracts_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
        "runs": [{"architecture": a, "window": w, "seed": s} for a, w, s in RUNS],
        "shared_settings": settings, "query_words": QUERY_WORDS, "related_pairs": RELATED_PAIRS,
        "rare_words": RARE_WORDS, "projection_groups": PROJECTION_GROUPS, "projected_models": PROJECTED,
        "sanity_limit": SANITY_LIMIT, "python": platform.python_version(), "numpy": np.__version__,
        "notes": ["Cosines and top-5 overlaps describe the vectors, not semantic accuracy.",
                  "Each projection panel is fitted separately; coordinates are not comparable across panels.",
                  "Training is deterministic for a given seed, numpy version and platform."],
    }
    path = Path(folder) / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def compare(documents, folder, config=None, progress=None, plot: bool = False, corpus=None) -> list[dict]:
    """Train the six RUNS (other settings from `config`), save each model and write
    summary.csv, neighbors.csv, pairs.csv, overlap.csv, projections of the PROJECTED
    models and manifest.json. Returns the summary rows."""
    from dataclasses import replace
    from .train import Config, train
    config = config or Config()
    folder = Path(folder)
    (folder / "models").mkdir(parents=True, exist_ok=True)
    models, summary, neighbors, pairs = {}, [], [], []
    for architecture, window, seed in RUNS:
        name = f"{architecture}_w{window}_s{seed}"
        if progress:
            progress(name)
        model = train(documents, replace(config, architecture=architecture, window=window, seed=seed))
        model.save(folder / "models" / f"{name}.npz")
        models[name] = model
        check = model.metadata["sanity"]
        summary.append({"model": name, "architecture": architecture, "window": window, "seed": seed,
                        "vocabulary": len(model), "final_loss": round(model.metadata["history"][-1]["loss"], 4),
                        "seconds": model.metadata["seconds"], "random_pair_mean": check["random_pair_mean"],
                        "random_pair_p95": check["random_pair_p95"], "sanity_passed": check["passed"]})
        neighbors += [{"model": name, **row} for row in neighbor_rows(model, QUERY_WORDS + RARE_WORDS)]
        pairs += [{"model": name, **row} for row in pair_rows(model, RELATED_PAIRS)]
    overlaps = []
    for architecture in ("cbow", "skipgram"):
        rows = {n: [{k: v for k, v in r.items() if k != "model"} for r in neighbors if r["model"] == n]
                for n in models if n.startswith(architecture)}
        for change, (a, b) in (("window 2 → 5", ("w2_s42", "w5_s42")), ("seed 42 → 7", ("w5_s42", "w5_s7"))):
            for row in overlap(rows[f"{architecture}_{a}"], rows[f"{architecture}_{b}"]):
                group = "rare" if row["word"] in RARE_WORDS else "query"
                overlaps.append({"architecture": architecture, "change": change, "group": group, **row})
    for name, rows in (("summary", summary), ("neighbors", neighbors), ("pairs", pairs), ("overlap", overlaps)):
        _write(folder / f"{name}.csv", rows)
    write_projections({name: models[name] for name in PROJECTED}, folder, plot)
    write_manifest(folder, config, documents, corpus)
    return summary
