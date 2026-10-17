"""Flask front end: search, Zipf and embeddings over one corpus loaded at start-up.

Everything is built once in create_app() and only read afterwards, so requests
never change shared state and need no locking (except matplotlib, which is not
thread-safe). Training and downloads stay on the command line.
"""
import csv
import io
import threading
import time
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file, url_for
from markupsafe import Markup

from ..corpus import load
from ..search import Index, Speller
from ..text import highlight
from ..zipf import CONDITIONS, analyze_corpus, compare_terms
from ..zipf.stats import REGION_FIELDS, SUMMARY_FIELDS, TERM_FIELDS, region_row, summary_row

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "glp1_1000" / "abstracts.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "embed_glp1"
EMBED_MODELS = {"cbow_w5_s42": "CBOW", "skipgram_w5_s42": "Skip-gram"}
PAGE_SIZES = (10, 20, 50)
SNIPPET_CHARS, SNIPPET_LEAD = 260, 40
EXAMPLES = ["GLP-1", '"type 2 diabetes" AND (semaglutide OR tirzepatide) NOT mice', "SGLT2", "semaglutde"]


def snippet(text: str, spans) -> tuple[Markup, bool]:
    """A highlighted window of the abstract that opens just before the first match.
    Returns (HTML, whether the whole text fits)."""
    if len(text) <= SNIPPET_CHARS:
        return Markup(highlight(text, spans)), True
    first = spans[0][0] if spans else 0
    start = max(0, first - SNIPPET_LEAD)
    if start:                                       # open at a word boundary
        space = text.find(" ", start, first)
        start = space + 1 if space != -1 else first
    end = min(len(text), start + SNIPPET_CHARS)
    if end < len(text):                             # and close at one
        space = text.rfind(" ", start + SNIPPET_CHARS // 2, end)
        end = space if space != -1 else end
    inside = [(max(s, start) - start, min(e, end) - start) for s, e in spans if e > start and s < end]
    html = highlight(text[start:end], inside)
    return Markup(("…" if start else "") + html + ("…" if end < len(text) else "")), False


def pages(total: int, page: int, size: int, link) -> dict:
    """What the pager macro needs: the page, its neighbours and the first/last page."""
    count = max(1, -(-total // size))
    page = min(max(page, 1), count)
    shown = sorted({1, count, *range(max(1, page - 2), min(count, page + 2) + 1)})
    return {"page": page, "pages": count, "total": total, "start": (page - 1) * size + 1 if total else 0,
            "end": min(page * size, total), "offset": (page - 1) * size,
            "links": [(n, link(n)) for n in shown],
            "previous": link(page - 1) if page > 1 else None, "next": link(page + 1) if page < count else None}


def _int(name: str, default: int, choices=None) -> int:
    try:
        value = int(request.args.get(name, default))
    except ValueError:
        return default
    return value if choices is None or value in choices else default


def _csv(fieldnames, rows, filename):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return send_file(io.BytesIO(buffer.getvalue().encode("utf-8-sig")), mimetype="text/csv",
                     download_name=filename, as_attachment=True, max_age=0)


def _load_embeddings(folder: Path):
    """The two window-5 models of an `embed compare` run, or an explanation of why there are none."""
    try:
        from ..embed import Embeddings
    except ImportError:
        return {}, "詞向量需要 numpy：python -m pip install numpy"
    missing = [n for n in EMBED_MODELS if not (folder / "models" / f"{n}.npz").exists()]
    if missing:
        return {}, (f"找不到 {folder / 'models'} 中的 {', '.join(missing)}.npz；"
                    f"請先執行 python -m bioir embed compare --out {folder} --plot")
    return {label: Embeddings.load(folder / "models" / f"{n}.npz") for n, label in EMBED_MODELS.items()}, None


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def create_app(corpus=DEFAULT_CORPUS, results=DEFAULT_RESULTS) -> Flask:
    corpus, results = Path(corpus), Path(results)
    app = Flask(__name__)
    started = time.perf_counter()
    documents = load(corpus)
    index = Index(documents)
    speller = Speller(index)
    embeddings, embed_error = _load_embeddings(results)
    site = {"corpus": corpus.name, "documents": len(documents), "vocabulary": len(index.postings),
            "seconds": time.perf_counter() - started}
    plot_lock = threading.Lock()

    @lru_cache(maxsize=8)
    def zipf_results(compound: bool, numbers: bool):
        return analyze_corpus(documents, compound=compound, numbers=numbers)

    @lru_cache(maxsize=8)
    def zipf_png(compound: bool, numbers: bool, baseline: str) -> bytes | None:
        try:
            from ..zipf.plot import plot_zipf
            buffer = io.BytesIO()
            with plot_lock:
                plot_zipf(zipf_results(compound, numbers), buffer, baseline)
            return buffer.getvalue()
        except RuntimeError:                       # matplotlib is not installed
            return None

    zipf_results(False, True)                      # warm the default page

    @app.context_processor
    def common():
        return {"site": site, "conditions": CONDITIONS}

    @app.after_request
    def headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.get("/")
    def search():
        query = request.args.get("query", "").strip()
        size = _int("per_page", 20, PAGE_SIZES)
        page = max(_int("page", 1), 1)
        context = {"query": query, "per_page": size, "page_sizes": PAGE_SIZES, "examples": EXAMPLES,
                   "results": None, "error": None, "suggestion": None}
        if not query:
            return render_template("search.html", **context)
        started = time.perf_counter()
        try:
            found = index.search(query, k=page * size)
        except ValueError as error:
            context["error"] = str(error)
            return render_template("search.html", **context), 400
        elapsed = (time.perf_counter() - started) * 1000
        pager = pages(found.total, page, size,
                      lambda n: url_for("search", query=query, page=n, per_page=size if size != 20 else None))
        cards = []
        for rank, (score, doc) in enumerate(found.hits[pager["offset"]:pager["offset"] + size], pager["start"]):
            spans = found.spans[doc.pmid]
            text, whole = snippet(doc.abstract, spans["abstract"])
            cards.append({"rank": rank, "score": score, "doc": doc, "snippet": text, "whole": whole,
                          "title": Markup(highlight(doc.title or "（無標題）", spans["title"])),
                          "abstract": Markup(highlight(doc.abstract, spans["abstract"]))})
        context.update(results=cards, total=found.total, terms=sorted(found.terms), elapsed=elapsed,
                       pagination=pager, suggestion=speller.suggest(query))
        return render_template("search.html", **context)

    def zipf_options() -> dict:
        baseline = request.args.get("baseline", "B")
        return {"compound": request.args.get("compound") == "1", "numbers": request.args.get("no_numbers") != "1",
                "baseline": baseline if baseline in CONDITIONS else "B"}

    @app.get("/zipf")
    def zipf():
        options = zipf_options()
        results_ = zipf_results(options["compound"], options["numbers"])
        base = results_[options["baseline"]]
        words = [t.term for t in base.terms[:10]]
        extra = [w.strip() for w in request.args.get("terms", "").split(",") if w.strip()][:20]
        comparison = compare_terms(base, words + extra, compound=options["compound"], numbers=options["numbers"])
        link = {k: v for k, v in (("baseline", options["baseline"] if options["baseline"] != "B" else None),
                                  ("compound", "1" if options["compound"] else None),
                                  ("no_numbers", None if options["numbers"] else "1")) if v}
        return render_template("zipf.html", results=results_, base=base, options=options, terms=", ".join(extra),
                               comparison=comparison, link=link)

    @app.get("/zipf.png")
    def zipf_image():
        options = zipf_options()
        png = zipf_png(options["compound"], options["numbers"], options["baseline"])
        if png is None:
            abort(404)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=3600)

    @app.get("/zipf/<name>.csv")
    def zipf_csv(name):
        options = zipf_options()
        results_ = zipf_results(options["compound"], options["numbers"])
        if name == "summary":
            return _csv(SUMMARY_FIELDS, map(summary_row, results_.values()), "summary.csv")
        if name == "regions":
            rows = (region_row(c, r) for c, stats in results_.items() for r in stats.regions)
            return _csv(REGION_FIELDS, rows, "regions.csv")
        if name in (f"terms_{c}" for c in CONDITIONS):
            stats = results_[name[-1]]
            return _csv(TERM_FIELDS, ({"rank": t.rank, "term": t.term, "cf": t.cf, "df": t.df,
                                       "cf_per_df": t.cf_per_df, "idf": t.idf} for t in stats.terms), f"{name}.csv")
        abort(404)

    @app.get("/embedding")
    def embedding():
        word = request.args.get("word", "").strip()[:60]
        left, right = (request.args.get(k, "").strip()[:60] for k in ("a", "b"))
        min_count = _int("min_count", 10, (1, 5, 10, 20))
        neighbors, pairs = {}, {}
        for label, model in embeddings.items():
            if word:
                try:
                    neighbors[label] = {"term": model.term(word), "rows": model.similar(word, 10, min_count)}
                except (KeyError, ValueError) as error:
                    neighbors[label] = {"error": error.args[0]}
            if left and right:
                try:
                    cosine = model.similarity(left, right)
                    check = model.metadata["sanity"]
                    pairs[label] = {"cosine": cosine,
                                    "z": (cosine - check["random_pair_mean"]) / check["random_pair_std"]}
                except (KeyError, ValueError) as error:
                    pairs[label] = {"error": error.args[0]}
        return render_template("embedding.html", models=embeddings, error=embed_error, word=word, a=left, b=right,
                               min_count=min_count, neighbors=neighbors, pairs=pairs,
                               summary=_read_csv(results / "summary.csv"),
                               quality=_read_csv(results / "projection_quality.csv"),
                               has_projection=(results / "projections.png").exists())

    @app.get("/embedding/projections.png")
    def projection_image():
        path = results / "projections.png"
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="image/png", max_age=3600)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "documents": site["documents"], "embeddings": bool(embeddings)}

    return app
