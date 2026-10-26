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
DEFAULT_VSM_CORPUS = ROOT / "data" / "glp1_mesh_3000" / "abstracts.jsonl"
DEFAULT_VSM_RESULTS = ROOT / "results" / "vsm_glp1"
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


def _load_vsm(corpus: Path, results: Path):
    """The vector space over the MeSH corpus, its evaluation queries with judgments, and the
    experiment's saved results — or an explanation of what is missing."""
    if not corpus.exists():
        return None, f"找不到向量空間語料 {corpus}"
    from ..vsm import VectorSpace
    from ..vsm.evaluate import judgments, load_queries
    documents = load(corpus)
    space = VectorSpace(documents)
    queries_file = corpus.with_name("queries.json")
    queries = load_queries(queries_file) if queries_file.exists() else []
    return {"space": space, "corpus": corpus.name, "queries": queries,
            "judgments": {q.id: judgments(documents, q) for q in queries},
            "summary": _read_csv(results / "summary.csv"), "comparisons": _read_csv(results / "comparisons.csv"),
            "chart": results / "metrics.png"}, None


# The homework's research questions: each with its one-line conclusion and the controlled
# comparisons (baseline, variant) that support it, looked up in comparisons.csv.
RQ_CARDS = [
    ("RQ1", "原始 TF 還是 log TF？", "log TF 壓縮重複出現的詞：MAP 上升，但前 10 名的 nDCG@10 沒有變好。",
     [("nnc.nnc", "lnc.lnc")]),
    ("RQ2", "IDF 乘一次還是兩次？", "IDF 只乘在查詢那邊最好；兩邊都乘（IDF²）讓罕見詞過重，顯著變差。",
     [("lnc.ltc", "ltc.ltc")]),
    ("RQ3", "TF-IDF 比只用 TF 或只用 IDF 好嗎？", "TF-IDF 顯著優於只用 IDF；和只用 TF 的差距不顯著。",
     [("bnc.btc", "lnc.ltc"), ("lnc.lnc", "lnc.ltc")]),
    ("RQ4", "長度正規化有幫助嗎？", "需要正規化，但 cosine 過度矯正、偏好短文件；pivoted unique 顯著改善 MAP。",
     [("nnn.nnn", "nnc.nnc"), ("ltn.ltn", "ltc.ltc"), ("lnc.ltc", "lnu.ltc")]),
]
CARD_METRICS = ["nDCG@10", "MAP"]


def _rq_cards(vsm) -> list[dict]:
    rows = {}
    for row in vsm["comparisons"]:
        rows.setdefault((row["baseline"], row["variant"], row["metric"]), row)
    cards = []
    for rq, question, conclusion, pairs in RQ_CARDS:
        found = [{"baseline": b, "variant": v, "metrics": [rows[b, v, m] for m in CARD_METRICS if (b, v, m) in rows]}
                 for b, v in pairs]
        cards.append({"rq": rq, "question": question, "conclusion": conclusion,
                      "pairs": [p for p in found if p["metrics"]]})
    return cards


def _best_by_metric(vsm) -> list[tuple[str, str, float]]:
    """RQ5: which scheme each metric ranks first."""
    summary = vsm["summary"]
    if not summary:
        return []
    best = []
    for metric in ["nDCG@10", "MAP", "P@10", "MRR"]:
        top = max(summary, key=lambda row: float(row[metric]))
        best.append((metric, top["scheme"], float(top[metric])))
    return best


def _explain_side(side, space) -> dict:
    """Each SMART letter of one side as the rule it applied, with this text's own inputs."""
    from ..vsm.weighting import BYTE_SIZE_ALPHA, PIVOT_SLOPE
    w = side.weighting
    tf = {"n": "tf 原值",
          "l": "1 + log tf",
          "a": f"0.5 + 0.5 × tf ÷ 最大 tf（{side.max_tf}）",
          "b": "有出現記 1，沒出現記 0",
          "L": f"(1 + log tf) ÷ (1 + log 平均 tf)，平均 tf = {side.average_tf:.2f}"}[w.tf]
    idf = {"n": "不乘 IDF（記 1）",
           "t": f"log(N ÷ df)，N = {len(space):,}",
           "p": f"max(0, log((N − df) ÷ df))，N = {len(space):,}"}[w.idf]
    norm = {"n": "不正規化（除以 1）",
            "c": f"除以向量長度 √(Σ 權重²) = {side.divisor:.3f}（計入全部 {side.distinct} 個詞，不只查詢詞）",
            "u": (f"除以 {1 - PIVOT_SLOPE:g} × 平均相異詞數 {space.pivot:.1f} + {PIVOT_SLOPE:g} × 相異詞數 {side.distinct}"
                  f" = {side.divisor:.3f}"),
            "b": f"除以 字元數 {side.characters:,}^{BYTE_SIZE_ALPHA:g} = {side.divisor:.3f}"}[w.norm]
    return {"side": side, "rules": [(w.tf, "TF", tf), (w.idf, "IDF", idf), (w.norm, "正規化", norm)]}


def create_app(corpus=DEFAULT_CORPUS, results=DEFAULT_RESULTS, vsm_corpus=DEFAULT_VSM_CORPUS,
               vsm_results=DEFAULT_VSM_RESULTS) -> Flask:
    corpus, results = Path(corpus), Path(results)
    app = Flask(__name__)
    started = time.perf_counter()
    documents = load(corpus)
    index = Index(documents)
    speller = Speller(index)
    embeddings, embed_error = _load_embeddings(results)
    vsm, vsm_error = _load_vsm(Path(vsm_corpus), Path(vsm_results))
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

    def vsm_context(default_doc: bool) -> tuple[dict, int]:
        """Both schemes' rankings, metrics and, for one document, weighting steps for the
        request's query; default_doc picks scheme A's first hit when no document is chosen."""
        from ..vsm import Scheme, Weighting
        from ..vsm.evaluate import score
        from ..vsm.experiment import SCHEMES
        query = request.args.get("query", "").strip()[:300]
        codes = [request.args.get("a", "lnc.ltc"), request.args.get("b", "ltc.ltc")]
        context = {"error": vsm_error, "vsm": vsm, "query": query, "schemes": SCHEMES, "codes": codes,
                   "columns": [], "evaluation": None, "step_doc": None}
        if vsm_error or not query:
            return context, 200
        space = vsm["space"]
        evaluation = next((q for q in vsm["queries"] if q.text.lower() == query.lower()), None)
        grades = vsm["judgments"][evaluation.id] if evaluation else {}
        for code in dict.fromkeys(codes):
            try:
                scheme = Scheme.parse(code)
            except ValueError as error:
                context["error"] = str(error)
                return context, 400
            ranking = space.search(query, scheme)
            hits = [{"hit": h, "doc": space.documents[h.pmid], "length": space.lengths[h.pmid],
                     "grade": grades.get(h.pmid, 0),
                     "parts": sorted(h.contributions.items(), key=lambda item: -item[1])} for h in ranking.hits[:10]]
            metrics = score([h.pmid for h in ranking.hits], grades).values if evaluation else None
            context["columns"].append({"scheme": scheme, "hits": hits, "metrics": metrics, "total": len(ranking.hits),
                                       "vector": ranking.query_weights, "unknown": ranking.unknown})
        terms = list(dict.fromkeys(t for c in context["columns"] for t in c["vector"]))
        top = context["columns"][0]["hits"]
        pmid = request.args.get("doc", "")              # the article shown under scheme A
        pmid_b = request.args.get("doc_b", "")          # under scheme B; defaults to A's
        pmid = pmid if pmid in space.tf else ""
        pmid_b = pmid_b if pmid_b in space.tf else ""
        if not pmid:
            pmid = pmid_b or (top[0]["hit"].pmid if top and default_doc else "")
        for i, column in enumerate(context["columns"]):
            p = (pmid_b or pmid) if i else pmid
            column["docs"] = []
            if not p:
                continue
            ranks = {h["hit"].pmid: h["hit"].rank for h in column["hits"]}
            query_side, document_side = space.breakdown(query, column["scheme"], p)
            products = [(q.term, q.weight, d.weight, q.weight * d.weight)
                        for q, d in zip(query_side.terms, document_side.terms)]
            entry = {"pmid": p, "doc": space.documents[p], "grade": grades.get(p, 0), "rank": ranks.get(p),
                     "products": products, "score": sum(item[3] for item in products),
                     "query": _explain_side(query_side, space), "document": _explain_side(document_side, space)}
            column["docs"].append(entry)
            column.update(steps=[entry["query"], entry["document"]], products=products, steps_score=entry["score"])
        context.update(evaluation=evaluation, relevant=len(grades), terms=terms,
                       df={t: space.df(t) for t in terms}, step_doc=space.documents.get(pmid),
                       step_grade=grades.get(pmid, 0), doc_a=pmid, doc_b=pmid_b)
        # DF and IDF of every term in the weighting steps, whatever IDF letter each scheme uses
        step_terms = {t.term for c in context["columns"] for e in c["docs"] for t in e["document"]["side"].terms}
        idf = Weighting("n", "t", "n")
        context["df"].update({t: space.df(t) for t in step_terms})
        context["idf"] = {t: idf.idf_weight(space.df(t), len(space)) for t in context["df"]}
        return context, 200

    @app.get("/vsm")
    def vector_space():
        context, status = vsm_context(default_doc=True)
        return render_template("vsm.html", **context), status

    @app.get("/hw3")
    def hw3():
        context, status = vsm_context(default_doc=False)
        context["examples"] = vsm["queries"] if vsm else []
        return render_template("hw3.html", **context), status

    @app.get("/hw3/results")
    def hw3_results():
        return render_template("hw3_results.html", vsm=vsm, error=vsm_error,
                               cards=_rq_cards(vsm) if vsm else [], best=_best_by_metric(vsm) if vsm else [])

    @app.get("/vsm/metrics.png")
    def vsm_chart():
        if not vsm or not vsm["chart"].exists():
            abort(404)
        return send_file(vsm["chart"], mimetype="image/png", max_age=3600)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "documents": site["documents"], "embeddings": bool(embeddings)}

    return app
