"""Command line: `python -m bioir search [QUERY]`, `python -m bioir zipf`,
`python -m bioir embed (train | similar | compare | project)`, `python -m bioir vsm (search | evaluate)`,
`python -m bioir serve`,
`python -m bioir fetch (PMID_FILE | --query Q)`."""
import argparse
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from .corpus import load, save_jsonl, write_manifest
from .search import Index, Speller
from .zipf.stats import CONDITIONS

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data" / "glp1_1000" / "abstracts.jsonl"
VSM_CORPUS = Path(__file__).resolve().parents[1] / "data" / "glp1_mesh_3000" / "abstracts.jsonl"


def snippet(text: str, spans, width: int = 220) -> str:
    """A window of the abstract around the first query match."""
    start = spans[0][0] if spans else 0
    left = max(0, start - width // 3)
    excerpt = text[left:left + width]
    return ("…" if left else "") + excerpt + ("…" if left + width < len(text) else "")


def run_query(index: Index, speller: Speller, query: str, k: int) -> None:
    started = time.perf_counter()
    try:
        results = index.search(query, k)
    except ValueError as error:
        print(f"查詢錯誤：{error}")
        return
    elapsed = (time.perf_counter() - started) * 1000
    print(f"符合 {results.total} 篇，顯示前 {len(results.hits)} 篇（{elapsed:.1f} ms）")
    for rank, (score, doc) in enumerate(results.hits, 1):
        print(f"\n{rank:>2}. [{score:5.2f}] PMID {doc.pmid}  {doc.title}")
        print(textwrap.fill(snippet(doc.abstract, results.spans[doc.pmid]["abstract"]), 100,
                            initial_indent="    ", subsequent_indent="    "))
    suggestion = speller.suggest(query)
    if suggestion:
        print(f"\n你是不是要找：{suggestion}")


def search(args) -> int:
    started = time.perf_counter()
    index = Index(load(args.corpus))
    speller = Speller(index)
    print(f"已索引 {len(index)} 篇、{len(index.postings)} 個詞（{time.perf_counter() - started:.2f} s）")
    if args.query:
        run_query(index, speller, " ".join(args.query), args.k)
        return 0
    print('輸入查詢（AND / OR / NOT / 括號 / "片語"），空白行離開。')
    while query := input("\n> ").strip():
        run_query(index, speller, query, args.k)
    return 0


def _fit_cells(fit) -> str:
    """exponent, R² and RMSE columns; "n/a" where a fit or R² does not exist."""
    if not fit:
        return f"{'n/a':>8}  {'n/a':>5}  {'n/a':>5}"
    r2 = "n/a" if fit.r2 is None else f"{fit.r2:.3f}"
    return f"{fit.exponent:>8.3f}  {r2:>5}  {fit.rmse:>5.3f}"


def _number(value, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def zipf(args) -> int:
    from .zipf import analyze_corpus, compare_terms, write_csv
    if args.plot and not args.out:
        raise ValueError("--plot 需要搭配 --out 指定輸出資料夾")
    documents = load(args.corpus)
    options = {"compound": args.compound, "numbers": not args.no_numbers}
    results = analyze_corpus(documents, **options)
    base = results[args.baseline]
    tokenizer = "保留連字號複合詞" if args.compound else "字母／數字分開"
    print(f"Zipf 分析：{len(documents)} 篇摘要；斷詞：{tokenizer}；"
          f"{'排除' if args.no_numbers else '保留'}純數字；基準條件 {args.baseline}\n")

    print("     tokens  vocabulary   hapax  exponent     R²   RMSE")
    for condition, stats in results.items():
        print(f"{condition}  {stats.tokens:>9,}  {stats.vocabulary:>10,}  {stats.hapax:>6,}  "
              f"{_fit_cells(stats.fit)}   {CONDITIONS[condition]}")

    print(f"\n條件 {args.baseline} 分區擬合（前 10%／中間 80%／後 10% 的排名）：")
    print("region   ranks            terms  distinct cf  exponent     R²   RMSE")
    for region in base.regions:
        ranks = f"{region.rank_start:,}–{region.rank_end:,}"
        print(f"{region.name:<7}  {ranks:<15}  {region.terms:>5,}  {region.distinct_frequencies:>11,}  "
              f"{_fit_cells(region.fit)}")

    if args.top:
        print(f"\n各條件前 {args.top} 名（term  cf）：")
        print("".join(f"{condition:<24}" for condition in results))
        for i in range(args.top):
            cells = [f"{s.terms[i].term[:15]:<15}{s.terms[i].cf:>7,}  " if i < len(s.terms) else " " * 24
                     for s in results.values()]
            print("".join(cells))

    # CF / DF / IDF: the baseline's ten most frequent terms plus the words asked for.
    words = [t.term for t in base.terms[:10]] + [w.strip() for w in args.terms.split(",") if w.strip()]
    comparison = compare_terms(base, words, **options)
    print(f"\n條件 {args.baseline} 的 CF／DF／IDF（N = {len(documents):,}）：")
    print(f"{'word':<18}{'term':<16}{'rank':>7}{'cf':>8}{'df':>7}{'cf/df':>7}{'idf':>7}")
    for row in comparison:
        print(f"{row['word'][:17]:<18}{(row['term'] or '（停用詞）')[:15]:<16}{_number(row['rank'], ',')[:7]:>7}"
              f"{_number(row['cf'], ','):>8}{_number(row['df'], ','):>7}"
              f"{_number(row['cf_per_df'], '.2f'):>7}{_number(row['idf'], '.3f'):>7}")

    if args.out:
        paths = write_csv(results, args.out, comparison)
        if args.plot:
            from .zipf.plot import plot_zipf
            paths.append(plot_zipf(results, Path(args.out) / "zipf.png", args.baseline))
        print(f"\n已輸出 {len(paths)} 個檔案到 {args.out}")
    return 0


def _embed_module():
    try:
        from . import embed
    except ImportError as error:
        raise RuntimeError(f"詞向量需要 numpy：python -m pip install numpy（{error}）") from None
    return embed


def _report_sanity(model) -> None:
    check = model.metadata["sanity"]
    verdict = "通過" if check["passed"] else f"未通過：向量集中在同一方向，相似度不可信（上限 {check['limit']}）"
    print(f"健全性檢查：隨機詞對 cosine 平均 {check['random_pair_mean']:.3f}、"
          f"95 百分位 {check['random_pair_p95']:.3f} → {verdict}")


def embed_train(args) -> int:
    embed = _embed_module()
    config = embed.Config(architecture=args.arch, dimensions=args.dims, window=args.window, epochs=args.epochs,
                          min_count=args.min_count, seed=args.seed, compound=not args.split)
    documents = load(args.corpus)
    print(f"訓練 {args.arch}：{len(documents)} 篇摘要，{args.dims} 維，window {args.window}，{args.epochs} 輪")
    model = embed.train(documents, config,
                        progress=lambda epoch, epochs, loss: print(f"  epoch {epoch:>2}/{epochs}  loss {loss:.4f}",
                                                                    flush=True))
    path = model.save(args.output)
    print(f"詞彙 {len(model):,} 個，{model.metadata['seconds']} 秒；已存 {path}")
    _report_sanity(model)
    return 0 if model.metadata["sanity"]["passed"] else 2


def embed_similar(args) -> int:
    model = _embed_module().Embeddings.load(args.model)
    _report_sanity(model)
    for word in args.words:
        try:
            neighbors = model.similar(word, args.k, args.min_count)
        except (KeyError, ValueError) as error:
            print(f"\n{word}：{error.args[0]}")
            continue
        print(f"\n{word} → {model.term(word)}（出現 {model.counts[model.index[model.term(word)]]:,} 次）")
        for rank, (term, cosine, count) in enumerate(neighbors, 1):
            print(f"  {rank:>2}. {term:<24}{cosine:>7.3f}{count:>8,}")
    return 0


def embed_compare(args) -> int:
    embed = _embed_module()
    config = embed.Config(dimensions=args.dims, epochs=args.epochs, min_count=args.min_count)
    documents = load(args.corpus)
    print(f"比較實驗：CBOW、Skip-gram × (window 2, seed 42)、(5, 42)、(5, 7)；{args.epochs} 輪")
    summary = embed.compare(documents, args.out, config, progress=lambda name: print(f"  訓練 {name}…", flush=True),
                            plot=args.plot, corpus=args.corpus)
    print(f"\n{'model':<18}{'vocab':>7}{'loss':>8}{'秒':>6}  random-pair cosine（平均 / p95）")
    for row in summary:
        mark = "" if row["sanity_passed"] else "  ← 未通過健全性檢查"
        print(f"{row['model']:<18}{row['vocabulary']:>7,}{row['final_loss']:>8.3f}{row['seconds']:>6.0f}  "
              f"{row['random_pair_mean']:.3f} / {row['random_pair_p95']:.3f}{mark}")
    print(f"\n已輸出 summary、neighbors、pairs、overlap、projections 的 CSV，manifest.json"
          f"{'、projections.png' if args.plot else ''} 與 models/*.npz 到 {args.out}")
    return 0


def embed_project(args) -> int:
    embed = _embed_module()
    models = {Path(path).stem: embed.Embeddings.load(path) for path in args.models}
    paths = embed.write_projections(models, args.out, args.plot)
    if not paths:
        raise ValueError("模型詞彙中找不到至少三個預設投影詞（bioir/embed/projection.py 的 PROJECTION_GROUPS）")
    import csv
    with paths[1].open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            detail = (f"explained variance {float(row['explained_variance']):.0%}" if row["method"] == "PCA"
                      else f"KL {float(row['kl_divergence']):.3f}")
            print(f"{row['model']:<20}{row['method']:<7}{row['words']:>3} 詞  {detail:<26}"
                  f"trustworthiness {float(row['trustworthiness']):.3f}")
    print(f"已輸出 {', '.join(p.name for p in paths)} 到 {args.out}")
    return 0


def vsm_search(args) -> int:
    from .vsm import Scheme, VectorSpace
    scheme = Scheme.parse(args.scheme)
    started = time.perf_counter()
    space = VectorSpace(load(args.corpus))
    print(f"已建立 {len(space):,} 篇的向量空間（{len(space.postings):,} 個詞，{time.perf_counter() - started:.1f} 秒）")
    print(f"加權 {scheme.code}：{scheme.describe()}")
    query = " ".join(args.query)
    started = time.perf_counter()
    ranking = space.search(query, scheme, args.k)
    elapsed = (time.perf_counter() - started) * 1000
    vector = "  ".join(f"{t} {w:.3f}" for t, w in ranking.query_weights.items())
    print(f"查詢向量：{vector or '（空）'}" + (f"；語料中沒有：{', '.join(ranking.unknown)}" if ranking.unknown else ""))
    print(f"前 {len(ranking.hits)} 名（{elapsed:.1f} ms）")
    for hit in ranking.hits:
        doc = space.documents[hit.pmid]
        print(f"\n{hit.rank:>2}. {hit.score:.4f}  PMID {hit.pmid}  {doc.title}")
        if args.explain:
            parts = sorted(hit.contributions.items(), key=lambda item: -item[1])
            print("    " + "  ".join(f"{t} +{v:.4f}" for t, v in parts))
    return 0


def vsm_evaluate(args) -> int:
    from .vsm.experiment import SCHEMES, run
    schemes = [s.strip() for s in args.schemes.split(",")] if args.schemes else list(SCHEMES)
    documents = load(args.corpus)
    print(f"{len(documents):,} 篇、查詢檔 {args.queries.name}、{len(schemes)} 種加權")
    summary = run(documents, args.queries, args.out, schemes, plot=args.plot, corpus=args.corpus,
                  progress=lambda code: print(f"  {code}", flush=True))
    print(f"\n{'scheme':<9}{'P@10':>7}{'R@100':>7}{'MRR':>7}{'MAP':>7}{'nDCG@10':>9}")
    for row in sorted(summary, key=lambda r: -r["nDCG@10"]):
        print(f"{row['scheme']:<9}{row['P@10']:>7.3f}{row['R@100']:>7.3f}{row['MRR']:>7.3f}{row['MAP']:>7.3f}"
              f"{row['nDCG@10']:>9.3f}")
    print(f"\n已輸出到 {args.out}（summary、per_query、comparisons、judgments、idf、tf_example、length_bias、manifest"
          f"{'、metrics.png' if args.plot else ''}）")
    return 0


def serve(args) -> int:
    try:
        from .web import create_app
    except ImportError as error:
        raise RuntimeError(f"網頁介面需要 flask：python -m pip install flask（{error}）") from None
    app = create_app(args.corpus, args.results, args.vsm_corpus, args.vsm_results)
    print(f"網頁介面：http://{args.host}:{args.port}　（Ctrl+C 停止；這是開發用伺服器）", flush=True)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


def fetch(args) -> int:
    from . import pubmed
    if bool(args.pmid_file) == bool(args.query):
        raise ValueError("請給 PMID_FILE 或 --query 其中之一")
    info = {"source": pubmed.EUTILS,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.query:
        pmids, found = pubmed.esearch(args.query, args.email, args.n)
        info.update(query=args.query, query_translation=found["query_translation"],
                    sort="relevance", pubmed_total=found["total"])
    else:
        pmids = pubmed.parse_pmids(Path(args.pmid_file).read_text(encoding="utf-8-sig"))
        info.update(query=None, pmid_file=Path(args.pmid_file).name)
    documents, missing = pubmed.fetch(pmids, args.email)
    save_jsonl(documents, args.output)
    manifest = write_manifest(args.output, **info, requested=len(pmids), count=len(documents), missing=missing)
    print(f"已存 {len(documents)} 篇到 {args.output}（SHA-256 {manifest['sha256'][:12]}…）；"
          f"未取得 {len(missing)} 篇 {' '.join(missing)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bioir", description="Minimal biomedical abstract search.")
    commands = parser.add_subparsers(dest="command", required=True)

    s = commands.add_parser("search", help="search a local corpus (interactive without QUERY)")
    s.add_argument("query", nargs="*")
    s.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help=".jsonl, .csv or PubMed .xml")
    s.add_argument("-k", type=int, default=10, help="results to show")
    s.set_defaults(run=search)

    z = commands.add_parser("zipf", help="rank-frequency statistics under preprocessing conditions A-D")
    z.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help=".jsonl, .csv or PubMed .xml")
    z.add_argument("--top", type=int, default=10, help="most frequent terms to list per condition (0: none)")
    z.add_argument("--out", type=Path, help="folder for summary.csv and terms_<A-D>.csv")
    z.add_argument("--plot", action="store_true", help="also write zipf.png (needs matplotlib)")
    z.add_argument("--compound", action="store_true",
                   help="keep hyphenated compounds whole, like the original project's tokenizer")
    z.add_argument("--no-numbers", action="store_true", help="drop pure-number tokens (2, 0.05, 45%%)")
    z.add_argument("--baseline", choices=list(CONDITIONS), default="B",
                   help="condition for region fits, CF/DF/IDF table and plot panels 2-3")
    z.add_argument("--terms", default="", help="comma-separated words to add to the CF/DF/IDF table")
    z.set_defaults(run=zipf)

    e = commands.add_parser("embed", help="word embeddings: train, similar, compare (needs numpy)")
    embed_commands = e.add_subparsers(dest="embed_command", required=True)
    t = embed_commands.add_parser("train", help="train CBOW or Skip-gram on a corpus")
    t.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    t.add_argument("--arch", choices=["cbow", "skipgram"], default="cbow")
    t.add_argument("--dims", type=int, default=100)
    t.add_argument("--window", type=int, default=5)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--min-count", type=int, default=3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--split", action="store_true", help="split letters and digits like the search index")
    t.add_argument("-o", "--output", default="embeddings.npz")
    t.set_defaults(run=embed_train)
    q = embed_commands.add_parser("similar", help="nearest words by cosine")
    q.add_argument("model", type=Path)
    q.add_argument("words", nargs="+")
    q.add_argument("-k", type=int, default=10)
    q.add_argument("--min-count", type=int, default=10, help="hide neighbours seen fewer times (rare words are noisy)")
    q.set_defaults(run=embed_similar)
    c = embed_commands.add_parser("compare", help="CBOW vs Skip-gram, window 2 vs 5, seed 42 vs 7")
    c.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    c.add_argument("--dims", type=int, default=100)
    c.add_argument("--epochs", type=int, default=20)
    c.add_argument("--min-count", type=int, default=3)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--plot", action="store_true", help="also draw projections.png (needs matplotlib)")
    c.set_defaults(run=embed_compare)
    j = embed_commands.add_parser("project", help="PCA and t-SNE of the predeclared word groups")
    j.add_argument("models", nargs="+", type=Path, help="one or more .npz models")
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--plot", action="store_true", help="also draw projections.png (needs matplotlib)")
    j.set_defaults(run=embed_project)

    v = commands.add_parser("vsm", help="vector-space model with SMART term weighting")
    vsm_commands = v.add_subparsers(dest="vsm_command", required=True)
    vs = vsm_commands.add_parser("search", help="rank documents by cosine (or inner product)")
    vs.add_argument("query", nargs="+")
    vs.add_argument("--corpus", type=Path, default=VSM_CORPUS)
    vs.add_argument("--scheme", default="lnc.ltc", help="SMART ddd.qqq, e.g. lnc.ltc, ltc.ltc, lnu.ltc")
    vs.add_argument("-k", type=int, default=10)
    vs.add_argument("--explain", action="store_true", help="show each query term's share of the score")
    vs.set_defaults(run=vsm_search)
    ve = vsm_commands.add_parser("evaluate", help="controlled weighting experiment with MeSH relevance")
    ve.add_argument("--corpus", type=Path, default=VSM_CORPUS)
    ve.add_argument("--queries", type=Path, default=VSM_CORPUS.with_name("queries.json"))
    ve.add_argument("--schemes", help="comma-separated subset; default: every scheme in bioir/vsm/experiment.py")
    ve.add_argument("--out", type=Path, required=True)
    ve.add_argument("--plot", action="store_true", help="also draw metrics.png (needs matplotlib)")
    ve.set_defaults(run=vsm_evaluate)

    w = commands.add_parser("serve", help="read-only web interface (needs flask)")
    w.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    w.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "embed_glp1",
                   help="folder written by `embed compare` (models/, projections.png)")
    w.add_argument("--vsm-corpus", type=Path, default=VSM_CORPUS, help="corpus with MeSH for the vector-space page")
    w.add_argument("--vsm-results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "vsm_glp1",
                   help="folder written by `vsm evaluate`")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=5000)
    w.set_defaults(run=serve)

    f = commands.add_parser("fetch", help="download PubMed abstracts to JSONL (+ .manifest.json)")
    f.add_argument("pmid_file", nargs="?", help="PMIDs or PubMed URLs, separated by spaces/commas/lines")
    f.add_argument("--query", help="PubMed query; fetch its most relevant PMIDs instead of a file")
    f.add_argument("-n", type=int, default=1000, help="how many PMIDs --query takes")
    f.add_argument("--email", required=True, help="contact address NCBI requires")
    f.add_argument("-o", "--output", default="abstracts.jsonl")
    f.set_defaults(run=fetch)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"錯誤：{error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
