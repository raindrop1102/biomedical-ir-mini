import csv
import importlib.util
import io
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from bioir import Document
from bioir.__main__ import main
from bioir.text import analyze
from bioir.zipf import analyze_corpus, compare_terms, preprocess, write_csv
from bioir.zipf.stats import Term, analyze_condition, fit_zipf, is_number, regions

DOCS = [
    Document("1", "", "The GLP-1RAs reduced weight. The patients' weight fell."),
    Document("2", "", "Patients with obesity used GLP1 analogues and lost weight."),
]


class PreprocessTest(unittest.TestCase):
    def test_conditions_are_cumulative(self):
        text = "The GLP-1RAs reduced weight."
        self.assertEqual(preprocess(text, "A"), ["the", "glp-1ras", "reduced", "weight."])
        self.assertEqual(preprocess(text, "B"), ["the", "glp", "1", "ras", "reduced", "weight"])
        self.assertEqual(preprocess(text, "C"), ["glp", "1", "ras", "reduced", "weight"])
        self.assertEqual(preprocess(text, "D"), ["glp", "1", "ra", "reduc", "weight"])

    def test_compound_tokenizer_keeps_hyphenated_words(self):
        self.assertEqual(preprocess("The GLP-1RAs work", "B", compound=True), ["the", "glp-1ras", "work"])

    def test_condition_d_is_what_search_indexes(self):
        for doc in DOCS:
            self.assertEqual(preprocess(doc.abstract, "D"), [t for t, _ in analyze(doc.abstract)])

    def test_numbers_can_be_dropped_in_every_condition(self):
        text = "In 2 trials (n=45), 0.05 of 10,000 GLP-1 users lost 5% weight (2)."
        self.assertEqual(preprocess(text, "A", numbers=False),
                         ["in", "trials", "(n=45),", "of", "glp-1", "users", "lost", "weight"])
        self.assertEqual(preprocess(text, "C", numbers=False), ["trials", "n", "glp", "users", "lost", "weight"])
        self.assertEqual(preprocess(text, "B", compound=True, numbers=False),
                         ["in", "trials", "n", "of", "glp-1", "users", "lost", "weight"])
        self.assertTrue(all(map(is_number, ["2", "0.05", "10,000", "45%", "(2).", "3"])))
        self.assertFalse(any(map(is_number, ["glp-1", "p53", "n=45", "covid-19", "."])))

    def test_unknown_condition(self):
        with self.assertRaises(ValueError):
            preprocess("x", "E")


class StatsTest(unittest.TestCase):
    def test_perfect_zipf_is_fitted_exactly(self):
        terms = [Term(r, f"t{r}", round(1000 / r), 1, 0.0) for r in (1, 2, 4, 5, 10)]
        fit = fit_zipf(terms)
        self.assertAlmostEqual(fit.exponent, 1.0, places=6)
        self.assertAlmostEqual(fit.r2, 1.0, places=9)
        self.assertAlmostEqual(fit.rmse, 0.0, places=6)
        self.assertIsNone(fit_zipf(terms[:1]))

    def test_equal_frequencies_have_no_r2(self):
        self.assertIsNone(fit_zipf([Term(1, "a", 3, 1, 0.0), Term(2, "b", 3, 1, 0.0)]).r2)

    def test_regions_are_10_80_10_percent_of_ranks(self):
        terms = [Term(r, f"t{r}", max(1, 100 // r), 1, 0.0) for r in range(1, 21)]
        head, middle, tail = regions(terms)
        self.assertEqual([(r.name, r.rank_start, r.rank_end, r.terms) for r in (head, middle, tail)],
                         [("head", 1, 2, 2), ("middle", 3, 18, 16), ("tail", 19, 20, 2)])
        # The tail is a plateau of cf = 1: one distinct frequency, slope 0, no R².
        self.assertEqual(tail.distinct_frequencies, 1)
        self.assertEqual((tail.fit.exponent, tail.fit.r2), (0.0, None))
        self.assertEqual(str(tail.fit.exponent), "0.0")          # never "-0.0"
        self.assertEqual([r.name for r in regions(terms[:1])], ["head"])
        self.assertEqual(regions([]), [])

    def test_counts_ranks_and_idf(self):
        stats = analyze_condition(DOCS, "C")
        self.assertEqual(stats.tokens, sum(t.cf for t in stats.terms))
        self.assertEqual(stats.vocabulary, len(stats.terms))
        weight = next(t for t in stats.terms if t.term == "weight")
        self.assertEqual((weight.rank, weight.cf, weight.df), (1, 3, 2))
        self.assertEqual(weight.idf, 0.0)
        reduced = next(t for t in stats.terms if t.term == "reduced")
        self.assertAlmostEqual(reduced.idf, math.log10(2))
        # Ties in cf are ranked alphabetically.
        tied = [t.term for t in stats.terms if t.cf == stats.terms[1].cf]
        self.assertEqual(tied, sorted(tied))
        self.assertEqual(stats.hapax, sum(t.cf == 1 for t in stats.terms))

    def test_compare_terms_uses_the_conditions_preprocessing(self):
        stats = analyze_condition(DOCS, "D")
        rows = {(r["word"], r["term"]): r for r in compare_terms(stats, ["Patients", "GLP-1", "the", "absent"])}
        patient = rows[("Patients", "patient")]
        self.assertEqual((patient["cf"], patient["df"], patient["cf_per_df"]), (2, 2, 1.0))
        self.assertIn(("GLP-1", "glp"), rows)
        self.assertIn(("GLP-1", "1"), rows)
        self.assertIsNone(rows[("the", None)]["cf"])          # a stop word yields no term under D
        self.assertIsNone(rows[("absent", "absent")]["df"])

    def test_stemming_never_loses_tokens(self):
        results = analyze_corpus(DOCS)
        self.assertEqual(results["C"].tokens, results["D"].tokens)
        self.assertLessEqual(results["D"].vocabulary, results["C"].vocabulary)


class OutputTest(unittest.TestCase):
    def test_csv_files(self):
        with tempfile.TemporaryDirectory() as folder:
            results = analyze_corpus(DOCS)
            paths = write_csv(results, folder, compare_terms(results["B"], ["weight"]))
            self.assertEqual([p.name for p in paths], ["summary.csv", "regions.csv", "terms_A.csv", "terms_B.csv",
                                                       "terms_C.csv", "terms_D.csv", "compare_terms.csv"])
            with paths[1].open(encoding="utf-8-sig") as file:
                self.assertEqual([(r["condition"], r["region"]) for r in csv.DictReader(file)][:3],
                                 [("A", "head"), ("A", "middle"), ("A", "tail")])
            with paths[2].open(encoding="utf-8-sig") as file:
                self.assertIn("cf_per_df", next(csv.DictReader(file)))
            with paths[0].open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["condition"] for r in rows], list("ABCD"))
            self.assertIn("exponent", rows[0])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder, "c.jsonl")
            corpus.write_text('{"pmid": "1", "abstract": "Weight loss and weight gain."}\n', encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["zipf", "--corpus", str(corpus), "--top", "2", "--out", str(Path(folder, "out")),
                             "--no-numbers", "--baseline", "C", "--terms", "Loss,gain"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("排除純數字", text)
            self.assertIn("條件 C 分區擬合", text)
            self.assertIn("gain", text)
            self.assertTrue(Path(folder, "out", "summary.csv").exists())
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["zipf", "--corpus", str(corpus), "--plot"]), 1)   # --plot needs --out

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot(self):
        from bioir.zipf.plot import plot_zipf
        with tempfile.TemporaryDirectory() as folder:
            path = plot_zipf(analyze_corpus(DOCS), Path(folder, "z.png"), baseline="C")
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
