import json
import math
import tempfile
import unittest
from pathlib import Path

from bioir import Document, load
from bioir.corpus import parse_pubmed_xml, save_jsonl
from bioir.vsm import Scheme, VectorSpace, Weighting
from bioir.vsm.evaluate import Query, judgments, load_queries, paired_permutation, score
from bioir.vsm.experiment import run

DOCS = [
    Document("1", "Semaglutide and weight", "Semaglutide semaglutide semaglutide reduced weight.",
             ("Semaglutide", "Weight Loss"), ("Semaglutide",)),
    Document("2", "Weight loss", "Weight loss after surgery. Weight regain was rare.", ("Weight Loss",)),
    Document("3", "Insulin", "Insulin secretion from beta cells.", ("Insulin",), ("Insulin",)),
    Document("4", "Semaglutide in mice", "Semaglutide changed insulin secretion in mice.", ("Semaglutide", "Insulin")),
]


class WeightingTest(unittest.TestCase):
    def test_tf_idf_and_norm_letters(self):
        w = Weighting("l", "t", "c")
        self.assertEqual(w.code, "ltc")
        self.assertAlmostEqual(w.tf_weight(3, 3), 1 + math.log10(3))
        self.assertAlmostEqual(w.idf_weight(2, 8), math.log10(4))
        self.assertEqual(Weighting("n", "n", "n").tf_weight(3, 5), 3.0)
        self.assertEqual(Weighting("b", "n", "n").tf_weight(7, 7), 1.0)
        self.assertAlmostEqual(Weighting("a", "n", "n").tf_weight(2, 4), 0.75)
        self.assertEqual(Weighting("n", "n", "n").tf_weight(0, 4), 0.0)
        self.assertAlmostEqual(Weighting("n", "p", "n").idf_weight(2, 8), math.log10(3))
        self.assertEqual(Weighting("n", "p", "n").idf_weight(5, 8), 0.0)     # df above N/2

    def test_log_average_tf_and_length_divisors(self):
        L = Weighting("L", "n", "n")
        self.assertAlmostEqual(L.tf_weight(4, 4, average_tf=2), (1 + math.log10(4)) / (1 + math.log10(2)))
        self.assertEqual(L.tf_weight(2, 4, average_tf=2), 1.0)          # a term at the text's average
        self.assertAlmostEqual(Weighting("l", "n", "c").divisor([3, 4], 2, 99, 50), 5.0)
        self.assertAlmostEqual(Weighting("l", "n", "u").divisor([3, 4], 100, 99, 50), 0.8 * 50 + 0.2 * 100)
        self.assertAlmostEqual(Weighting("l", "n", "b").divisor([3, 4], 2, 256, 50), 256 ** 0.375)
        self.assertEqual(Weighting("l", "n", "n").divisor([3, 4], 2, 99, 50), 1.0)

    def test_lecture_example_ltn_lnc(self):
        """Lecture 3, "tf-idf example: ltn.lnc" — query "best car insurance", document
        "car insurance auto insurance", N = 1,000,000; logs are base 10 like the lecture's. The lecture writes
        query.document; here that is the document scheme lnc with the query scheme ltn."""
        N = 1_000_000
        df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
        query_tf = {"best": 1, "car": 1, "insurance": 1}
        document_tf = {"auto": 1, "car": 1, "insurance": 2}
        query, document = Weighting("l", "t", "n"), Weighting("l", "n", "c")

        q = {t: query.tf_weight(tf, 1) * query.idf_weight(df[t], N) for t, tf in query_tf.items()}
        self.assertEqual([round(q[t], 1) for t in ("best", "car", "insurance")], [1.3, 2.0, 3.0])

        d = {t: document.tf_weight(tf, 2) * document.idf_weight(df[t], N) for t, tf in document_tf.items()}
        length = document.divisor(d.values(), 3, 0, 0)
        self.assertAlmostEqual(length, 1.92, places=2)                  # √(1² + 1² + 1.3²)
        d = {t: w / length for t, w in d.items()}
        self.assertEqual([round(d[t], 2) for t in ("auto", "car", "insurance")], [0.52, 0.52, 0.68])

        self.assertAlmostEqual(sum(q[t] * d.get(t, 0) for t in q), 3.08, delta=0.01)
        # The lecture's 3.08 adds the rounded products 1.04 + 2.04; unrounded it is 3.07.
        self.assertAlmostEqual(sum(round(q[t], 1) * round(d.get(t, 0), 2) for t in q), 3.08)

    def test_idf_log_base_does_not_change_cosine(self):
        """Lecture 3: "Why is the base of the log in idf immaterial?" — it rescales every IDF
        by one constant, which L2 normalization cancels."""
        space = VectorSpace(DOCS)
        rankings = [[(h.pmid, round(h.score, 12)) for h in space.search("semaglutide insulin weight",
                                                                        Scheme(Weighting("n", "t", "c", base),
                                                                               Weighting("n", "t", "c", base))).hits]
                    for base in (math.e, 10, 2)]
        self.assertEqual(rankings[0], rankings[1])
        self.assertEqual(rankings[0], rankings[2])

    def test_cosine_cancels_log_average_tf(self):
        """L divides every term of a document by the same 1 + log(ave tf), so after L2
        normalization Lnc and lnc are the same vector; with u they are not."""
        space = VectorSpace(DOCS)
        scores = lambda code: [(h.pmid, round(h.score, 12)) for h in space.search("semaglutide weight", code).hits]
        self.assertEqual(scores("Lnc.ltc"), scores("lnc.ltc"))
        self.assertNotEqual(scores("Lnu.ltc"), scores("lnu.ltc"))

    def test_schemes(self):
        scheme = Scheme.parse("lnc.ltc")
        self.assertEqual((scheme.document.code, scheme.query.code, scheme.code), ("lnc", "ltc", "lnc.ltc"))
        self.assertEqual(Scheme.parse("Lnu.ltc").document.describe(), "log average TF × no IDF, pivoted unique")
        for bad in ("ltc", "xtc.ltc", "lnc.ltcc", "lnc-ltc", "LNC.ltc", "bm25"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Scheme.parse(bad)


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space = VectorSpace(DOCS)

    def test_index_counts_title_and_abstract(self):
        self.assertEqual(self.space.tf["1"]["semaglutid"], 4)
        self.assertEqual(self.space.df("semaglutid"), 2)
        self.assertEqual(self.space.max_tf["1"], 4)

    def test_raw_scores_are_inner_products(self):
        ranking = self.space.search("semaglutide weight", "nnn.nnn")
        scores = {h.pmid: h.score for h in ranking.hits}
        self.assertEqual(scores["1"], 4 * 1 + 2 * 1)                 # tf(semaglutid) + tf(weight)
        self.assertEqual(scores["2"], 3)
        for hit in ranking.hits:
            self.assertAlmostEqual(sum(hit.contributions.values()), hit.score)

    def test_cosine_matches_a_hand_computation(self):
        n = len(self.space)
        weighting = Weighting("l", "t", "c")
        doc = {t: (1 + math.log10(tf)) * math.log10(n / self.space.df(t)) for t, tf in self.space.tf["4"].items()}
        query = {t: math.log10(n / self.space.df(t)) for t in ("semaglutid", "insulin")}
        expected = (sum(query[t] * doc[t] for t in query)
                    / math.sqrt(sum(v * v for v in doc.values())) / math.sqrt(sum(v * v for v in query.values())))
        hit = next(h for h in self.space.search("semaglutide insulin", Scheme(weighting, weighting)).hits
                   if h.pmid == "4")
        self.assertAlmostEqual(hit.score, expected)
        self.assertLessEqual(max(h.score for h in self.space.search("semaglutide insulin", "ltc.ltc").hits), 1 + 1e-9)

    def test_normalization_and_ranking_details(self):
        raw = [h.pmid for h in self.space.search("weight", "ntn.ntn").hits]
        self.assertEqual(raw[0], "2")                                 # three occurrences
        ranking = self.space.search("weight unicorn", "lnc.ltc", k=1)
        self.assertEqual(len(ranking.hits), 1)
        self.assertEqual(ranking.unknown, ["unicorn"])
        self.assertEqual(self.space.search("the of", "lnc.ltc").hits, [])
        self.assertTrue(self.space.search("semaglutide weight", "lnb.ltc").hits)
        # Ties are broken by PMID, so rankings are reproducible.
        tied = self.space.search("secretion", "bnc.bnc").hits
        self.assertEqual([h.pmid for h in tied], sorted(h.pmid for h in tied))


class EvaluationTest(unittest.TestCase):
    def test_mesh_judgments_and_grades(self):
        self.assertEqual(judgments(DOCS, Query("q", "x", ("Semaglutide",))), {"1": 2, "4": 1})
        self.assertEqual(judgments(DOCS, Query("q", "x", ("Semaglutide", "Weight Loss"))), {"1": 2})
        self.assertEqual(judgments(DOCS, Query("q", "x", (), ("Insulin", "Weight Loss"))), {"1": 1, "2": 1, "3": 2, "4": 1})

    def test_metrics_by_hand(self):
        grades = {"a": 2, "c": 1, "z": 1}
        m = score(["b", "a", "c", "d", "e"], grades).values
        self.assertAlmostEqual(m["P@5"], 2 / 5)
        self.assertAlmostEqual(m["R@10"], 2 / 3)
        self.assertAlmostEqual(m["MRR"], 1 / 2)
        self.assertAlmostEqual(m["MAP"], (1 / 2 + 2 / 3) / 3)
        dcg = 3 / math.log2(3) + 1 / math.log2(4)
        ideal = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(m["nDCG@10"], dcg / ideal)
        empty = score([], grades).values
        self.assertEqual((empty["MRR"], empty["MAP"], empty["nDCG@10"]), (0.0, 0.0, 0.0))

    def test_paired_permutation(self):
        self.assertEqual(paired_permutation([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(paired_permutation([1] * 5, [0] * 5), 2 / 32)
        with self.assertRaises(ValueError):
            paired_permutation([1] * 17, [0] * 17)


class MeshCorpusTest(unittest.TestCase):
    def test_round_trip_and_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "c.jsonl")
            save_jsonl(DOCS + [Document("5", "No MeSH", "Text.")], path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertNotIn("mesh", lines[-1])                       # files without MeSH keep three fields
            self.assertEqual(load(path)[:4], DOCS)
            csv_path = Path(folder, "c.csv")
            csv_path.write_text("pmid,abstract,mesh\n9,x,Insulin; Humans\n", encoding="utf-8")
            self.assertEqual(load(csv_path)[0].mesh, ("Insulin", "Humans"))

    def test_major_topic_from_a_qualifier(self):
        xml = (b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article><ArticleTitle>T"
               b"</ArticleTitle><Abstract><AbstractText>A.</AbstractText></Abstract></Article><MeshHeadingList>"
               b"<MeshHeading><DescriptorName MajorTopicYN='N'>Humans</DescriptorName></MeshHeading>"
               b"<MeshHeading><DescriptorName MajorTopicYN='N'>Obesity</DescriptorName>"
               b"<QualifierName MajorTopicYN='Y'>drug therapy</QualifierName></MeshHeading>"
               b"</MeshHeadingList></MedlineCitation></PubmedArticle></PubmedArticleSet>")
        document = parse_pubmed_xml(xml)[0]
        self.assertEqual((document.mesh, document.mesh_major), (("Humans", "Obesity"), ("Obesity",)))


class ExperimentTest(unittest.TestCase):
    def test_run_writes_every_output(self):
        with tempfile.TemporaryDirectory() as folder:
            queries = Path(folder, "queries.json")
            queries.write_text(json.dumps({"queries": [
                {"id": "Q1", "query": "semaglutide weight", "mesh_all": ["Semaglutide"]},
                {"id": "Q2", "query": "insulin secretion", "mesh_any": ["Insulin"]}]}), encoding="utf-8")
            self.assertEqual([q.id for q in load_queries(queries)], ["Q1", "Q2"])
            summary = run(DOCS, queries, Path(folder, "out"), ["lnc.ltc", "ltc.ltc", "lnu.ltc"])
            self.assertEqual([r["scheme"] for r in summary], ["lnc.ltc", "ltc.ltc", "lnu.ltc"])
            for name in ("summary", "per_query", "comparisons", "judgments", "idf", "tf_example", "length_bias"):
                self.assertTrue(Path(folder, "out", f"{name}.csv").exists(), name)
            manifest = json.loads(Path(folder, "out", "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["queries"][0]["relevant"], 2)


if __name__ == "__main__":
    unittest.main()
