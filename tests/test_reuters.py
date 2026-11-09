import math
import unittest
from pathlib import Path

from bioir.news.queries import NewsQuery, coverage, judgments, load_queries
from bioir.news.reuters import Article, load_reuters, parse

DATA = Path(__file__).resolve().parents[1] / "data" / "reuters21578"

SAMPLE = """<!DOCTYPE lewis SYSTEM "lewis.dtd">
<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="1" NEWID="1">
<TOPICS><D>crude</D><D>nat-gas</D></TOPICS>
<TEXT>&#2;
<TITLE>OIL &amp; GAS &lt;XON&gt; NEWS</TITLE>
<BODY>Crude prices rose.
    Petroleum supply fell.
 Reuter
&#3;</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="NO" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="2" NEWID="2">
<TOPICS></TOPICS><TEXT><TITLE>NOT IN MODAPTE</TITLE><BODY>x</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="NOT-USED" CGISPLIT="TRAINING-SET" OLDID="3" NEWID="3">
<TOPICS><D>earn</D></TOPICS><TEXT><TITLE>NOT USED</TITLE><BODY>x</BODY></TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="PUBLISHED-TESTSET" OLDID="4" NEWID="4">
<TOPICS></TOPICS><TEXT TYPE="BRIEF">&#2;
<TITLE>BRIEF WITH NO TOPIC</TITLE>
&#3;</TEXT>
</REUTERS>
<REUTERS TOPICS="YES" LEWISSPLIT="TEST" CGISPLIT="PUBLISHED-TESTSET" OLDID="5" NEWID="5">
<TOPICS><D>money-supply</D></TOPICS><TEXT TYPE="UNPROC">&#2;
FED WEEKLY REPORT reserves rose
&#3;</TEXT>
</REUTERS>
"""


class ParseTest(unittest.TestCase):
    def test_modapte_filter_fields_and_cleaning(self):
        articles = {a.newid: a for a in parse(SAMPLE)}
        self.assertEqual(sorted(articles), ["1", "4", "5"])          # TOPICS="NO" and NOT-USED are dropped
        first = articles["1"]
        self.assertEqual((first.split, first.topics), ("train", ("crude", "nat-gas")))
        self.assertEqual(first.title, "OIL & GAS <XON> NEWS")         # entities unescaped
        self.assertEqual(first.body, "Crude prices rose. Petroleum supply fell.")   # no Reuter sign-off
        brief = articles["4"]
        self.assertEqual((brief.split, brief.topics, brief.title, brief.body), ("test", (), "BRIEF WITH NO TOPIC", ""))
        self.assertEqual(articles["5"].body, "FED WEEKLY REPORT reserves rose")    # UNPROC: the whole <TEXT>

    def test_document_view(self):
        document = parse(SAMPLE)[0].document
        self.assertEqual((document.pmid, document.title, document.abstract),
                         ("1", "OIL & GAS <XON> NEWS", "Crude prices rose. Petroleum supply fell."))


class QueryTest(unittest.TestCase):
    ARTICLES = [Article("1", "test", "Oil", "Crude oil prices rose.", ("crude",)),
                Article("2", "test", "Petroleum", "Petroleum supply fell.", ("crude", "nat-gas")),
                Article("3", "train", "Oil", "Crude oil.", ("crude",)),
                Article("4", "test", "Wheat", "Wheat exports.", ("grain",))]

    def test_topic_judgments_use_only_test_stories(self):
        query = NewsQuery("C", "crude oil prices", "crude", "keyword")
        self.assertEqual(judgments(self.ARTICLES, query), {"1": 1, "2": 1})
        self.assertEqual(query.terms, {"crude", "oil", "price"})

    def test_coverage_counts_stories_sharing_query_terms(self):
        query = NewsQuery("C", "crude oil", "crude", "keyword")
        self.assertEqual(coverage(self.ARTICLES, query), {"relevant": 2, "any_term": 1, "all_terms": 1, "no_term": 1})

    def test_query_file(self):
        queries = load_queries(Path(__file__).resolve().parents[1] / "data" / "reuters_queries.json")
        self.assertEqual(len(queries), 15)
        self.assertEqual({q.topic for q in queries}, {"earn", "acq", "crude", "grain", "trade"})
        self.assertEqual(sum(q.kind == "paraphrase" for q in queries), 5)


class MethodATest(unittest.TestCase):
    def test_tfidf_ranks_test_stories_and_cannot_reach_stories_without_query_terms(self):
        from bioir.news.retrieval import evaluate, mean, test_space, tfidf_ranker
        articles = QueryTest.ARTICLES
        space = test_space(articles)
        self.assertEqual(sorted(space.documents), ["1", "2", "4"])          # training stories are not searched
        query = NewsQuery("C", "crude oil", "crude", "keyword")
        self.assertEqual(tfidf_ranker(space)(query.text), ["1"])            # story 2 shares no term: score 0
        row = evaluate(tfidf_ranker(space), articles, [query])[0]
        self.assertEqual((row["relevant"], row["relevant_retrieved"], row["P@10"], row["MRR"]), (2, 1, 0.1, 1.0))
        self.assertEqual(row["relevant_retrieved"], coverage(articles, query)["any_term"])
        self.assertEqual(mean([row, row])["MRR"], 1.0)


class MethodBTest(unittest.TestCase):
    """A hand-made 2-D embedding: oil and petroleum point the same way, wheat elsewhere."""

    def setUp(self):
        try:
            import numpy as np
            from bioir.embed import Embeddings
        except ImportError:
            self.skipTest("numpy not installed")
        vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.6]])
        self.model = Embeddings(["oil", "petroleum", "wheat", "prices"], [5, 5, 5, 5], vectors, {})
        self.articles = [Article("1", "test", "", "Oil prices rose.", ("crude",)),
                         Article("2", "test", "", "Wheat wheat exports.", ("grain",)),
                         Article("3", "train", "", "Petroleum.", ("crude",))]

    def test_reaches_a_story_that_shares_no_query_word(self):
        from bioir.news.semantic import EmbeddingSpace, words
        space = EmbeddingSpace(self.articles, self.model)
        self.assertEqual(space.ids, ["1", "2"])                              # test stories only
        self.assertEqual(words("The petroleum (2) prices!", self.model), ["petroleum", "prices"])
        self.assertEqual(space.ranker()("petroleum"), ["1", "2"])            # "1" never says petroleum
        self.assertEqual(space.ranker()("the of"), [])                       # no known word: no ranking
        scores, _ = space.scores("petroleum")
        self.assertAlmostEqual(float(scores[1]), 0.1 / math.hypot(0.9, 0.1))   # cosine(petroleum, wheat)
        self.assertGreater(scores[0], 0.9)                                     # oil and prices lie near petroleum

    def test_explain_splits_the_score_into_word_contributions(self):
        from bioir.news.semantic import EmbeddingSpace
        for weighting in ("tfidf", "mean"):
            space = EmbeddingSpace(self.articles, self.model, weighting)
            scores, _ = space.scores("petroleum")
            for i, newid in enumerate(space.ids):
                steps = space.explain("petroleum", newid)
                self.assertAlmostEqual(steps["score"], float(scores[i]))      # Σ share × cos is the cosine
                for row in steps["rows"]:
                    self.assertAlmostEqual(row["contribution"], row["share"] * row["cosine"])
        steps = EmbeddingSpace(self.articles, self.model).explain("petroleum", "1")
        self.assertEqual([r["word"] for r in steps["query"]], ["petroleum"])
        self.assertEqual({r["word"] for r in steps["rows"]}, {"oil", "prices"})
        self.assertFalse(any(r["in_query"] for r in steps["rows"]))         # no shared word, yet a high score
        self.assertIsNone(space.explain("the of", "1"))                      # no known query word

    def test_weighting_and_hybrid(self):
        import math
        from collections import Counter
        from bioir.news.retrieval import test_space
        from bioir.news.semantic import EmbeddingSpace, hybrid
        space = EmbeddingSpace(self.articles, self.model)
        self.assertAlmostEqual(space.idf("oil"), math.log10(2))
        self.assertAlmostEqual(space.idf("unseen"), math.log10(2))           # df clamped to 1
        vector = space.vector(Counter({"oil": 1}))
        self.assertAlmostEqual(float((vector ** 2).sum()), 1.0)
        with self.assertRaises(ValueError):
            EmbeddingSpace(self.articles, self.model, "other")
        ranking = hybrid(test_space(self.articles), "ltc.ltc", space)("oil")
        self.assertEqual(ranking[0], "1")


@unittest.skipUnless(DATA.exists(), "Reuters-21578 not downloaded (see README)")
class CollectionTest(unittest.TestCase):
    def test_matches_the_official_modapte_counts(self):
        articles = load_reuters(DATA)
        self.assertEqual(sum(a.split == "train" for a in articles), 9603)
        self.assertEqual(sum(a.split == "test" for a in articles), 3299)
        self.assertTrue(all(a.title or a.body for a in articles))
        test_counts = {t: sum(a.split == "test" and t in a.topics for a in articles)
                       for t in ("earn", "acq", "crude", "grain", "trade")}
        self.assertEqual(test_counts, {"earn": 1087, "acq": 719, "crude": 189, "grain": 149, "trade": 117})


if __name__ == "__main__":
    unittest.main()
