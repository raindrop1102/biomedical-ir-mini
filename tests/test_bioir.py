import json
import tempfile
import unittest
from pathlib import Path

from bioir import Document, Index, Speller, load
from bioir.corpus import parse_pubmed_xml
from bioir.porter import cvc, measure, stem
from bioir.search.query import parse
from bioir.search.spelling import edit_distance
from bioir.text import analyze

DOCS = [
    Document("1", "GLP-1 receptor agonists in type 2 diabetes",
             "GLP-1RAs such as semaglutide lower HbA1c in patients with type 2 diabetes."),
    Document("2", "Semaglutide and weight loss",
             "Semaglutide produced weight loss. Weight loss was larger than with placebo."),
    Document("3", "Tirzepatide in mice", "Tirzepatide reduced body weight in obese mice."),
    Document("4", "Insulin therapy", "Insulin type 1 versus type 2 and diabetes outcomes in a long cohort "
             "study of many patients followed for several years with repeated measurements."),
    Document("5", "SGLT2 inhibitors", "SGLT-2 inhibitors and glp1 analogues were compared."),
]


class PorterTest(unittest.TestCase):
    def test_examples_from_the_paper(self):
        cases = {"caresses": "caress", "ponies": "poni", "cats": "cat", "feed": "feed",
                 "agreed": "agre", "plastered": "plaster", "motoring": "motor", "hopping": "hop",
                 "filing": "file", "happy": "happi", "relational": "relat", "conditional": "condit",
                 "generalization": "gener", "hopefulness": "hope", "revival": "reviv",
                 "adoption": "adopt", "controll": "control", "roll": "roll", "probate": "probat"}
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(stem(word), expected)

    def test_conditions_and_non_words(self):
        self.assertEqual([measure(w) for w in ("tr", "trouble", "troubles", "private")], [0, 1, 2, 2])
        self.assertTrue(cvc("hop"))
        self.assertFalse(cvc("snow"))
        self.assertEqual(stem("p53"), "p53")


class AnalyzeTest(unittest.TestCase):
    def test_identifier_variants_share_terms(self):
        terms = lambda text: [t for t, _ in analyze(text)]
        self.assertEqual(terms("GLP-1RAs"), ["glp", "1", "ra"])
        self.assertEqual(terms("GLP-1RAs"), terms("glp1 RAs"))
        self.assertEqual(terms("SGLT-2"), terms("SGLT2"))

    def test_stop_words_keep_positions_and_possessives_drop(self):
        self.assertEqual(analyze("the patient's dose"), [("patient", 1), ("dose", 2)])


class SearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = Index(DOCS)

    def ids(self, query):
        return {doc.pmid for _, doc in self.index.search(query, k=100).hits}

    def test_identifier_spellings_match_each_other(self):
        self.assertEqual(self.ids("GLP-1"), {"1", "5"})
        self.assertEqual(self.ids("GLP1"), self.ids("glp-1"))
        self.assertEqual(self.ids("SGLT2"), {"5"})

    def test_boolean_operators(self):
        self.assertEqual(self.ids("semaglutide weight"), {"2"})
        self.assertEqual(self.ids("semaglutide OR tirzepatide"), {"1", "2", "3"})
        self.assertEqual(self.ids("weight NOT mice"), {"2"})
        self.assertEqual(self.ids("(semaglutide OR tirzepatide) AND NOT weight"), {"1"})
        self.assertEqual(self.ids("NOT diabetes"), {"2", "3", "5"})

    def test_phrases_need_adjacency_but_not_across_fields(self):
        self.assertEqual(self.ids('"type 2 diabetes"'), {"1"})
        self.assertEqual(self.ids("type 2 diabetes"), {"1", "4"})
        self.assertEqual(self.ids('"diabetes glp"'), set())  # title end + abstract start

    def test_bm25_prefers_more_occurrences(self):
        hits = self.index.search("weight", k=3).hits
        self.assertEqual(hits[0][1].pmid, "2")
        self.assertGreater(hits[0][0], hits[1][0])

    def test_empty_and_invalid_queries(self):
        self.assertEqual(self.index.search("the of").total, 0)
        self.assertIsNone(parse("and the"))
        for bad in ("(glp", "glp )", "NOT"):
            with self.subTest(query=bad), self.assertRaises(ValueError):
                parse(bad)


class SpellingTest(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting", 5), 3)
        self.assertEqual(edit_distance("weigth", "weight", 2), 1)
        self.assertEqual(edit_distance("abcdef", "uvwxyz", 2), 3)

    def test_suggestions(self):
        speller = Speller(Index(DOCS))
        self.assertEqual(speller.suggest("semaglutde AND weigth"), "semaglutide AND weight")
        self.assertIsNone(speller.suggest("semaglutide NOT mice"))


class CorpusTest(unittest.TestCase):
    def test_formats_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            jsonl, csv_file = Path(folder, "a.jsonl"), Path(folder, "a.csv")
            jsonl.write_text("\n".join(json.dumps(r) for r in [
                {"pmid": 7, "title": " A ", "abstract": "x\n y"},
                {"pmid": "7", "title": "dup", "abstract": "z"}]), encoding="utf-8")
            csv_file.write_text("pmid,abstract\n8,\"one, two\"\n", encoding="utf-8")
            self.assertEqual(load(jsonl), [Document("7", "A", "x y")])
            self.assertEqual(load(csv_file), [Document("8", "", "one, two")])
            with self.assertRaises(ValueError):
                load(Path(folder, "a.txt"))

    def test_pubmed_xml(self):
        xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>9</PMID><Article>
            <ArticleTitle>T <i>x</i></ArticleTitle><Abstract><AbstractText Label="A">One.</AbstractText>
            <AbstractText>Two.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
            </PubmedArticleSet>"""
        self.assertEqual(parse_pubmed_xml(xml), [Document("9", "T x", "A: One. Two.")])


if __name__ == "__main__":
    unittest.main()
