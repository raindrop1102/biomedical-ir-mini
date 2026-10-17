import importlib.util
import tempfile
import unittest
from pathlib import Path

from bioir import Document
from bioir.corpus import save_jsonl

HAS_FLASK = importlib.util.find_spec("flask") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

DOCS = [
    Document("1", "GLP-1 receptor agonists <b>in</b> type 2 diabetes",
             "GLP-1RAs such as semaglutide lower HbA1c in patients with type 2 diabetes. " * 6),
    Document("2", "Semaglutide and weight loss", "Semaglutide produced weight loss. Weight loss was larger."),
    Document("3", "Tirzepatide in mice", "Tirzepatide reduced body weight in obese mice. <script>alert(1)</script>"),
]


@unittest.skipUnless(HAS_FLASK, "flask not installed")
class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bioir.web import create_app
        cls.folder = tempfile.TemporaryDirectory()
        corpus = Path(cls.folder.name, "corpus.jsonl")
        save_jsonl(DOCS, corpus)
        cls.client = create_app(corpus, Path(cls.folder.name, "no-results")).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def get(self, url, status=200):
        response = self.client.get(url)
        self.assertEqual(response.status_code, status, url)
        return response.get_data(as_text=True)

    def test_landing_page(self):
        page = self.get("/")
        self.assertIn("查詢語法", page)
        self.assertIn("corpus.jsonl", page)

    def test_search_highlights_and_escapes(self):
        page = self.get("/?query=GLP-1")
        self.assertIn("找到 1 篇", page)
        self.assertIn("<mark>GLP-1</mark>", page)              # the whole identifier, as one phrase
        self.assertIn("&lt;b&gt;in&lt;/b&gt;", page)           # markup in a title stays text
        self.assertIn("完整摘要", page)                          # a long abstract gets a snippet
        page = self.get("/?query=mice")
        self.assertNotIn("<script>alert(1)", page)
        self.assertIn("&lt;script&gt;", page)
        page = self.get("/?query=%3Cscript%3Ealert(1)%3C/script%3E")
        self.assertNotIn("<script>alert(1)", page)

    def test_boolean_query_pagination_and_errors(self):
        page = self.get("/?query=semaglutide+OR+tirzepatide&per_page=10")
        self.assertIn("找到 3 篇", page)
        self.assertIn("NOT", self.get("/?query=weight+NOT+mice"))   # query echoed back
        self.assertIn("查詢無法執行", self.get("/?query=(glp", status=400))
        self.assertIn("您是不是要找", self.get("/?query=semaglutde"))
        self.get("/?query=weight&page=99&per_page=7")             # out-of-range values fall back

    def test_zipf_page_and_downloads(self):
        page = self.get("/zipf?baseline=D&no_numbers=1&terms=Patients,the")
        self.assertIn("條件 D 的分區擬合", page)
        self.assertIn("patient", page)
        self.assertIn("（停用詞或純數字）", page)
        self.assertTrue(self.get("/zipf/summary.csv").startswith("﻿condition"))
        self.assertIn("cf_per_df", self.get("/zipf/terms_B.csv?compound=1"))
        self.get("/zipf/other.csv", status=404)
        image = self.client.get("/zipf.png")
        self.assertEqual(image.status_code, 200 if HAS_MATPLOTLIB else 404)
        if HAS_MATPLOTLIB:
            self.assertTrue(image.data.startswith(b"\x89PNG"))

    def test_embedding_page_without_models(self):
        page = self.get("/embedding?word=semaglutide")
        self.assertIn("尚未提供詞向量模型", page)
        self.get("/embedding/projections.png", status=404)

    def test_health_and_headers(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.get_json()["documents"], 3)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")


@unittest.skipUnless(HAS_FLASK and HAS_NUMPY, "flask or numpy not installed")
class EmbeddingPageTest(unittest.TestCase):
    def test_neighbors_and_pairs(self):
        from bioir.embed import Config, train
        from bioir.web import create_app
        with tempfile.TemporaryDirectory() as folder:
            docs = [Document(str(i), "", f"The {('cat', 'dog')[i % 2]} slept near the barn. "
                                         f"A {('car', 'bus')[i % 2]} drove down the road.") for i in range(1, 81)]
            corpus = Path(folder, "toy.jsonl")
            save_jsonl(docs, corpus)
            settings = dict(dimensions=8, window=2, epochs=3, min_count=1, subsample=0)
            for name, arch in (("cbow_w5_s42", "cbow"), ("skipgram_w5_s42", "skipgram")):
                train(docs, Config(architecture=arch, **settings)).save(Path(folder, "models", f"{name}.npz"))
            client = create_app(corpus, folder).test_client()
            page = client.get("/embedding?word=cat&min_count=1&a=cat&b=dog").get_data(as_text=True)
            self.assertIn("CBOW", page)
            self.assertIn("Skip-gram", page)
            self.assertIn("高於隨機詞對", page)
            self.assertIn("不在詞彙表中", client.get("/embedding?word=zebra").get_data(as_text=True))
            self.assertIn("不是單一詞", client.get("/embedding?word=two+words").get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
