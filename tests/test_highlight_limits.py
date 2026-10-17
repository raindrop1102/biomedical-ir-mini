import json
import tempfile
import unittest
import warnings
from pathlib import Path

from bioir import Document, Index, highlight, load
from bioir.corpus import manifest_path, save_jsonl, write_manifest
from bioir.search.query import MAX_QUERY_CHARS, MAX_WORD_CHARS, MAX_WORDS, parse

DOCS = [
    Document("1", "GLP-1 receptor agonists", "In type 2 diabetes, GLP-1RAs lower HbA1c; type 1 was excluded."),
    Document("2", "<script>alert('GLP-1')</script>", "Tags & <b>GLP-1</b> analogues."),
]


class SpanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = Index(DOCS)

    def spans(self, query, pmid="1", field="abstract"):
        results = self.index.search(query)
        text = getattr(self.index.documents[pmid], field)
        return [text[start:end] for start, end in results.spans[pmid][field]]

    def test_spans_cover_whole_phrase_occurrences(self):
        self.assertEqual(self.spans("GLP-1"), ["GLP-1"])          # inside GLP-1RAs; the lone "1" is not a hit
        self.assertEqual(self.spans("GLP-1", field="title"), ["GLP-1"])
        self.assertEqual(self.spans('"type 2 diabetes" OR hba1c'), ["type 2 diabetes", "HbA1c"])
        self.assertEqual(self.spans("type NOT insulin"), ["type", "type"])   # NOT parts are never marked
        self.assertEqual(self.index.search("diabetes NOT excluded").spans, {})  # doc 1 is not a hit

    def test_only_returned_hits_get_spans(self):
        results = self.index.search("GLP-1", k=1)
        self.assertEqual(results.spans.keys(), {results.hits[0][1].pmid})


class HighlightTest(unittest.TestCase):
    def test_marks_spans_and_escapes_everything_else(self):
        doc = DOCS[1]
        results = Index(DOCS).search("GLP-1")
        title = highlight(doc.title, results.spans["2"]["title"])
        self.assertEqual(title, "&lt;script&gt;alert(&#x27;<mark>GLP-1</mark>&#x27;)&lt;/script&gt;")
        self.assertNotIn("<script>", title)
        abstract = highlight(doc.abstract, results.spans["2"]["abstract"])
        self.assertEqual(abstract, "Tags &amp; &lt;b&gt;<mark>GLP-1</mark>&lt;/b&gt; analogues.")

    def test_marked_text_is_escaped_too_and_overlaps_merge(self):
        self.assertEqual(highlight("<script>x", [(0, 8)]), "<mark>&lt;script&gt;</mark>x")
        self.assertEqual(highlight("abcdef", [(3, 5), (0, 2), (1, 4)]), "<mark>abcde</mark>f")
        self.assertEqual(highlight("a<b", []), "a&lt;b")


class QueryLimitTest(unittest.TestCase):
    def test_limits(self):
        parse(" ".join(["glp"] * MAX_WORDS))
        parse("x" * MAX_WORD_CHARS)
        too_much = (" ".join(["glp"] * (MAX_WORDS + 1)),
                    f'"{" ".join(["glp"] * (MAX_WORDS + 1))}"',     # words inside a phrase count
                    "x" * (MAX_WORD_CHARS + 1),
                    "a" * (MAX_QUERY_CHARS + 1))
        for query in too_much:
            with self.subTest(query=query[:30]), self.assertRaises(ValueError):
                parse(query)

    def test_operators_and_parentheses_do_not_count(self):
        words = ["w"] * MAX_WORDS
        parse("(" + " AND ".join(words) + ")")
        self.assertEqual(len(Index(DOCS).search(" OR ".join(["glp"] * MAX_WORDS)).hits), 2)


class ManifestTest(unittest.TestCase):
    def test_unreadable_or_foreign_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder, "a.jsonl")
            save_jsonl(DOCS, corpus)
            self.assertEqual(write_manifest(corpus, query="q")["file"], "a.jsonl")
            manifest = manifest_path(corpus)
            data = json.loads(manifest.read_text(encoding="utf-8"))

            manifest.write_text(json.dumps({**data, "file": "a.csv", "sha256": "0" * 64}), encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("error")   # describes a.csv, not this file
                load(corpus)

            manifest.write_text("not json", encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "integrity not checked"):
                load(corpus)


if __name__ == "__main__":
    unittest.main()
