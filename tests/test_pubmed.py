"""PubMed client tests. None of them touch the network: urlopen and sleep are replaced."""
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.parse
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from bioir import Document, load, pubmed
from bioir.__main__ import main
from bioir.corpus import manifest_path, parse_pubmed_xml

XML = b"""<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN"
  "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>A <i>GLP-1</i> study</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">First part.</AbstractText>
<AbstractText Label="RESULTS">Second part.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>124</PMID><Article><ArticleTitle>No abstract</ArticleTitle>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>125</PMID><Article><ArticleTitle>Plain</ArticleTitle>
<Abstract><AbstractText>Unstructured.</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def http_error(code):
    return urllib.error.HTTPError(pubmed.EUTILS, code, "error", {}, io.BytesIO())


class NoNetwork(unittest.TestCase):
    """Every test starts with an urlopen that fails the test, a no-op sleep and a fresh rate-limit clock."""

    def setUp(self):
        self.urlopen = mock.Mock(side_effect=AssertionError("unexpected network access"))
        self.sleep = mock.Mock()
        for name, value in (("urlopen", self.urlopen), ("sleep", self.sleep), ("_last_request", float("-inf"))):
            patcher = mock.patch.object(pubmed, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def body(self, call_index=-1) -> dict:
        """Decoded POST parameters of one urlopen call."""
        args = self.urlopen.call_args_list[call_index].args
        return dict(urllib.parse.parse_qsl(args[1].decode()))


class ParseTest(unittest.TestCase):
    def test_structured_abstract_keeps_labels(self):
        self.assertEqual(parse_pubmed_xml(XML), [
            Document("123", "A GLP-1 study", "BACKGROUND: First part. RESULTS: Second part."),
            Document("125", "Plain", "Unstructured.")])

    def test_ncbi_errors_entities_and_bad_xml_raise(self):
        bad = (b"<eFetchResult><ERROR>Empty id list</ERROR></eFetchResult>",
               b'<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>',
               b"<PubmedArticleSet><oops>")
        for data in bad:
            with self.subTest(data=data[:30]), self.assertRaises(ValueError):
                parse_pubmed_xml(data)

    def test_pmids_and_urls(self):
        text = "123, 123\nhttps://pubmed.ncbi.nlm.nih.gov/456/ ; http://www.pubmed.ncbi.nlm.nih.gov/789?from=x"
        self.assertEqual(pubmed.parse_pmids(text), ["123", "456", "789"])
        for bad in ("abc 123", "0", "12x34", "https://example.com/123/"):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                pubmed.parse_pmids(bad)


class RequestTest(NoNetwork):
    def test_retries_429_and_5xx_with_backoff(self):
        self.urlopen.side_effect = [http_error(429), http_error(503), io.BytesIO(b"ok")]
        self.assertEqual(pubmed.request("efetch", {}, "a@b.c"), b"ok")
        self.assertEqual(self.urlopen.call_count, 3)
        backoff = [c.args[0] for c in self.sleep.call_args_list if c.args[0] >= 1]
        self.assertEqual(backoff, [1, 2])
        self.assertEqual(self.urlopen.call_args.kwargs["timeout"], pubmed.TIMEOUT)

    def test_client_errors_are_not_retried(self):
        self.urlopen.side_effect = [http_error(400)]
        with self.assertRaisesRegex(pubmed.PubMedError, "HTTP 400"):
            pubmed.request("efetch", {}, "a@b.c")
        self.assertEqual(self.urlopen.call_count, 1)

    def test_gives_up_after_five_attempts_without_leaking_the_key(self):
        secret = "SECRET-KEY-123"
        self.urlopen.side_effect = urllib.error.URLError(f"timed out: {pubmed.EUTILS}?api_key={secret}")
        with mock.patch.dict(os.environ, {"NCBI_API_KEY": secret}):
            with self.assertRaises(pubmed.PubMedError) as caught:
                pubmed.request("esearch", {"term": "x"}, "a@b.c")
        self.assertEqual(self.urlopen.call_count, pubmed.ATTEMPTS)
        error = caught.exception
        self.assertNotIn(secret, str(error))
        self.assertIsNone(error.__cause__)
        self.assertTrue(error.__context__ is None or error.__suppress_context__)
        # The key travels in the POST body, never in the URL.
        url, data = self.urlopen.call_args.args
        self.assertNotIn(secret, url)
        self.assertEqual(self.body()["api_key"], secret)

    def test_no_key_parameter_without_the_environment_variable(self):
        self.urlopen.side_effect = [io.BytesIO(b"ok")]
        with mock.patch.dict(os.environ, {}, clear=True):
            pubmed.request("efetch", {}, "a@b.c")
        self.assertNotIn("api_key", self.body())

    def test_requests_are_spaced_by_the_shared_clock(self):
        self.urlopen.side_effect = lambda *a, **k: io.BytesIO(b"ok")
        with mock.patch.object(pubmed, "monotonic", side_effect=[100.0, 100.0, 100.1, 100.36]):
            pubmed.request("efetch", {}, "a@b.c")
            pubmed.request("efetch", {}, "a@b.c")
        first, second = (c.args[0] for c in self.sleep.call_args_list)
        self.assertEqual(first, 0.0)
        self.assertAlmostEqual(second, pubmed.MIN_INTERVAL - 0.1)


class EUtilsTest(NoNetwork):
    def test_esearch_sorts_by_relevance(self):
        payload = {"esearchresult": {"count": "30875", "idlist": ["3", "1", "2"],
                                     "querytranslation": '"glp-1"[All Fields]'}}
        self.urlopen.side_effect = [io.BytesIO(json.dumps(payload).encode())]
        pmids, info = pubmed.esearch("GLP-1", "a@b.c", 3)
        self.assertEqual(pmids, ["3", "1", "2"])
        self.assertEqual(info, {"total": 30875, "query_translation": '"glp-1"[All Fields]'})
        body = self.body()
        self.assertEqual((body["term"], body["sort"], body["retmax"]), ("GLP-1", "relevance", "3"))
        self.assertTrue(self.urlopen.call_args.args[0].endswith("/esearch.fcgi"))

    def test_esearch_rejects_bad_input_before_any_request(self):
        for term, count in (("", 10), ("glp-1", 0), ("glp-1", pubmed.MAX_RESULTS + 1)):
            with self.subTest(term=term, count=count), self.assertRaises(ValueError):
                pubmed.esearch(term, "a@b.c", count)
        payload = {"esearchresult": {"ERROR": "Invalid query"}}
        self.urlopen.side_effect = [io.BytesIO(json.dumps(payload).encode())]
        with self.assertRaisesRegex(ValueError, "Invalid query"):
            pubmed.esearch("((", "a@b.c")

    def test_fetch_batches_and_reports_missing(self):
        self.urlopen.side_effect = lambda *a, **k: io.BytesIO(XML)
        with mock.patch.object(pubmed, "BATCH", 2):
            documents, missing = pubmed.fetch(["125", "https://pubmed.ncbi.nlm.nih.gov/123/", "124", "999"],
                                              "a@b.c")
        self.assertEqual([d.pmid for d in documents], ["125", "123"])
        self.assertEqual(missing, ["124", "999"])
        self.assertEqual([self.body(i)["id"] for i in range(2)], ["125,123", "124,999"])


class FetchCommandTest(NoNetwork):
    def test_query_fetch_writes_a_manifest_that_load_verifies(self):
        def fake_urlopen(url, data, timeout):
            if url.endswith("esearch.fcgi"):
                result = {"count": "2", "idlist": ["123", "125"], "querytranslation": "glp-1"}
                return io.BytesIO(json.dumps({"esearchresult": result}).encode())
            return io.BytesIO(XML)
        self.urlopen.side_effect = fake_urlopen
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder, "abstracts.jsonl")
            with redirect_stdout(io.StringIO()):
                code = main(["fetch", "--query", "GLP-1", "-n", "2", "--email", "a@b.c", "-o", str(output)])
            self.assertEqual(code, 0)
            manifest = json.loads(manifest_path(output).read_text(encoding="utf-8"))
            self.assertEqual(manifest_path(output).name, "abstracts.manifest.json")
            self.assertEqual((manifest["query"], manifest["count"], manifest["sort"]), ("GLP-1", 2, "relevance"))
            self.assertIn("retrieved_at_utc", manifest)
            self.assertEqual(len(manifest["sha256"]), 64)

            with warnings.catch_warnings():
                warnings.simplefilter("error")       # an intact corpus must load silently
                self.assertEqual([d.pmid for d in load(output)], ["123", "125"])

            output.write_text(output.read_text(encoding="utf-8").replace("Second", "Altered"), encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "SHA-256"):
                load(output)

    def test_needs_exactly_one_source(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            self.assertEqual(main(["fetch", "--email", "a@b.c"]), 1)
        self.urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
