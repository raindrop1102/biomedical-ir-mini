"""Homework 5: the basic index, BSBI and SPIMI must build the same postings and answer alike."""
import random
import tempfile
import unittest
from pathlib import Path

from bioir.indexing import IndexReader, InvertedIndex, boolean, bsbi, intersect, query_terms, scaled, spimi, union
from bioir.indexing.bsbi import invert_pairs
from bioir.indexing.storage import read_bsbi_block, read_spimi_block

DOCS = [
    (1, ["inform", "retriev", "system", "inform"]),
    (2, ["retriev", "system", "algorithm"]),
    (3, ["inform", "system"]),
    (4, ["system", "algorithm", "algorithm"]),
    (5, ["inform", "retriev"]),
]


def random_docs(n=300, seed=3):
    rng = random.Random(seed)
    vocabulary = [f"t{i}" for i in range(400)]
    return [(i, [rng.choice(vocabulary[: rng.randint(5, 400)]) for _ in range(rng.randint(0, 40))]) for i in range(1, n + 1)]


class BasicIndexTest(unittest.TestCase):
    def test_lecture_example(self):
        index = InvertedIndex(DOCS)
        self.assertEqual(index.vocabulary, ["algorithm", "inform", "retriev", "system"])
        self.assertEqual(index.lookup("inform"), [1, 3, 5])         # docs 1 counted once
        self.assertEqual(index.lookup("retriev"), [1, 2, 5])
        self.assertEqual(index.lookup("algorithm"), [2, 4])
        self.assertEqual(index.df("system"), 4)
        self.assertEqual(index.lookup("missing"), [])

    def test_merge_operations(self):
        self.assertEqual(intersect([1, 3, 5, 9], [2, 3, 9, 10]), [3, 9])
        self.assertEqual(union([1, 3, 5], [2, 3, 9]), [1, 2, 3, 5, 9])
        self.assertEqual(intersect([], [1]), [])
        self.assertEqual(union([], [4]), [4])

    def test_boolean_queries_use_the_documents_preprocessing(self):
        index = InvertedIndex(DOCS)
        self.assertEqual(query_terms("Information Retrieval of information"), ["inform", "retriev"])
        self.assertEqual(boolean(index, "information retrieval", "AND"), [1, 5])
        self.assertEqual(boolean(index, "information retrieval", "OR"), [1, 2, 3, 5])
        self.assertEqual(boolean(index, "the of", "AND"), [])      # nothing left after stop words
        with self.assertRaises(ValueError):
            boolean(index, "x", "NOT")


class ConstructionTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def assert_same_as_basic(self, folder: Path, docs):
        basic, reader = InvertedIndex(docs), IndexReader(folder)
        self.assertEqual(reader.vocabulary, basic.vocabulary)
        for term in basic.vocabulary:
            self.assertEqual(reader.postings(term), basic.lookup(term), term)
            self.assertEqual(reader.df(term), basic.df(term))

    def test_bsbi_blocks_sort_and_merge(self):
        trace = {}
        stats = bsbi(DOCS, self.root / "out", self.root / "work", block_docs=2, trace=trace)
        self.assertEqual(stats.blocks, 3)
        self.assertEqual(stats.pairs, sum(len(t) for _, t in DOCS))          # every occurrence is a pair
        self.assertEqual(stats.intermediate_files, 4)                         # 3 blocks + terms.tsv
        self.assertEqual(trace["pairs"][:4], [(0, 1, "inform"), (1, 1, "retriev"), (2, 1, "system"), (0, 1, "inform")])
        self.assertEqual(trace["sorted"][0], (0, 1, "inform", 2))             # the repeat sits next to its twin
        self.assert_same_as_basic(self.root / "out", DOCS)
        # termIDs agree across blocks: "inform" is 0 in block 1 and block 2
        self.assertEqual(next(read_bsbi_block(self.root / "work" / "bsbi_block_002.bin")), (0, [3]))

    def test_invert_pairs_drops_repeats(self):
        self.assertEqual(invert_pairs([(0, 1), (0, 1), (0, 4), (2, 3)]), [(0, [1, 4]), (2, [3])])

    def test_spimi_flushes_at_capacity_and_merges(self):
        stats = spimi(DOCS, self.root / "out", self.root / "work", capacity=4)
        self.assertEqual(stats.pairs, 0)
        self.assertGreater(stats.blocks, 1)
        self.assertEqual(stats.intermediate_files, stats.blocks)
        first = list(read_spimi_block(self.root / "work" / "spimi_block_001.bin"))
        self.assertEqual([t for t, _ in first], sorted(t for t, _ in first))  # written in term order
        self.assertEqual(sum(b["documents"] for b in stats.blocks_info), len(DOCS))
        self.assert_same_as_basic(self.root / "out", DOCS)

    def test_any_block_size_gives_the_same_index(self):
        docs = random_docs()
        for size in (1, 7, 50, 1000):
            out = self.root / f"bsbi_{size}"
            bsbi(docs, out, self.root / f"bw_{size}", block_docs=size)
            self.assert_same_as_basic(out, docs)
        for capacity in (1, 37, 500, 10**6):
            out = self.root / f"spimi_{capacity}"
            spimi(docs, out, self.root / f"sw_{capacity}", capacity=capacity)
            self.assert_same_as_basic(out, docs)
        a, b = IndexReader(self.root / "bsbi_7"), IndexReader(self.root / "spimi_37")
        for query in ("t1 t2", "t3 t30 t300", "t0"):
            text = " ".join(query.split())
            self.assertEqual(boolean(a, text, "AND"), boolean(b, text, "AND"))
            self.assertEqual(boolean(a, text, "OR"), boolean(b, text, "OR"))

    def test_final_postings_files_have_the_same_size(self):
        docs = random_docs(120)
        a = bsbi(docs, self.root / "a", self.root / "aw", block_docs=10)
        b = spimi(docs, self.root / "b", self.root / "bw", capacity=200)
        self.assertEqual(a.final["postings.bin"], b.final["postings.bin"])
        self.assertEqual(a.postings, b.postings)
        self.assertEqual(a.terms, b.terms)

    def test_bad_settings(self):
        with self.assertRaises(ValueError):
            bsbi(DOCS, self.root / "o", self.root / "w", block_docs=0)
        with self.assertRaises(ValueError):
            spimi(DOCS, self.root / "o", self.root / "w", capacity=0)


class ScaledTest(unittest.TestCase):
    def test_repeats_and_renumbers(self):
        docs = list(scaled(DOCS, 12))
        self.assertEqual([d for d, _ in docs], list(range(1, 13)))
        self.assertEqual(docs[5][1], DOCS[0][1])


if __name__ == "__main__":
    unittest.main()
