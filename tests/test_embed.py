import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from bioir import Document
from bioir.text import sentences

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from bioir.__main__ import main
    from bioir.embed import Config, Embeddings, compare, corpus_sentences, overlap, sanity, train
    from bioir.embed.projection import pca, project, trustworthiness, tsne
    from bioir.embed.train import _contexts, _negative_sampling_step


def toy_documents(n: int = 300, seed: int = 0):
    """Animals and vehicles never share a sentence frame, so each group should cluster."""
    rng = np.random.default_rng(seed)
    frames = {"animal": (["cat", "dog", "fox"], ["The {} slept near the barn.", "A hungry {} ate the grain.",
                                                  "Our {} chased the hen."]),
              "vehicle": (["car", "bus", "van"], ["The {} drove down the road.", "A new {} parked by the station.",
                                                   "Our {} needed fuel today."])}
    docs = []
    for i in range(n):
        nouns, templates = frames["animal" if i % 2 else "vehicle"]
        text = " ".join(templates[rng.integers(3)].format(nouns[rng.integers(3)]) for _ in range(4))
        docs.append(Document(str(i + 1), "", text))
    return docs


class SentenceTest(unittest.TestCase):
    def test_sentence_split_keeps_decimals(self):
        self.assertEqual(sentences("HbA1c fell (p < 0.05). Weight fell 2.5 kg. 10 withdrew!"),
                         ["HbA1c fell (p < 0.05).", "Weight fell 2.5 kg.", "10 withdrew!"])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TrainingPiecesTest(unittest.TestCase):
    def test_corpus_keeps_stop_words_and_compounds_but_drops_numbers(self):
        docs = [Document("1", "", "The GLP-1 RAs reduced weight by 5%. Patients' HbA1c fell.")]
        self.assertEqual(corpus_sentences(docs), [["the", "glp-1", "ras", "reduced", "weight", "by"],
                                                   ["patients", "hba1c", "fell"]])
        self.assertEqual(corpus_sentences(docs, compound=False)[0][:3], ["the", "glp", "ras"])

    def test_contexts_never_cross_sentences(self):
        rng = np.random.default_rng(1)
        ids = [np.array([0, 1, 2]), np.array([3, 4])]
        targets, contexts = _contexts(ids, None, 2, rng)
        self.assertEqual(sorted(targets.tolist()), [0, 1, 2, 3, 4])
        for target, context in zip(targets, contexts):
            same = {0, 1, 2} if target < 3 else {3, 4}
            self.assertTrue(set(context[context >= 0].tolist()) <= same - {target})
            self.assertEqual(context.shape, (4,))

    def test_negative_sampling_gradients_match_finite_differences(self):
        rng = np.random.default_rng(2)
        outputs = rng.normal(size=(6, 4))
        hidden = rng.normal(size=(2, 4))
        positives, negatives = np.array([0, 1]), np.array([[2, 3], [4, 5]])

        def loss(out, hid):
            return _negative_sampling_step(out.copy(), hid, positives, negatives, 0.0)[0]

        _, hidden_gradient = _negative_sampling_step(outputs.copy(), hidden, positives, negatives, 0.0)
        updated = outputs.copy()
        _negative_sampling_step(updated, hidden, positives, negatives, 1.0)   # rate 1: outputs -= dL/dout
        eps = 1e-6
        for index in np.ndindex(hidden.shape):
            bump = np.zeros_like(hidden); bump[index] = eps
            numeric = (loss(outputs, hidden + bump) - loss(outputs, hidden - bump)) / (2 * eps)
            self.assertAlmostEqual(hidden_gradient[index], numeric, places=5)
        for index in np.ndindex(outputs.shape):
            bump = np.zeros_like(outputs); bump[index] = eps
            numeric = (loss(outputs + bump, hidden) - loss(outputs - bump, hidden)) / (2 * eps)
            self.assertAlmostEqual(outputs[index] - updated[index], numeric, places=5)

    def test_a_negative_equal_to_the_positive_is_ignored(self):
        outputs = np.ones((3, 2))
        loss, gradient = _negative_sampling_step(outputs, np.ones((1, 2)), np.array([0]), np.array([[0, 0]]), 0.1)
        self.assertAlmostEqual(loss, float(np.logaddexp(0, -2)))       # only the positive term remains

    def test_sanity_flags_vectors_sharing_one_direction(self):
        words = [f"w{i}" for i in range(50)]
        rng = np.random.default_rng(3)
        collapsed = Embeddings(words, np.ones(50), np.ones((50, 8)) + 0.01 * rng.normal(size=(50, 8)), {})
        healthy = Embeddings(words, np.ones(50), rng.normal(size=(50, 8)), {})
        self.assertFalse(sanity(collapsed)["passed"])
        self.assertTrue(sanity(healthy)["passed"])

    def test_overlap(self):
        a = [{"word": "x", "rank": 1, "neighbor": "a"}, {"word": "x", "rank": 2, "neighbor": "b"}]
        b = [{"word": "x", "rank": 1, "neighbor": "b"}, {"word": "x", "rank": 2, "neighbor": "c"}]
        self.assertEqual(overlap(a, b, k=2), [{"word": "x", "overlap": 0.5, "shared": "b"}])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class ProjectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        centers = np.eye(10)[:2] * 5
        self.vectors = np.vstack([centers[i // 6] + rng.normal(size=10) * 0.3 for i in range(12)])

    def test_pca_separates_two_clusters_on_the_first_axis(self):
        coordinates, explained = pca(self.vectors)
        self.assertEqual(coordinates.shape, (12, 2))
        self.assertTrue(0 < explained[1] <= explained[0] < 1)
        first, second = coordinates[:6, 0], coordinates[6:, 0]
        self.assertTrue(first.max() < second.min() or second.max() < first.min())

    def test_tsne_is_deterministic_and_keeps_neighbourhoods(self):
        a, kl, perplexity = tsne(self.vectors, seed=1)
        b, _, _ = tsne(self.vectors, seed=1)
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(kl, 0)
        self.assertAlmostEqual(perplexity, 11 / 3)
        self.assertGreater(trustworthiness(self.vectors, a), 0.9)
        with self.assertRaises(ValueError):
            tsne(self.vectors[:2])

    def test_trustworthiness_bounds(self):
        unit = self.vectors[:, :2] / np.linalg.norm(self.vectors[:, :2], axis=1, keepdims=True)
        self.assertAlmostEqual(trustworthiness(unit, unit), 1.0)
        shuffled = np.random.default_rng(5).permutation(unit)
        self.assertLess(trustworthiness(unit, shuffled), 1.0)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = toy_documents()
        settings = dict(dimensions=16, window=3, epochs=15, min_count=1, subsample=0, batch_size=32)
        cls.models = {arch: train(cls.docs, Config(architecture=arch, **settings)) for arch in ("cbow", "skipgram")}

    def test_both_architectures_learn_the_two_groups(self):
        for arch, model in self.models.items():
            with self.subTest(arch=arch):
                self.assertGreater(model.similarity("cat", "dog"), model.similarity("cat", "car"))
                self.assertGreater(model.similarity("bus", "van"), model.similarity("bus", "fox"))
                self.assertLess(model.metadata["history"][-1]["loss"], model.metadata["history"][0]["loss"])
                self.assertIn("sanity", model.metadata)

    def test_lookup_save_and_load(self):
        model = self.models["cbow"]
        self.assertEqual(model.term("Cat's"), "cat")
        self.assertIsNone(model.term("type 2"))
        self.assertIsNone(model.term("2"))
        with self.assertRaises(KeyError):
            model.similar("zebra")
        rare = model.similar("cat", 50, min_count=10 ** 6)
        self.assertEqual(rare, [])
        with tempfile.TemporaryDirectory() as folder:
            path = model.save(Path(folder, "m"))
            self.assertEqual(path.suffix, ".npz")
            loaded = Embeddings.load(path)
            self.assertEqual(loaded.words, model.words)
            self.assertTrue(np.allclose(loaded.vectors, model.vectors))
            self.assertEqual(loaded.metadata["config"]["architecture"], "cbow")
            self.assertEqual(loaded.similar("cat", 3), model.similar("cat", 3))

    def test_project_and_plot_word_groups(self):
        groups = {"animals": ["cat", "dog", "fox"], "vehicles": ["car", "bus", "van", "zebra"]}
        rows, quality = project(self.models["cbow"], groups)
        self.assertEqual({r["word"] for r in rows}, {"cat", "dog", "fox", "car", "bus", "van"})
        self.assertEqual([q["method"] for q in quality], ["PCA", "t-SNE"])
        with self.assertRaises(ValueError):
            project(self.models["cbow"])                    # the GLP-1 groups are not in this vocabulary
        from bioir.embed import write_projections
        with tempfile.TemporaryDirectory() as folder:
            plot = importlib.util.find_spec("matplotlib") is not None
            paths = write_projections(self.models, folder, plot=plot, groups=groups)
            self.assertEqual([p.name for p in paths],
                             ["projections.csv", "projection_quality.csv"] + (["projections.png"] if plot else []))
            self.assertEqual(write_projections(self.models, Path(folder, "none")), [])

    def test_training_is_deterministic(self):
        settings = dict(dimensions=8, window=2, epochs=2, min_count=1, subsample=0)
        a, b = (train(self.docs[:40], Config(**settings)) for _ in range(2))
        self.assertTrue(np.array_equal(a.vectors, b.vectors))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class CommandLineTest(unittest.TestCase):
    def test_train_similar_and_compare(self):
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder, "toy.jsonl")
            from bioir.corpus import save_jsonl
            save_jsonl(toy_documents(60), corpus)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["embed", "train", "--corpus", str(corpus), "--epochs", "3", "--dims", "8",
                             "--min-count", "1", "-o", str(Path(folder, "toy.npz"))])
                self.assertIn(code, (0, 2))                 # 2 = trained but failed the sanity check
                self.assertEqual(main(["embed", "similar", str(Path(folder, "toy.npz")), "cat", "zebra",
                                       "--min-count", "1"]), 0)
            text = output.getvalue()
            self.assertIn("健全性檢查", text)
            self.assertIn("不在詞彙表中", text)
            from bioir.embed.train import Config as TrainConfig
            summary = compare(toy_documents(40), Path(folder, "cmp"),
                              TrainConfig(dimensions=8, epochs=1, min_count=1, subsample=0))
            self.assertEqual([row["model"] for row in summary],
                             ["cbow_w2_s42", "cbow_w5_s42", "cbow_w5_s7",
                              "skipgram_w2_s42", "skipgram_w5_s42", "skipgram_w5_s7"])
            for name in ("summary", "neighbors", "pairs", "overlap"):
                self.assertTrue(Path(folder, "cmp", f"{name}.csv").exists())
            self.assertFalse(Path(folder, "cmp", "projections.csv").exists())   # GLP-1 words absent: skipped
            import json
            manifest = json.loads(Path(folder, "cmp", "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["corpus"]["documents"], 40)
            self.assertEqual(len(manifest["runs"]), 6)
            self.assertEqual(len(list(Path(folder, "cmp", "models").glob("*.npz"))), 6)


if __name__ == "__main__":
    unittest.main()
