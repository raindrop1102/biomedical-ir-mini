"""Word2vec — CBOW and Skip-gram with negative sampling — in NumPy.

Defaults follow the ablation of the original BiomedicalIR CBOW, whose vectors had
collapsed onto one direction (random word pairs averaged cosine 0.93):
- not enough training was the cause; batched updates were not (batch 1, 16 and
  128 gave identical vectors). So every CBOW context word receives the full
  hidden-layer gradient, as word2vec does, instead of 1/len(context) of it, and
  training runs 20 epochs;
- embeddings keep stop words and are not stemmed, so contexts stay intact and
  neighbours are readable words; compound tokens ("glp-1") stay whole and pure
  numbers are dropped;
- min_count 3, because words seen twice gave unstable, noisy neighbours.
train() measures the cosine of random word pairs and records whether it passed.
"""
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from ..text import is_number, normalize, sentences, tokens
from .evaluate import sanity
from .model import Embeddings

ARCHITECTURES = ("cbow", "skipgram")


@dataclass(frozen=True)
class Config:
    architecture: str = "cbow"
    dimensions: int = 100
    window: int = 5            # largest distance; each target samples 1..window (word2vec's shrinking)
    epochs: int = 20
    min_count: int = 3
    negatives: int = 5
    learning_rate: float = 0.025   # decays linearly to 0.0001 of itself
    subsample: float = 1e-3        # frequent-word downsampling threshold; 0 turns it off
    batch_size: int = 256
    seed: int = 42
    compound: bool = True          # False: the search tokenizer's split ("glp", "1")

    def validate(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {', '.join(ARCHITECTURES)}")
        for name in ("dimensions", "window", "epochs", "min_count", "negatives", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 < self.learning_rate <= 1 or self.subsample < 0:
            raise ValueError("learning_rate must be in (0, 1] and subsample at least 0")


def corpus_sentences(documents, compound: bool = True) -> list[list[str]]:
    """Abstract sentences as token lists: stop words kept, no stemming, pure numbers dropped."""
    result = []
    for doc in documents:
        for sentence in sentences(doc.abstract):
            words = [normalize(w) for w in tokens(sentence, compound) if not is_number(w)]
            if words:
                result.append(words)
    return result


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def _negative_sampling_step(outputs, hidden, positives, negatives, rate):
    """Update output vectors for one batch; return (summed loss, gradient w.r.t. hidden).

    hidden: (B, d) inputs to the output layer; positives: (B,) target ids;
    negatives: (B, k) sampled ids. A negative that equals its positive is ignored.
    """
    candidates = np.column_stack((positives, negatives))
    valid = np.ones(candidates.shape)
    valid[:, 1:] = negatives != positives[:, None]
    vectors = outputs[candidates]                                  # (B, 1+k, d)
    logits = np.einsum("bd,bkd->bk", hidden, vectors)
    labels = np.zeros_like(logits)
    labels[:, 0] = 1
    loss = float(((np.logaddexp(0, logits) - labels * logits) * valid).sum())
    errors = (_sigmoid(logits) - labels) * valid                   # dLoss/dlogit
    hidden_gradient = np.einsum("bk,bkd->bd", errors, vectors)
    np.add.at(outputs, candidates.ravel(),
              (-rate * errors[:, :, None] * hidden[:, None, :]).reshape(-1, hidden.shape[1]))
    return loss, hidden_gradient


def _contexts(ids, keep, window, rng):
    """(targets, contexts) for one epoch. contexts is (n, 2·window) with −1 padding.
    Sentences are separated by `window` padding cells, so contexts never cross them;
    each target uses a random window 1..window, so nearer words count more often."""
    padded = np.full(sum(len(s) for s in ids) + window * (len(ids) + 1), -1, dtype=np.int64)
    position = window
    for sentence in ids:
        padded[position:position + len(sentence)] = sentence
        position += len(sentence) + window
    if keep is not None:                                            # frequent-word subsampling
        present = padded >= 0
        dropped = rng.random(len(padded)) >= keep[np.maximum(padded, 0)]
        padded[present & dropped] = -1
    centers = np.flatnonzero(padded >= 0)
    offsets = np.array([o for o in range(-window, window + 1) if o])
    contexts = padded[centers[:, None] + offsets[None, :]]
    reach = rng.integers(1, window + 1, len(centers))
    contexts[np.abs(offsets)[None, :] > reach[:, None]] = -1
    has_context = (contexts >= 0).any(axis=1)
    return padded[centers][has_context], contexts[has_context]


def train(documents, config: Config = Config(), progress=None) -> Embeddings:
    """Train on the documents' abstracts. progress(epoch, epochs, loss) is called after each epoch."""
    config.validate()
    started = time.perf_counter()
    corpus = corpus_sentences(documents, config.compound)
    counts = Counter(w for sentence in corpus for w in sentence)
    words = sorted((w for w, c in counts.items() if c >= config.min_count), key=lambda w: (-counts[w], w))
    if len(words) < 2:
        raise ValueError("Fewer than two words reach min_count; load more abstracts or lower min_count.")
    index = {w: i for i, w in enumerate(words)}
    ids = [np.array([index[w] for w in s if w in index], dtype=np.int64) for s in corpus]
    ids = [s for s in ids if len(s) > 1]
    frequency = np.array([counts[w] for w in words], dtype=np.float64)
    total = frequency.sum()
    keep = None
    if config.subsample:
        ratio = frequency / total / config.subsample
        keep = np.minimum(1.0, (np.sqrt(ratio) + 1) / ratio)       # word2vec's keep probability
    noise = np.cumsum(frequency ** 0.75)
    noise /= noise[-1]

    rng = np.random.default_rng(config.seed)
    d = config.dimensions
    inputs = rng.uniform(-0.5 / d, 0.5 / d, (len(words), d))
    outputs = np.zeros((len(words), d))
    history = []
    estimated_steps = None
    step = 0
    for epoch in range(config.epochs):
        targets, contexts = _contexts(ids, keep, config.window, rng)
        if config.architecture == "skipgram":
            rows, cols = np.nonzero(contexts >= 0)
            targets, centers = contexts[rows, cols], targets[rows]     # predict each context word from its center
        order = rng.permutation(len(targets))
        batches = math.ceil(len(order) / config.batch_size)
        estimated_steps = estimated_steps or batches * config.epochs
        epoch_loss = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            rate = config.learning_rate * max(1e-4, 1 - step / estimated_steps)
            step += 1
            negatives = np.minimum(np.searchsorted(noise, rng.random((len(batch), config.negatives)), "right"),
                                   len(words) - 1)
            if config.architecture == "cbow":
                context = contexts[batch]
                mask = context >= 0
                safe = np.maximum(context, 0)
                hidden = (inputs[safe] * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
                loss, gradient = _negative_sampling_step(outputs, hidden, targets[batch], negatives, rate)
                # Full gradient to every context word (word2vec), not gradient / len(context).
                spread = np.broadcast_to((-rate * gradient)[:, None, :], (*context.shape, d))
                np.add.at(inputs, safe[mask], spread[mask])
            else:
                center = centers[batch]
                loss, gradient = _negative_sampling_step(outputs, inputs[center], targets[batch], negatives, rate)
                np.add.at(inputs, center, -rate * gradient)
            epoch_loss += loss
        history.append({"epoch": epoch + 1, "loss": epoch_loss / len(targets), "examples": len(targets)})
        if progress:
            progress(epoch + 1, config.epochs, history[-1]["loss"])

    metadata = {"config": asdict(config), "documents": len(documents), "sentences": len(corpus),
                "tokens": int(sum(counts.values())), "distinct_words": len(counts), "vocabulary": len(words),
                "history": history, "seconds": round(time.perf_counter() - started, 1),
                "embedding": "input vectors",
                "preprocessing": "abstract sentences; lower-case; stop words kept; no stemming; numbers dropped; "
                                 + ("compound tokens" if config.compound else "letters and digits split")}
    model = Embeddings(words, frequency.astype(np.int64), inputs.astype(np.float32), metadata)
    model.metadata["sanity"] = sanity(model)
    return model
