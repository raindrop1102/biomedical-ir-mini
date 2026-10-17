"""2-D projections of selected word vectors: PCA and exact t-SNE, in NumPy.

Vectors are L2-normalized first, so distances follow cosine similarity.
Trustworthiness (Venna & Kaski) says how far the 2-D neighbourhoods can be
believed: 1 means every word's k nearest neighbours on the plot are also near
it in the full space. A projection is a picture of a few words, not a test of
the model, and separate fits (PCA vs t-SNE, or two models) have unrelated axes.
"""
import numpy as np

# Words fixed before training, in groups that differ in kind, so the plot can show
# whether related words land together. Words missing from a vocabulary are skipped.
PROJECTION_GROUPS = {
    "GLP-1 drugs": ["semaglutide", "liraglutide", "exenatide", "dulaglutide", "tirzepatide"],
    "receptor": ["glp-1", "receptor", "agonist", "agonists", "glucagon"],
    "glycaemia": ["insulin", "glucose", "glycemic", "diabetes", "secretion", "pancreatic"],
    "weight & appetite": ["obesity", "weight", "body", "loss", "appetite", "gastric", "emptying"],
}


def _unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _squared_distances(points):
    sums = (points ** 2).sum(1)
    return np.maximum(sums[:, None] + sums[None, :] - 2 * points @ points.T, 0)


def pca(vectors):
    """(coordinates (n, 2), explained variance ratio of the two components)."""
    centered = _unit(vectors) - _unit(vectors).mean(0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    return centered @ components[:2].T, (variance[:2] / variance.sum()).tolist()


def _affinities(distances, perplexity):
    """Symmetric P of t-SNE: a Gaussian per point whose entropy matches log(perplexity)."""
    n = len(distances)
    target = np.log(perplexity)
    p = np.zeros((n, n))
    for i in range(n):
        d = np.delete(distances[i], i)
        low, high, beta = 0.0, np.inf, 1.0
        for _ in range(100):                       # binary search on the precision beta
            weights = np.exp(-(d - d.min()) * beta)
            row = weights / weights.sum()
            entropy = -(row * np.log(np.maximum(row, 1e-12))).sum()
            if abs(entropy - target) < 1e-5:
                break
            if entropy > target:
                low, beta = beta, beta * 2 if high == np.inf else (beta + high) / 2
            else:
                high, beta = beta, (beta + low) / 2
        p[i, np.arange(n) != i] = row
    return (p + p.T) / (2 * n)


def tsne(vectors, perplexity=None, iterations=1000, seed=42):
    """Exact t-SNE (fine for tens of words). Returns (coordinates (n, 2), KL divergence, perplexity)."""
    n = len(vectors)
    if n < 3:
        raise ValueError("t-SNE needs at least three words")
    perplexity = perplexity or min(15.0, max(2.0, (n - 1) / 3))
    p = np.maximum(_affinities(_squared_distances(_unit(vectors)), perplexity), 1e-12)
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1e-4, (n, 2))
    velocity = np.zeros_like(y)
    gains = np.ones_like(y)
    rate = max(n / 12 / 4, 50)                    # scikit-learn's "auto" learning rate
    for step in range(iterations):
        exaggeration, momentum = (12.0, 0.5) if step < 250 else (1.0, 0.8)
        kernel = 1 / (1 + _squared_distances(y))
        np.fill_diagonal(kernel, 0)
        q = np.maximum(kernel / kernel.sum(), 1e-12)
        forces = (exaggeration * p - q) * kernel
        gradient = 4 * (forces.sum(1)[:, None] * y - forces @ y)
        gains = np.where(np.sign(gradient) != np.sign(velocity), gains + 0.2, gains * 0.8).clip(0.01)
        velocity = momentum * velocity - rate * gains * gradient
        y += velocity
        y -= y.mean(0)
    kl = float((p * np.log(p / q)).sum())
    return y, kl, perplexity


def trustworthiness(vectors, coordinates, k=None):
    n = len(vectors)
    k = k or min(5, (n - 1) // 2)
    original = _squared_distances(_unit(vectors))
    np.fill_diagonal(original, np.inf)
    projected = _squared_distances(np.asarray(coordinates, dtype=float))
    np.fill_diagonal(projected, np.inf)
    rank = np.argsort(np.argsort(original, axis=1), axis=1) + 1   # rank of j among i's neighbours, 1 = nearest
    penalty = 0
    for i in range(n):
        near_on_plot = np.argsort(projected[i])[:k]
        penalty += np.maximum(rank[i, near_on_plot] - k, 0).sum()
    return float(1 - 2 / (n * k * (2 * n - 3 * k - 1)) * penalty)


def project(model, groups=None) -> tuple[list[dict], list[dict]]:
    """PCA and t-SNE of the grouped words found in the model.
    Returns (point rows: method, word, group, x, y; quality rows: one per method)."""
    groups = groups or PROJECTION_GROUPS
    words, labels = [], []
    for group, members in groups.items():
        for word in members:
            term = model.term(word)
            if term in model.index and term not in words:
                words.append(term)
                labels.append(group)
    if len(words) < 3:
        raise ValueError("Fewer than three projection words are in the vocabulary")
    vectors = model.vectors[[model.index[w] for w in words]].astype(float)
    k = min(5, (len(words) - 1) // 2)
    coordinates, explained = pca(vectors)
    quality = [{"method": "PCA", "words": len(words), "explained_variance": round(sum(explained), 4),
                "kl_divergence": None, "trustworthiness": round(trustworthiness(vectors, coordinates, k), 4),
                "perplexity": None}]
    points = [("PCA", coordinates)]
    coordinates, kl, perplexity = tsne(vectors)
    quality.append({"method": "t-SNE", "words": len(words), "explained_variance": None,
                    "kl_divergence": round(kl, 4), "trustworthiness": round(trustworthiness(vectors, coordinates, k), 4),
                    "perplexity": round(perplexity, 2)})
    points.append(("t-SNE", coordinates))
    rows = [{"method": method, "word": word, "group": group, "x": round(float(x), 5), "y": round(float(y), 5)}
            for method, xy in points for word, group, (x, y) in zip(words, labels, xy)]
    return rows, quality
