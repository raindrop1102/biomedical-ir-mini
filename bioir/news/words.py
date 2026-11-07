"""Word-level analysis for Reuters: Top-5 neighbours and chosen word pairs.

Fixed before training. The words cover the five topics and the vocabulary-mismatch
queries; the pairs mix near-synonyms that should be close (oil / petroleum), words
that share contexts but mean opposite things (profit / loss, import / export), and
words of one topic that are not synonyms (wheat / grain). Close vectors mean the
words appear in similar contexts — not that they mean the same thing, and not that
they would retrieve the same documents.
"""
from ..embed.evaluate import neighbor_rows, pair_rows

WORDS = ["oil", "petroleum", "crude", "opec", "profit", "loss", "dividend", "merger", "takeover", "buyout",
         "wheat", "cereal", "harvest", "tariff", "deficit", "protectionism"]

PAIRS = [("oil", "petroleum"), ("oil", "crude"), ("profit", "loss"), ("profit", "earnings"),
         ("merger", "acquisition"), ("takeover", "buyout"), ("wheat", "grain"), ("wheat", "cereal"),
         ("harvest", "crop"), ("tariff", "protectionism"), ("import", "export"), ("deficit", "surplus"),
         ("supply", "demand"), ("oil", "wheat")]


# Word groups for the PCA / t-SNE figure, one per topic studied.
PROJECTION_GROUPS = {
    "crude": ["oil", "crude", "petroleum", "opec", "barrel", "gasoline"],
    "earn": ["profit", "loss", "dividend", "earnings", "revenues", "quarter"],
    "acq": ["merger", "acquisition", "takeover", "buyout", "stake", "tender"],
    "grain": ["wheat", "corn", "grain", "harvest", "crop", "cereal"],
    "trade": ["tariff", "imports", "exports", "deficit", "surplus", "protectionism"],
}


def word_report(model, k: int = 5) -> tuple[list[dict], list[dict]]:
    """(Top-k neighbour rows for WORDS, cosine and z rows for PAIRS)."""
    return neighbor_rows(model, WORDS, k), pair_rows(model, PAIRS)
