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

# What each pair is, decided with the pairs (before training). Near-synonyms and opposites are the
# contrast that matters: a cosine near 1 says "used in the same contexts", which opposites also are.
RELATIONS = {
    ("oil", "petroleum"): "近義", ("oil", "crude"): "近義", ("profit", "earnings"): "近義",
    ("merger", "acquisition"): "近義", ("takeover", "buyout"): "近義",
    ("profit", "loss"): "反義／相對", ("import", "export"): "反義／相對", ("deficit", "surplus"): "反義／相對",
    ("supply", "demand"): "反義／相對",
    ("wheat", "grain"): "上下位", ("wheat", "cereal"): "上下位",
    ("harvest", "crop"): "相關", ("tariff", "protectionism"): "相關",
    ("oil", "wheat"): "無關（對照）",
}
RELATION_ORDER = ["近義", "反義／相對", "上下位", "相關", "無關（對照）"]


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
