"""Term weighting in SMART notation.

A scheme is three letters — term frequency, document frequency, normalization —
written separately for documents and queries: "lnc.ltc" weights documents with
log TF, no IDF and cosine normalization, and queries with log TF, IDF and cosine.
Letters are case-sensitive, as in SMART: "L" and "l" are different TF weights.

    TF   n  raw tf                       l  1 + log(tf)
         b  1 if the term occurs         a  0.5 + 0.5 · tf / max tf in the text
         L  (1 + log tf) / (1 + log ave tf), ave tf = tokens / distinct terms in the text
    IDF  n  1 (none)                     t  log(N / df)
         p  max(0, log((N − df) / df))   probabilistic
    norm n  none                         c  divide by the vector's L2 length (cosine)
         u  pivoted unique: divide by (1 − s) · pivot + s · distinct terms, where the
            pivot is the corpus average of distinct terms per document and s = 0.2
         b  byte size: divide by the text's length in characters ** α, α = 0.375

Logs are base 10, as in the lecture, so its worked examples come out exactly. The
base only rescales IDF by a constant, which L2 normalization cancels, but it does
change the shape of 1 + log(tf).

Only the document side's normalization changes a ranking: a query's divisor is the
same for every document.

These are exactly the variants in Lecture 3's SMART table; the lecture gives u only
as "1/u", so the pivot and slope are Singhal, Buckley & Mitra's, and α < 1 is ours.
"""
import math
from dataclasses import dataclass

TF_CODES = {"n": "raw TF", "l": "log TF", "b": "binary", "a": "augmented TF", "L": "log average TF"}
IDF_CODES = {"n": "no IDF", "t": "IDF", "p": "probabilistic IDF"}
NORM_CODES = {"n": "no normalization", "c": "L2 (cosine)", "u": "pivoted unique", "b": "byte size"}

PIVOT_SLOPE = 0.2           # Singhal, Buckley & Mitra (1996)
BYTE_SIZE_ALPHA = 0.375     # any α < 1 penalizes length less than proportionally


@dataclass(frozen=True)
class Weighting:
    tf: str
    idf: str
    norm: str
    base: float = 10

    def __post_init__(self):
        if self.tf not in TF_CODES or self.idf not in IDF_CODES or self.norm not in NORM_CODES:
            raise ValueError(f"Unknown SMART weighting {self.code!r}: TF one of {''.join(TF_CODES)}, "
                             f"IDF one of {''.join(IDF_CODES)}, normalization one of {''.join(NORM_CODES)}")

    @property
    def code(self) -> str:
        return self.tf + self.idf + self.norm

    def describe(self) -> str:
        return f"{TF_CODES[self.tf]} × {IDF_CODES[self.idf]}, {NORM_CODES[self.norm]}"

    def _log(self, x: float) -> float:
        return math.log(x, self.base)

    def tf_weight(self, tf: int, max_tf: int, average_tf: float = 1.0) -> float:
        if tf <= 0:
            return 0.0
        if self.tf == "n":
            return float(tf)
        if self.tf == "l":
            return 1 + self._log(tf)
        if self.tf == "L":
            return (1 + self._log(tf)) / (1 + self._log(average_tf))
        if self.tf == "b":
            return 1.0
        return 0.5 + 0.5 * tf / max_tf

    def idf_weight(self, df: int, documents: int) -> float:
        if self.idf == "n":
            return 1.0
        if self.idf == "t":
            return self._log(documents / df)
        return max(0.0, self._log((documents - df) / df)) if df < documents else 0.0

    def divisor(self, weights, distinct: int, characters: int, pivot: float) -> float:
        """What a text's term weights are divided by: its normalization factor."""
        if self.norm == "c":
            return math.sqrt(sum(w * w for w in weights))
        if self.norm == "u":
            return (1 - PIVOT_SLOPE) * pivot + PIVOT_SLOPE * distinct
        if self.norm == "b":
            return characters ** BYTE_SIZE_ALPHA
        return 1.0


@dataclass(frozen=True)
class Scheme:
    document: Weighting
    query: Weighting

    @classmethod
    def parse(cls, text: str) -> "Scheme":
        parts = text.strip().split(".")
        if len(parts) != 2 or any(len(p) != 3 for p in parts):
            raise ValueError(f"A scheme is 'ddd.qqq' in SMART notation (for example lnc.ltc), not {text!r}")
        return cls(Weighting(*parts[0]), Weighting(*parts[1]))

    @property
    def code(self) -> str:
        return f"{self.document.code}.{self.query.code}"

    def describe(self) -> str:
        return f"documents: {self.document.describe()}; queries: {self.query.describe()}"
