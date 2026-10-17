"""Term weighting in SMART notation.

A scheme is three letters — term frequency, document frequency, normalization —
written separately for documents and queries: "lnc.ltc" weights documents with
log TF, no IDF and cosine normalization, and queries with log TF, IDF and cosine.

    TF   n  raw tf                       l  1 + ln(tf)
         b  1 if the term occurs         a  0.5 + 0.5 · tf / max tf in the text
    IDF  n  1 (none)                     t  ln(N / df)
         p  max(0, ln((N − df) / df))    probabilistic
    norm n  none                         c  divide by the vector's L2 length (cosine)

"bm25" is accepted as a document scheme for reference: Okapi BM25 term weights
(k1 = 1.2, b = 0.75) with binary query weights, i.e. the search page's ranking.
"""
import math
from dataclasses import dataclass

TF_CODES = {"n": "raw TF", "l": "log TF", "b": "binary", "a": "augmented TF"}
IDF_CODES = {"n": "no IDF", "t": "IDF", "p": "probabilistic IDF"}
NORM_CODES = {"n": "no normalization", "c": "L2 (cosine)"}


@dataclass(frozen=True)
class Weighting:
    tf: str
    idf: str
    norm: str

    def __post_init__(self):
        if self.tf not in TF_CODES or self.idf not in IDF_CODES or self.norm not in NORM_CODES:
            raise ValueError(f"Unknown SMART weighting {self.code!r}: TF one of {''.join(TF_CODES)}, "
                             f"IDF one of {''.join(IDF_CODES)}, normalization one of {''.join(NORM_CODES)}")

    @property
    def code(self) -> str:
        return self.tf + self.idf + self.norm

    def describe(self) -> str:
        return f"{TF_CODES[self.tf]} × {IDF_CODES[self.idf]}, {NORM_CODES[self.norm]}"

    def tf_weight(self, tf: int, max_tf: int) -> float:
        if tf <= 0:
            return 0.0
        if self.tf == "n":
            return float(tf)
        if self.tf == "l":
            return 1 + math.log(tf)
        if self.tf == "b":
            return 1.0
        return 0.5 + 0.5 * tf / max_tf

    def idf_weight(self, df: int, documents: int) -> float:
        if self.idf == "n":
            return 1.0
        if self.idf == "t":
            return math.log(documents / df)
        return max(0.0, math.log((documents - df) / df)) if df < documents else 0.0


@dataclass(frozen=True)
class Scheme:
    document: Weighting | None      # None means BM25
    query: Weighting

    @classmethod
    def parse(cls, text: str) -> "Scheme":
        text = text.strip().lower()
        if text == "bm25":
            return cls(None, Weighting("b", "n", "n"))
        parts = text.split(".")
        if len(parts) != 2 or any(len(p) != 3 for p in parts):
            raise ValueError(f"A scheme is 'ddd.qqq' in SMART notation (for example lnc.ltc) or bm25, not {text!r}")
        return cls(Weighting(*parts[0]), Weighting(*parts[1]))

    @property
    def code(self) -> str:
        return "bm25" if self.document is None else f"{self.document.code}.{self.query.code}"

    def describe(self) -> str:
        if self.document is None:
            return "Okapi BM25 (k1 = 1.2, b = 0.75), binary query"
        return f"documents: {self.document.describe()}; queries: {self.query.describe()}"
