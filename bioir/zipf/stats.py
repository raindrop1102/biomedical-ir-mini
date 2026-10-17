"""Rank-frequency statistics and Zipf fits under four cumulative preprocessing conditions.

    A  lower-case, split on whitespace (punctuation stays attached)
    B  A + tokenizer (bioir.text.tokens)
    C  B + stop-word removal
    D  C + Porter stemming    — the terms the search index uses

Only the abstract is analyzed. Ranks: collection frequency descending, ties
broken by term ascending. The fit is ordinary least squares of
log10(cf) = intercept + slope · log10(rank); exponent = −slope. It is made over
every rank and, separately, over three fixed regions: the first 10% of ranks
(head), the next 80% (middle) and the last 10% (tail).
"""
import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..porter import stem
from ..text import STOP_WORDS, is_number, normalize, tokens

CONDITIONS = {
    "A": "lower-case + whitespace split",
    "B": "A + tokenizer",
    "C": "B + stop-word removal",
    "D": "C + Porter stemming",
}

@dataclass(frozen=True)
class Term:
    rank: int
    term: str
    cf: int        # collection frequency
    df: int        # document frequency
    idf: float     # ln(N / df)

    @property
    def cf_per_df(self) -> float:
        """Average occurrences in the documents that contain the term."""
        return self.cf / self.df


@dataclass(frozen=True)
class Fit:
    slope: float
    intercept: float
    exponent: float
    r2: float | None   # None when every point has the same frequency
    rmse: float        # in log10(cf) units
    points: int
    rank_start: int
    rank_end: int


@dataclass(frozen=True)
class Region:
    name: str                    # head / middle / tail
    rank_start: int
    rank_end: int
    terms: int
    distinct_frequencies: int    # few distinct values = a staircase; one = a flat plateau with no R²
    fit: Fit | None


@dataclass
class ConditionStats:
    condition: str
    documents: int
    tokens: int
    vocabulary: int
    hapax: int                   # terms occurring once in the collection
    fit: Fit | None
    regions: list[Region]
    terms: list[Term]

    @property
    def average_tokens(self) -> float:
        return self.tokens / self.documents if self.documents else 0.0

    @property
    def type_token_ratio(self) -> float:
        return self.vocabulary / self.tokens if self.tokens else 0.0


def preprocess(text: str, condition: str, compound: bool = False, numbers: bool = True) -> list[str]:
    """Terms of one text under a condition; D equals the search index's terms.
    numbers=False drops pure-number tokens in every condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; use one of {', '.join(CONDITIONS)}")
    words = text.lower().split() if condition == "A" else tokens(text, compound)
    if not numbers:
        words = [w for w in words if not is_number(w)]
    if condition in "CD":
        words = [w for w in words if w not in STOP_WORDS]
    # normalize() is a no-op on split tokens; on compound ones it drops a possessive 's first.
    return [stem(normalize(w)) for w in words] if condition == "D" else words


def fit_zipf(terms: list[Term]) -> Fit | None:
    """OLS of log10(cf) on log10(rank)."""
    n = len(terms)
    if n < 2:
        return None
    xs = [math.log10(t.rank) for t in terms]
    ys = [math.log10(t.cf) for t in terms]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    sse = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    return Fit(slope, intercept, -slope + 0.0, 1 - sse / sst if sst else None, math.sqrt(sse / n),
               n, terms[0].rank, terms[-1].rank)


def regions(terms: list[Term]) -> list[Region]:
    """Head (first 10% of ranks, at least one), middle (next 80%), tail (last 10%); empty parts omitted."""
    n = len(terms)
    head_end = max(1, math.ceil(n * 0.1))
    tail_start = max(head_end, math.floor(n * 0.9))
    parts = [("head", terms[:head_end]), ("middle", terms[head_end:tail_start]), ("tail", terms[tail_start:])]
    return [Region(name, part[0].rank, part[-1].rank, len(part), len({t.cf for t in part}), fit_zipf(part))
            for name, part in parts if part]


def analyze_condition(documents, condition: str, compound: bool = False, numbers: bool = True) -> ConditionStats:
    cf, df, total = Counter(), Counter(), 0
    for doc in documents:
        terms = preprocess(doc.abstract, condition, compound, numbers)
        cf.update(terms)
        df.update(set(terms))
        total += len(terms)
    n = len(documents)
    ranked = sorted(cf.items(), key=lambda item: (-item[1], item[0]))
    rows = [Term(rank, term, count, df[term], math.log(n / df[term]))
            for rank, (term, count) in enumerate(ranked, 1)]
    return ConditionStats(condition, n, total, len(rows), sum(t.cf == 1 for t in rows),
                          fit_zipf(rows), regions(rows), rows)


def analyze_corpus(documents, compound: bool = False, numbers: bool = True) -> dict[str, ConditionStats]:
    return {c: analyze_condition(documents, c, compound, numbers) for c in CONDITIONS}


def compare_terms(stats: ConditionStats, words, compound: bool = False, numbers: bool = True) -> list[dict]:
    """CF / DF / IDF rows for chosen words, each analyzed under the same condition as `stats`
    ("Patients" becomes "patient" under D; "GLP-1" becomes glp and 1 with the split tokenizer).
    A word that yields no term, or a term not in the corpus, gets empty statistics."""
    lookup = {t.term: t for t in stats.terms}
    rows = []
    for word in dict.fromkeys(words):
        for term in preprocess(word, stats.condition, compound, numbers) or [None]:
            found = lookup.get(term)
            rows.append({"word": word, "term": term, "rank": found and found.rank, "cf": found and found.cf,
                         "df": found and found.df, "cf_per_df": found and found.cf_per_df,
                         "idf": found and found.idf})
    return rows


SUMMARY_FIELDS = ["condition", "documents", "tokens", "vocabulary", "average_tokens",
                  "type_token_ratio", "hapax"] + [f.name for f in fields(Fit)]
TERM_FIELDS = ["rank", "term", "cf", "df", "cf_per_df", "idf"]
REGION_FIELDS = ["condition", "region", "rank_start", "rank_end", "terms", "distinct_frequencies",
                 "slope", "intercept", "exponent", "r2", "rmse"]


def summary_row(stats: ConditionStats) -> dict:
    row = {name: getattr(stats, name) for name in SUMMARY_FIELDS[:7]}
    return row | (asdict(stats.fit) if stats.fit else dict.fromkeys(SUMMARY_FIELDS[7:]))


def region_row(condition: str, region: Region) -> dict:
    fit = asdict(region.fit) if region.fit else {}
    return {"condition": condition, "region": region.name, "rank_start": region.rank_start,
            "rank_end": region.rank_end, "terms": region.terms,
            "distinct_frequencies": region.distinct_frequencies} | {k: fit.get(k) for k in REGION_FIELDS[6:]}


def _write(path: Path, fieldnames, rows) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_csv(results: dict[str, ConditionStats], folder, comparison: list[dict] | None = None) -> list[Path]:
    """summary.csv, regions.csv, terms_<condition>.csv (every term) and, when given,
    compare_terms.csv; returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [_write(folder / "summary.csv", SUMMARY_FIELDS, map(summary_row, results.values())),
             _write(folder / "regions.csv", REGION_FIELDS,
                    (region_row(c, r) for c, stats in results.items() for r in stats.regions))]
    for condition, stats in results.items():
        paths.append(_write(folder / f"terms_{condition}.csv", TERM_FIELDS,
                            (asdict(t) | {"cf_per_df": t.cf_per_df} for t in stats.terms)))
    if comparison:
        paths.append(_write(folder / "compare_terms.csv", list(comparison[0]), comparison))
    return paths
