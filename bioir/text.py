"""Text analysis shared by search, Zipf and embeddings: tokenizing, normalization, sentences.

Every run of letters or of digits is its own token, so hyphens, slashes and
letter/digit boundaries all split: "GLP-1RAs", "GLP1 RAs" and "glp-1 ras" all
become glp · 1 · ra at consecutive positions. Query words are matched as
phrases over those parts (see query.py), so every spelling variant of a
biomedical identifier lands on the same postings.
"""
import html
import re
from typing import Iterator

from .porter import stem

TOKEN = re.compile(r"[^\W\d_]+(?:['’]s\b)?|\d+")
COMPOUND = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")   # the original project's tokenizer
# A number with optional grouping, decimals, percent sign and surrounding punctuation:
# "2", "0.05", "10,000", "45%", "(2)". Words that merely contain digits ("glp-1", "p53") are not numbers.
NUMBER = re.compile(r"\W*\d[\d.,]*%?\W*")
# A sentence ends at . ! or ? followed by whitespace and a capital, digit or bracket;
# "p < 0.05 and" or "2.5 mg" do not end one. Abbreviations such as "e.g. The" may still split.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

STOP_WORDS = frozenset("""
a an and are as at be been being but by for from had has have he her his if in into
is it its of on or she such that the their them then there these they this those to
was were which while will with
""".split())


def tokenize(text: str) -> Iterator[re.Match]:
    """Raw token matches, so callers can use their character spans."""
    return TOKEN.finditer(text)


def normalize(token: str) -> str:
    """Case-fold and drop a possessive 's: "Patient's" → "patient"."""
    token = token.casefold()
    return token[:-2] if token.endswith(("'s", "’s")) else token


def analyze(text: str, start: int = 0) -> list[tuple[str, int]]:
    """(term, position) pairs. Stop words are dropped but keep their position,
    so phrases never match across a removed word."""
    pairs = []
    for position, match in enumerate(tokenize(text), start):
        word = normalize(match.group())
        if word not in STOP_WORDS:
            pairs.append((stem(word), position))
    return pairs


def analyze_spans(text: str) -> list[tuple[str, int, int, int]]:
    """analyze() plus each term's character span: (term, position, start, end)."""
    spans = []
    for position, match in enumerate(tokenize(text)):
        word = normalize(match.group())
        if word not in STOP_WORDS:
            spans.append((stem(word), position, match.start(), match.end()))
    return spans


def highlight(text: str, spans) -> str:
    """HTML for text with every (start, end) span inside <mark>. All text is escaped,
    so an abstract can never inject markup; overlapping spans are merged."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        elif start < end:
            merged.append([start, end])
    out, position = [], 0
    for start, end in merged:
        out += [html.escape(text[position:start]), "<mark>", html.escape(text[start:end]), "</mark>"]
        position = end
    out.append(html.escape(text[position:]))
    return "".join(out)


def tokens(text: str, compound: bool = False) -> list[str]:
    """Normalized tokens before stop-word removal and stemming.

    compound=False is the tokenizer analyze() uses. compound=True keeps
    hyphen/apostrophe compounds whole ("glp-1ras", "patient's"), as the original
    BiomedicalIR did, for analyses that must be comparable with it.
    """
    if compound:
        return COMPOUND.findall(text.casefold())
    return [normalize(match.group()) for match in tokenize(text)]


def is_number(token: str) -> bool:
    return NUMBER.fullmatch(token) is not None


def sentences(text: str) -> list[str]:
    return [part for part in SENTENCE_END.split(text) if part.strip()]


def words(text: str) -> Iterator[str]:
    """Alphabetic surface words (unstemmed) for the spelling dictionary."""
    for match in tokenize(text):
        word = normalize(match.group())
        if word.isalpha():
            yield word
