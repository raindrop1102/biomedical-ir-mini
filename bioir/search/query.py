"""Boolean query parser.

    expr   := and ("OR" and)*
    and    := unary (["AND"] unary)*          adjacent words mean AND
    unary  := "NOT" unary | "(" expr ")" | WORD | "QUOTED PHRASE"

Operators are upper-case only, so "or"/"not" in lower case are ordinary words.
Each WORD is a phrase over its analyzed parts ("GLP-1" → glp 1).
Queries over MAX_QUERY_CHARS characters, MAX_WORDS words or with a word over
MAX_WORD_CHARS characters are rejected before parsing.

The parse tree uses tuples:
    ("phrase", ((term, offset), ...)) | ("and", [nodes]) | ("or", [nodes]) | ("not", node)
"""
import re

from ..text import analyze

_LEXER = re.compile(r'"[^"]*"?|[()]|[^\s()"]+')
OPERATORS = frozenset({"AND", "OR", "NOT"})
MAX_QUERY_CHARS, MAX_WORDS, MAX_WORD_CHARS = 500, 20, 50


def _check_limits(query: str, tokens: list[str]) -> None:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Query is longer than {MAX_QUERY_CHARS} characters")
    words = [w for t in tokens if t not in OPERATORS for w in t.strip('"()').split()]
    if len(words) > MAX_WORDS:
        raise ValueError(f"Query has more than {MAX_WORDS} words")
    long = next((w for w in words if len(w) > MAX_WORD_CHARS), None)
    if long:
        raise ValueError(f"Word longer than {MAX_WORD_CHARS} characters: {long[:20]}…")


def parse(query: str):
    """Return the parse tree, or None when the query has no searchable terms."""
    tokens = _LEXER.findall(query)
    _check_limits(query, tokens)
    i = 0

    def peek():
        return tokens[i] if i < len(tokens) else None

    def take():
        nonlocal i
        i += 1
        return tokens[i - 1]

    def combine(kind, nodes):
        nodes = [n for n in nodes if n is not None]
        return None if not nodes else nodes[0] if len(nodes) == 1 else (kind, nodes)

    def parse_or():
        nodes = [parse_and()]
        while peek() == "OR":
            take()
            nodes.append(parse_and())
        return combine("or", nodes)

    def parse_and():
        nodes = []
        while peek() not in (None, "OR", ")"):
            if peek() == "AND":
                take()
            else:
                nodes.append(parse_unary())
        return combine("and", nodes)

    def parse_unary():
        token = take()
        if token == "NOT":
            if peek() in (None, "OR", ")", "AND"):
                raise ValueError("NOT needs something to exclude")
            child = parse_unary()
            return ("not", child) if child is not None else None
        if token == "(":
            node = parse_or()
            if peek() != ")":
                raise ValueError("Missing ')'")
            take()
            return node
        pairs = analyze(token.strip('"'))
        if not pairs:
            return None
        first = pairs[0][1]
        return ("phrase", tuple((term, position - first) for term, position in pairs))

    tree = parse_or()
    if i < len(tokens):
        raise ValueError(f"Unexpected '{tokens[i]}'")
    return tree


def positive_terms(node) -> set[str]:
    """Terms that contribute to ranking: everything outside a NOT."""
    if node is None or node[0] == "not":
        return set()
    if node[0] == "phrase":
        return {term for term, _ in node[1]}
    return set().union(*map(positive_terms, node[1]))


def positive_phrases(node) -> list[tuple]:
    """The ((term, offset), ...) phrases outside any NOT: what a hit should highlight."""
    if node is None or node[0] == "not":
        return []
    if node[0] == "phrase":
        return [node[1]]
    return list(dict.fromkeys(p for child in node[1] for p in positive_phrases(child)))
