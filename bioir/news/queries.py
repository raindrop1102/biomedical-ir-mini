"""Evaluation queries for Reuters-21578 and their topic-based relevance judgments.

Relevance comes from the collection itself: a test story is relevant to a query when
the query's topic is one of the story's TOPICS. This is binary and topical — a story
filed under crude is relevant to every crude query, whatever its wording.

Lexical coverage — the share of relevant stories that contain at least one query term
after preprocessing — bounds what pure term matching (TF-IDF) can ever retrieve: a
relevant story sharing no term with the query scores 0. Word embeddings are the
attempt to reach those stories.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from ..text import analyze


@dataclass(frozen=True)
class NewsQuery:
    id: str
    text: str
    topic: str
    kind: str          # "keyword" or "paraphrase"

    @property
    def terms(self) -> set[str]:
        """The query as the index sees it: stop words removed, Porter stems."""
        return {t for t, _ in analyze(self.text)}


def load_queries(path) -> list[NewsQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [NewsQuery(q["id"], q["query"], q["topic"], q.get("kind", "keyword")) for q in data["queries"]]


def judgments(articles, query: NewsQuery) -> dict[str, int]:
    """newid → 1 for every test story filed under the query's topic."""
    return {a.newid: 1 for a in articles if a.split == "test" and query.topic in a.topics}


def coverage(articles, query: NewsQuery) -> dict:
    """How many relevant test stories share any / every query term with the query."""
    relevant = [a for a in articles if a.split == "test" and query.topic in a.topics]
    terms = query.terms
    story_terms = [{t for t, _ in analyze(f"{a.title} . {a.body}")} for a in relevant]
    return {"relevant": len(relevant),
            "any_term": sum(bool(terms & s) for s in story_terms),
            "all_terms": sum(terms <= s for s in story_terms),
            "no_term": sum(not terms & s for s in story_terms)}
