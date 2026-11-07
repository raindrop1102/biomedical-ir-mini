"""Read Reuters-21578 (the 22 reut2-*.sgm files) and keep the ModApte split.

Each story is one <REUTERS ...> ... </REUTERS> block:

    <REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" ... NEWID="1">
    <TOPICS><D>cocoa</D></TOPICS>
    <TEXT><TITLE>BAHIA COCOA REVIEW</TITLE> ... <BODY>Showers continued ... Reuter &#3;</BODY></TEXT>
    </REUTERS>

ModApte (README section VIII.B):
    train  LEWISSPLIT="TRAIN" and TOPICS="YES"   9,603 stories
    test   LEWISSPLIT="TEST"  and TOPICS="YES"   3,299 stories
Everything else (NOT-USED, TOPICS="NO", BYPASS) is left out. Some TOPICS="YES"
stories carry no topic at all; they stay in the split but are relevant to no query.
"""
import html
import re
from dataclasses import dataclass
from pathlib import Path

from ..corpus import Document

STORY = re.compile(r"<REUTERS(?P<attributes>[^>]*)>(?P<content>.*?)</REUTERS>", re.S)
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')          # [^"]* also matches NOT-USED, unlike \w+
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")        # &#2; &#3; ... left after unescaping
SIGNATURE = re.compile(r"\s*\bReuter\s*$", re.I)    # the wire service's sign-off ending most bodies


@dataclass(frozen=True)
class Article:
    newid: str
    split: str                  # "train" or "test"
    title: str
    body: str
    topics: tuple[str, ...]     # e.g. ("crude", "nat-gas"); empty for some ModApte stories

    @property
    def document(self) -> Document:
        """The same story as the Document the search, Zipf and vector-space code read."""
        return Document(self.newid, self.title, self.body)


def _field(content: str, tag: str) -> str:
    match = re.search(fr"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", content, re.S)   # <TEXT TYPE="BRIEF"> has attributes
    return match.group(1) if match else ""


def _body(content: str) -> str:
    """<BODY>, or for TYPE="UNPROC" stories, which have no TITLE or BODY, the whole <TEXT> without tags."""
    text = _field(content, "TEXT")
    return _field(text, "BODY") if "<BODY>" in text or "<TITLE>" in text else re.sub(r"<[^>]+>", " ", text)


def _clean(text: str) -> str:
    text = CONTROL.sub(" ", html.unescape(text))      # &lt; → <, &amp; → &, &#3; → control char → space
    return SIGNATURE.sub("", re.sub(r"\s+", " ", text)).strip()


def parse(text: str) -> list[Article]:
    """Every ModApte story in one .sgm file's text."""
    articles = []
    for story in STORY.finditer(text):
        attributes = dict(ATTRIBUTE.findall(story["attributes"]))
        split = {"TRAIN": "train", "TEST": "test"}.get(attributes.get("LEWISSPLIT"))
        if attributes.get("TOPICS") != "YES" or split is None:
            continue
        content = story["content"]
        topics = tuple(re.findall(r"<D>(.*?)</D>", _field(content, "TOPICS")))
        articles.append(Article(attributes["NEWID"], split, _clean(_field(content, "TITLE")),
                                _clean(_body(content)), topics))
    return articles


def load_reuters(folder) -> list[Article]:
    """All ModApte stories from reut2-000.sgm … reut2-021.sgm, in NEWID order."""
    files = sorted(Path(folder).glob("reut2-*.sgm"))
    if not files:
        raise FileNotFoundError(f"No reut2-*.sgm files in {folder}; see README for the download")
    # The collection predates UTF-8 and contains a few Latin-1 bytes.
    articles = [a for path in files for a in parse(path.read_text(encoding="latin-1"))]
    return sorted(articles, key=lambda a: int(a.newid))
