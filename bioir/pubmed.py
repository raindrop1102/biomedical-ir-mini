"""NCBI E-utilities client (standard library only): ESearch for PMIDs, EFetch for abstracts.

Every request goes through request(), which
- spaces requests at least MIN_INTERVAL apart across all threads (NCBI allows
  3 requests/second without an API key),
- retries HTTP 429 / 5xx and network errors with exponential backoff,
- sends its parameters as a POST body and takes the optional API key from the
  NCBI_API_KEY environment variable, so the key never appears in a URL and the
  errors it raises quote neither URL nor key.
"""
import http.client
import json
import os
import re
import threading
import urllib.error
import urllib.parse
from time import monotonic, sleep
from urllib.request import urlopen

from .corpus import parse_pubmed_xml

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH = 200            # PMIDs per EFetch request
MIN_INTERVAL = 0.36    # seconds between any two requests from this process
ATTEMPTS = 5           # tries per request before giving up
TIMEOUT = 30           # seconds; urllib applies it to connecting and to every read
MAX_RESULTS = 9999     # PubMed's ESearch cannot page beyond this

_PUBMED_URL = re.compile(r"https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?(?:[?#]\S*)?", re.I)
_PMID = re.compile(r"[1-9]\d{0,8}")

_lock = threading.Lock()
_last_request = float("-inf")


class PubMedError(OSError):
    """NCBI was unreachable or refused the request."""


def request(endpoint: str, params: dict, email: str) -> bytes:
    """POST one E-utilities request and return the response body."""
    global _last_request
    body = {"db": "pubmed", "tool": "bioir-mini", "email": email, **params}
    if os.environ.get("NCBI_API_KEY"):
        body["api_key"] = os.environ["NCBI_API_KEY"]
    data = urllib.parse.urlencode(body).encode()
    problem = ""
    for attempt in range(ATTEMPTS):
        if attempt:
            sleep(min(2 ** (attempt - 1), 16))       # 1, 2, 4, 8 s
        with _lock:                                  # one shared clock for every thread
            sleep(max(0.0, MIN_INTERVAL - (monotonic() - _last_request)))
            _last_request = monotonic()
        try:
            with urlopen(f"{EUTILS}/{endpoint}.fcgi", data, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            error.close()
            if error.code != 429 and error.code < 500:
                raise PubMedError(f"NCBI {endpoint} 回應 HTTP {error.code}") from None
            problem = f"HTTP {error.code}"
        except (OSError, http.client.HTTPException) as error:   # URLError, timeouts, resets
            # Only the exception type: its text may embed the request.
            problem = type(error).__name__
    raise PubMedError(f"NCBI {endpoint} 試了 {ATTEMPTS} 次仍失敗（{problem}），請稍後再試")


def parse_pmids(text: str) -> list[str]:
    """PMIDs or PubMed article URLs separated by whitespace, commas or semicolons; duplicates dropped."""
    values = [v for v in re.split(r"[\s,;]+", _PUBMED_URL.sub(r"\1", text)) if v]
    bad = [v for v in values if not _PMID.fullmatch(v)]
    if bad:
        raise ValueError(f"不是 PMID 或 PubMed 文章網址：{' '.join(bad[:5])}")
    return list(dict.fromkeys(values))


def esearch(term: str, email: str, count: int = 1000) -> tuple[list[str], dict]:
    """The first `count` PMIDs for a PubMed query, most relevant first,
    plus NCBI's total hit count and how it translated the query."""
    if not term.strip():
        raise ValueError("查詢式不可為空")
    if not 1 <= count <= MAX_RESULTS:
        raise ValueError(f"篇數須介於 1 到 {MAX_RESULTS}")
    payload = json.loads(request("esearch", {"term": term, "sort": "relevance", "retmode": "json",
                                             "retmax": count}, email))
    result = payload.get("esearchresult", {})
    if "error" in payload or "ERROR" in result:
        raise ValueError(f"PubMed 無法解析查詢式：{payload.get('error') or result.get('ERROR')}")
    return result.get("idlist", []), {"total": int(result.get("count", 0)),
                                      "query_translation": result.get("querytranslation", "")}


def fetch(pmids, email: str):
    """Return (documents in input order, PMIDs that had no abstract or were not found)."""
    pmids = parse_pmids(" ".join(pmids))
    found = {}
    for offset in range(0, len(pmids), BATCH):
        xml = request("efetch", {"retmode": "xml", "id": ",".join(pmids[offset:offset + BATCH])}, email)
        for document in parse_pubmed_xml(xml):
            found[document.pmid] = document
    return [found[p] for p in pmids if p in found], [p for p in pmids if p not in found]
