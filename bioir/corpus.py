"""Documents, the JSONL / CSV / PubMed-XML formats they are stored in, and corpus manifests."""
import csv
import hashlib
import json
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    pmid: str
    title: str
    abstract: str
    mesh: tuple[str, ...] = ()          # MeSH descriptors assigned by NLM indexers (empty if not indexed)
    mesh_major: tuple[str, ...] = ()    # the subset marked as a major topic of the article


def _clean(text) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _text(element) -> str:
    return _clean(" ".join(element.itertext())) if element is not None else ""


def _section(element) -> str:
    text = _text(element)
    label = element.get("Label")
    return f"{label}: {text}" if text and label else text


def _mesh(citation) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(all descriptors, major-topic descriptors). A heading is a major topic when its
    descriptor or any of its qualifiers carries MajorTopicYN="Y"."""
    names, major = [], []
    for heading in citation.findall("MeshHeadingList/MeshHeading"):
        descriptor = heading.find("DescriptorName")
        name = _text(descriptor)
        if not name:
            continue
        names.append(name)
        flags = [descriptor.get("MajorTopicYN")] + [q.get("MajorTopicYN") for q in heading.findall("QualifierName")]
        if "Y" in flags:
            major.append(name)
    return tuple(names), tuple(major)


def _terms(value) -> tuple[str, ...]:
    """MeSH as stored: a JSON list, or "A; B" in a CSV cell."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if isinstance(value, str) and value.strip():
        return tuple(v.strip() for v in value.split(";") if v.strip())
    return ()


def parse_pubmed_xml(data: bytes) -> list[Document]:
    """PubmedArticleSet (EFetch output) → documents. Structured abstract sections
    keep their label ("BACKGROUND: ...") and are joined with spaces; MeSH headings are kept."""
    # ElementTree expands internal entities. EFetch output never declares any, so refuse
    # them instead of relying only on expat's (>= 2.4.1) entity-amplification limits.
    if b"<!ENTITY" in data:
        raise ValueError("XML declares entities; refusing to parse it")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"Malformed XML: {error}") from None
    errors = [_text(e) for e in root.iter("ERROR")]
    if errors:
        raise ValueError("NCBI: " + "; ".join(errors))
    documents = []
    for citation in root.iter("MedlineCitation"):
        pmid = _text(citation.find("PMID"))
        sections = citation.findall("Article/Abstract/AbstractText")
        abstract = " ".join(filter(None, map(_section, sections)))
        if pmid and abstract:
            documents.append(Document(pmid, _text(citation.find("Article/ArticleTitle")), abstract,
                                      *_mesh(citation)))
    return documents


def manifest_path(path) -> Path:
    """abstracts.jsonl → abstracts.manifest.json"""
    path = Path(path)
    return path.with_name(f"{path.stem}.manifest.json")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path, **info) -> dict:
    """Write where a corpus file came from (query, time, counts) and its SHA-256 next to it."""
    manifest = {**info, "file": Path(path).name, "sha256": sha256(path)}
    manifest_path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _check_manifest(path: Path) -> None:
    """Warn when the corpus no longer matches the SHA-256 its manifest recorded."""
    target = manifest_path(path)
    if not target.exists():
        return
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
        if manifest.get("file", path.name) != path.name:
            return                                   # describes a sibling with the same stem
        expected = manifest["sha256"]
    except (OSError, ValueError, KeyError, AttributeError):
        warnings.warn(f"{target.name} has no readable sha256; integrity not checked", stacklevel=3)
        return
    if expected != sha256(path):
        warnings.warn(f"{path.name} does not match the SHA-256 in {target.name}: "
                      "the file changed or was damaged after it was downloaded", stacklevel=3)


def load(path) -> list[Document]:
    """Read .jsonl, .csv (pmid, title, abstract columns) or PubMed .xml; first copy of a PMID wins.
    A neighbouring manifest (see write_manifest) is checked and a mismatch is warned about."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml":
        documents = parse_pubmed_xml(path.read_bytes())
    elif suffix in (".jsonl", ".csv"):
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows = csv.DictReader(file) if suffix == ".csv" else (json.loads(l) for l in file if l.strip())
            try:
                documents = [Document(str(r["pmid"]).strip(), _clean(r.get("title")), _clean(r["abstract"]),
                                      _terms(r.get("mesh")), _terms(r.get("mesh_major"))) for r in rows]
            except KeyError as error:
                raise ValueError(f"{path.name}: every record needs a {error} field") from None
    else:
        raise ValueError(f"Unsupported corpus format: {path.name} (use .jsonl, .csv or .xml)")
    _check_manifest(path)
    unique = {}
    for document in documents:
        unique.setdefault(document.pmid, document)
    return list(unique.values())


def save_jsonl(documents, path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for document in documents:
            # Files without MeSH keep the original three fields.
            record = {k: v for k, v in asdict(document).items() if v or k in ("pmid", "title", "abstract")}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
