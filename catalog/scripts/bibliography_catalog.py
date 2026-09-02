#!/usr/bin/env python3
"""Shared, dependency-free helpers for the versioned bibliography catalog."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO / "catalog" / "bibliography" / "methods_bibliography_v1.json"
LEGACY_PATH = REPO / "docs" / "_extras" / "methods_bibliography.py"
REVIEWED_BIB_PATH = REPO / "docs" / "methods" / "_finetuning_bibliography.bib"
SOFTWARE_CFF_PATH = REPO / "CITATION.cff"
PARITY_PATHS = (
    Path("parity/REFERENCES.md"),
    Path("benchmarks/parity_timing/truth_sources.lock.json"),
)
SOURCE_PATHS = (
    LEGACY_PATH.relative_to(REPO),
    REVIEWED_BIB_PATH.relative_to(REPO),
    SOFTWARE_CFF_PATH.relative_to(REPO),
    *PARITY_PATHS,
)
UNKNOWN_PUBLICATION_FIELDS = (
    "authors",
    "title",
    "year",
    "journal",
    "doi",
    "url",
    "bibtex",
)


class ProvenanceError(ValueError):
    """A recorded Git provenance object cannot be resolved locally."""


def sha256_text(text: str) -> str:
    """Return the UTF-8 content digest used by the JSON provenance records."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for both catalog and generated artifacts."""
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def source_revision() -> str:
    """Resolve the last revision that changed an imported baseline source.

    Pinning the source revision, rather than the current checkout revision,
    keeps ``--check`` stable after unrelated commits and makes the provenance
    a meaningful rollback target.
    """
    result = subprocess.run(
        [
            "git",
            "log",
            "-1",
            "--format=%H",
            "--",
            *(str(path) for path in SOURCE_PATHS),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if not revision:
        raise ProvenanceError(
            "could not resolve a source revision for bibliography inputs"
        )
    return resolve_revision(revision)


def resolve_revision(revision: str, *, repo: Path = REPO) -> str:
    """Resolve a commit and emit a deterministic shallow-clone diagnostic."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ProvenanceError(
            f"bibliography source revision {revision!r} is unavailable in {repo}; "
            "fetch full history (actions/checkout fetch-depth: 0) before migrating or validating"
        )
    return result.stdout.strip()


def git_blob_text(revision: str, path: Path, *, repo: Path = REPO) -> str:
    """Read one repository-relative source exactly as it existed at ``revision``."""
    if path.is_absolute():
        try:
            path = path.relative_to(repo)
        except ValueError as exc:
            raise ProvenanceError(f"source path is outside repository: {path}") from exc
    resolved = resolve_revision(revision, repo=repo)
    source_name = path.as_posix()
    result = subprocess.run(
        ["git", "show", f"{resolved}:{source_name}"],
        cwd=repo,
        capture_output=True,
    )
    if result.returncode:
        raise ProvenanceError(
            f"bibliography source {source_name!r} is unavailable at {resolved}; "
            "fetch full history (actions/checkout fetch-depth: 0) before migrating or validating"
        )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceError(
            f"bibliography source {source_name!r} at {resolved} is not UTF-8"
        ) from exc


def extract_legacy_bibliography_text(
    text: str, *, source_name: str = str(LEGACY_PATH)
) -> dict[str, dict[str, Any]]:
    """Load the historical literal dictionary without importing its doc code."""
    tree = ast.parse(text, filename=source_name)
    for node in tree.body:
        target = getattr(node, "target", None)
        if isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
        if isinstance(target, ast.Name) and target.id == "BIBLIOGRAPHY":
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                break
            return value
    raise ValueError(f"{source_name}: BIBLIOGRAPHY literal not found")


def extract_legacy_bibliography(path: Path = LEGACY_PATH) -> dict[str, dict[str, Any]]:
    """Load a local fixture or source file without executing documentation code."""
    return extract_legacy_bibliography_text(
        path.read_text(encoding="utf-8"), source_name=str(path)
    )


def _consume_braced(text: str, start: int) -> tuple[str, int]:
    """Return a BibTeX braced value and the offset after its closing brace."""
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected a braced BibTeX value")
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("unterminated braced BibTeX value")


def extract_reviewed_bibtex_entries(text: str) -> dict[str, dict[str, Any]]:
    """Extract the reviewed BibTeX entries while retaining each exact source slice.

    This deliberately does not normalize TeX accents, author spelling, or missing
    metadata.  ``bibtex`` is invoked by the regression test as the authoritative
    syntax parser; this small extractor only identifies the reviewed source entry
    and its literal field values for provenance checking.
    """
    entries: dict[str, dict[str, Any]] = {}
    start = 0
    header = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{")
    field_header = re.compile(r"\s*,?\s*(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*=\s*")
    while match := header.search(text, start):
        body, end = _consume_braced(text, match.end() - 1)
        start = end
        key, separator, fields_text = body.partition(",")
        key = key.strip()
        if not separator or not key:
            raise ValueError("reviewed BibTeX entry is missing a citation key")
        if key in entries:
            raise ValueError(f"duplicate reviewed BibTeX citation key: {key}")
        fields: dict[str, str] = {}
        offset = 0
        while offset < len(fields_text):
            field_match = field_header.match(fields_text, offset)
            if not field_match:
                if fields_text[offset:].strip(" \t\r\n,"):
                    raise ValueError(f"{key}: malformed reviewed BibTeX field")
                break
            name = field_match.group("name").lower()
            value_start = field_match.end()
            if value_start >= len(fields_text):
                raise ValueError(f"{key}: missing value for {name}")
            if fields_text[value_start] == "{":
                value, offset = _consume_braced(fields_text, value_start)
            elif fields_text[value_start] == '"':
                quote_end = fields_text.find('"', value_start + 1)
                if quote_end < 0:
                    raise ValueError(f"{key}: unterminated quoted value for {name}")
                value = fields_text[value_start + 1 : quote_end]
                offset = quote_end + 1
            else:
                raw_value = fields_text[value_start:].split(",", 1)[0]
                value = raw_value.strip()
                offset = value_start + len(raw_value)
            fields[name] = value.strip()
        raw_entry = text[match.start() : end]
        entries[key] = {
            "entry_type": match.group("type").lower(),
            "fields": fields,
            "raw_entry": raw_entry,
        }
    if not entries:
        raise ValueError("reviewed BibTeX source contains no entries")
    return entries


def _reviewed_publication(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Map literal reviewed BibTeX fields without filling in absent facts."""
    fields = entry["fields"]
    authors = fields.get("author")
    title = fields.get("title")
    year = fields.get("year")
    if not authors or not title or not year or not year.isdigit():
        raise ValueError(f"{key}: reviewed BibTeX needs author, title, and numeric year")
    structured_values = {
        "authors": authors,
        "title": title,
        "year": year,
        "journal": fields.get("journal") or fields.get("booktitle"),
        "doi": fields.get("doi"),
        "url": fields.get("url"),
        "bibtex": entry["raw_entry"],
    }
    return {
        "id": f"reviewed-finetuning-{key}",
        "method_key": None,
        "method_title": None,
        "legacy_citation": None,
        "authors": [part.strip() for part in re.split(r"\s+and\s+", authors) if part.strip()],
        "title": title,
        "year": int(year),
        "journal": structured_values["journal"],
        "doi": structured_values["doi"],
        "url": structured_values["url"],
        "bibtex": structured_values["bibtex"],
        "provenance": {
            "source": "docs/methods/_finetuning_bibliography.bib",
            "source_sha256": None,  # Filled once the source blob has been digested.
            "entry_key": key,
            "entry_sha256": sha256_text(entry["raw_entry"]),
            "fields_status": "reviewed_structured",
            "missing_fields": [
                field for field, value in structured_values.items() if value is None
            ],
        },
    }


def build_catalog(*, revision: str | None = None) -> dict[str, Any]:
    """Migrate the legacy public strings losslessly into bibliography schema v1.

    The old source is prose, not a bibliography database.  Therefore no parser
    guesses author lists, venues, years, DOI/URL values, or BibTeX.  Each is
    represented by ``null`` plus an explicit provenance status; the original
    citation string remains the lossless source for later reviewed enrichment.
    """
    resolved_revision = resolve_revision(revision or source_revision())
    legacy_text = git_blob_text(resolved_revision, LEGACY_PATH)
    legacy = extract_legacy_bibliography_text(
        legacy_text,
        source_name=f"{resolved_revision}:docs/_extras/methods_bibliography.py",
    )
    software_cff = git_blob_text(resolved_revision, SOFTWARE_CFF_PATH)
    reviewed_bib = git_blob_text(resolved_revision, REVIEWED_BIB_PATH)
    publications: list[dict[str, Any]] = []
    for key in sorted(legacy):
        entry = legacy[key]
        paper = entry.get("paper")
        title = entry.get("title")
        if not isinstance(paper, str) or not paper or not isinstance(title, str):
            raise ValueError(f"{LEGACY_PATH}: {key!r} has no literal paper/title")
        publications.append(
            {
                "id": f"legacy-{key}",
                "method_key": key,
                "method_title": title,
                "legacy_citation": paper,
                "authors": None,
                "title": None,
                "year": None,
                "journal": None,
                "doi": None,
                "url": None,
                "bibtex": None,
                "provenance": {
                    "source": f"docs/_extras/methods_bibliography.py:BIBLIOGRAPHY.{key}.paper",
                    "fields_status": "unstructured_legacy",
                    "missing_fields": list(UNKNOWN_PUBLICATION_FIELDS),
                },
            }
        )
    reviewed_entries = extract_reviewed_bibtex_entries(reviewed_bib)
    for key in sorted(reviewed_entries):
        reviewed = _reviewed_publication(key, reviewed_entries[key])
        reviewed["provenance"]["source_sha256"] = sha256_text(reviewed_bib)
        publications.append(reviewed)
    return {
        "catalog_version": 1,
        "baseline": {
            "legacy_python": "docs/_extras/methods_bibliography.py",
            "legacy_python_sha256": sha256_text(legacy_text),
            "source_revision": resolved_revision,
        },
        "parity_provenance": [
            {
                "path": str(path),
                "sha256": sha256_text(git_blob_text(resolved_revision, path)),
            }
            for path in PARITY_PATHS
        ],
        "reviewed_sources": [
            {
                "path": str(REVIEWED_BIB_PATH.relative_to(REPO)),
                "sha256": sha256_text(reviewed_bib),
            }
        ],
        "publications": publications,
        "software_citations": [
            {
                "id": "nirs4all-methods",
                "source": "CITATION.cff",
                "source_sha256": sha256_text(software_cff),
                "cff": software_cff,
            }
        ],
    }


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Read a catalog created by ``migrate_bibliography.py``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return data
