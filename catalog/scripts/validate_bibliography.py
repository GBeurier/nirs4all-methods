#!/usr/bin/env python3
"""Offline scientific-review gate for bibliography catalog v1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bibliography_catalog import (
    CATALOG_PATH,
    LEGACY_PATH,
    PARITY_PATHS,
    REPO,
    REVIEWED_BIB_PATH,
    SOFTWARE_CFF_PATH,
    UNKNOWN_PUBLICATION_FIELDS,
    extract_legacy_bibliography_text,
    extract_reviewed_bibtex_entries,
    git_blob_text,
    load_catalog,
    resolve_revision,
    sha256_text,
)
from generate_bibliography import link_report
from validate_catalog import validate_schema

ID_RE = re.compile(r"^legacy-[a-z0-9_]+$")
REVIEWED_ID_RE = re.compile(r"^reviewed-finetuning-[A-Za-z0-9_-]+$")


def validate(data: dict) -> list[str]:
    """Return review failures; all checks are local and deterministic."""
    errors: list[str] = []
    schema = json.loads(
        (REPO / "catalog" / "schema" / "bibliography_v1.json").read_text(
            encoding="utf-8"
        )
    )
    errors.extend(f"schema: {error}" for error in validate_schema(data, schema))
    if data.get("catalog_version") != 1:
        errors.append("catalog_version must be 1")
    baseline = data.get("baseline") or {}
    revision = baseline.get("source_revision")
    if not isinstance(revision, str):
        errors.append("baseline source_revision is missing")
        return errors
    try:
        revision = resolve_revision(revision)
        legacy_text = git_blob_text(revision, LEGACY_PATH)
        reviewed_bib = git_blob_text(revision, REVIEWED_BIB_PATH)
        software_cff = git_blob_text(revision, SOFTWARE_CFF_PATH)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if baseline.get("source_revision") != revision:
        errors.append("baseline source_revision must be a full immutable commit id")
    if baseline.get("legacy_python") != "docs/_extras/methods_bibliography.py":
        errors.append("legacy bibliography source path does not match v1 contract")
    if baseline.get("legacy_python_sha256") != sha256_text(legacy_text):
        errors.append("legacy bibliography digest does not match recorded source blob")
    expected = extract_legacy_bibliography_text(
        legacy_text, source_name=f"{revision}:docs/_extras/methods_bibliography.py"
    )
    publications = data.get("publications") or []
    legacy_publications = [
        entry
        for entry in publications
        if isinstance(entry, dict)
        and (entry.get("provenance") or {}).get("fields_status") == "unstructured_legacy"
    ]
    by_key = {entry.get("method_key"): entry for entry in legacy_publications}
    if set(by_key) != set(expected):
        errors.append("method keys do not exactly match the historical bibliography")
    if len(legacy_publications) != len(expected) or len(by_key) != len(legacy_publications):
        errors.append("historical bibliography must contain exactly one unstructured record per method")
    ids = [entry.get("id") for entry in publications if isinstance(entry, dict)]
    if len(ids) != len(set(ids)):
        errors.append("publication identifiers are not unique")
    for key, old in expected.items():
        entry = by_key.get(key)
        if not entry:
            continue
        if entry.get("id") != f"legacy-{key}" or not ID_RE.fullmatch(entry["id"]):
            errors.append(f"{key}: unstable publication id")
        if entry.get("legacy_citation") != old.get("paper"):
            errors.append(f"{key}: legacy citation was changed")
        if entry.get("method_title") != old.get("title"):
            errors.append(f"{key}: method title was changed")
        if entry.get("provenance", {}).get("source") != (
            f"docs/_extras/methods_bibliography.py:BIBLIOGRAPHY.{key}.paper"
        ):
            errors.append(
                f"{key}: provenance source does not identify its legacy paper"
            )
        if any(entry.get(field) is not None for field in UNKNOWN_PUBLICATION_FIELDS):
            errors.append(f"{key}: unreviewed structured citation fact was introduced")
        provenance = entry.get("provenance") or {}
        if provenance.get("fields_status") != "unstructured_legacy":
            errors.append(f"{key}: missing unstructured-legacy provenance")
        if tuple(provenance.get("missing_fields") or []) != UNKNOWN_PUBLICATION_FIELDS:
            errors.append(f"{key}: missing-field declaration is incomplete")
    reviewed_entries = extract_reviewed_bibtex_entries(reviewed_bib)
    reviewed_publications = [
        entry
        for entry in publications
        if isinstance(entry, dict)
        and (entry.get("provenance") or {}).get("fields_status") == "reviewed_structured"
    ]
    reviewed_by_key = {
        (entry.get("provenance") or {}).get("entry_key"): entry
        for entry in reviewed_publications
    }
    if set(reviewed_by_key) != set(reviewed_entries):
        errors.append("reviewed BibTeX keys do not exactly match the tracked source")
    if len(reviewed_publications) != len(reviewed_entries) or len(reviewed_by_key) != len(reviewed_publications):
        errors.append("reviewed BibTeX must contain exactly one structured record per source entry")
    reviewed_source_digest = sha256_text(reviewed_bib)
    expected_reviewed_sources = [
        {"path": str(REVIEWED_BIB_PATH.relative_to(REPO)), "sha256": reviewed_source_digest}
    ]
    if data.get("reviewed_sources") != expected_reviewed_sources:
        errors.append("reviewed source digest does not match the pinned source blob")
    for key, parsed in reviewed_entries.items():
        entry = reviewed_by_key.get(key)
        if not entry:
            continue
        provenance = entry.get("provenance") or {}
        fields = parsed["fields"]
        expected_authors = [
            author.strip()
            for author in re.split(r"\s+and\s+", fields.get("author", ""))
            if author.strip()
        ]
        expected_values = {
            "id": f"reviewed-finetuning-{key}",
            "method_key": None,
            "method_title": None,
            "legacy_citation": None,
            "authors": expected_authors,
            "title": fields.get("title"),
            "year": int(fields["year"]) if fields.get("year", "").isdigit() else None,
            "journal": fields.get("journal") or fields.get("booktitle"),
            "doi": fields.get("doi"),
            "url": fields.get("url"),
            "bibtex": parsed["raw_entry"],
        }
        if not REVIEWED_ID_RE.fullmatch(str(entry.get("id"))):
            errors.append(f"{key}: unstable reviewed publication id")
        for field, expected_value in expected_values.items():
            if entry.get(field) != expected_value:
                errors.append(f"{key}: {field} does not match its reviewed BibTeX entry")
        if provenance.get("source") != str(REVIEWED_BIB_PATH.relative_to(REPO)):
            errors.append(f"{key}: reviewed provenance source is not the tracked BibTeX")
        if provenance.get("source_sha256") != reviewed_source_digest:
            errors.append(f"{key}: reviewed provenance source digest does not match")
        if provenance.get("entry_sha256") != sha256_text(parsed["raw_entry"]):
            errors.append(f"{key}: reviewed provenance entry digest does not match")
        expected_missing = [
            field for field, value in expected_values.items()
            if field in UNKNOWN_PUBLICATION_FIELDS and value is None
        ]
        if provenance.get("missing_fields") != expected_missing:
            errors.append(f"{key}: reviewed missing-field declaration does not match source")
    try:
        expected_parity = [
            {"path": str(path), "sha256": sha256_text(git_blob_text(revision, path))}
            for path in PARITY_PATHS
        ]
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if data.get("parity_provenance") != expected_parity:
        errors.append("parity/donor provenance digests do not match local sources")
    software = data.get("software_citations") or []
    if len(software) != 1:
        errors.append("expected exactly one software citation")
    else:
        if software[0].get("source") != "CITATION.cff":
            errors.append("software citation source path does not match v1 contract")
        if software[0].get("source_sha256") != sha256_text(software_cff):
            errors.append("software CFF digest does not match recorded source blob")
        if software[0].get("cff") != software_cff:
            errors.append("software CFF was not preserved verbatim")
    for link in link_report(data)["checked_links"]:
        if not link["offline_syntax_valid"]:
            errors.append(f"invalid offline {link['field']} syntax: {link['value']}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    args = parser.parse_args(argv)
    errors = validate(load_catalog(args.catalog))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: bibliography scientific-review gate (offline; no links fetched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
