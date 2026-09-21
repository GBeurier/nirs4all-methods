#!/usr/bin/env python3
"""Render and validate the current, curated method-science reference index.

This is intentionally separate from the lossless historical bibliography
catalog.  The curated records contain reviewed prose plus DOI/URL provenance,
but not enough structured metadata to manufacture BibTeX.  The index makes
those references globally discoverable and gates their link syntax without
pretending that every string is a reviewed BibTeX record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_PAGE = ROOT / "docs" / "methods" / "scientific-references.md"
OUT_LINKS = ROOT / "docs" / "_static" / "scientific-reference-links.json"
SOURCE_FILES = (
    "methods_bibliography.py",
    "scientific_legacy.py",
    "scientific_augmentation_filter_split.py",
    "scientific_remaining.py",
    "scientific_aom.py",
)
# DOI suffixes legitimately contain parenthesis, semicolons, colons and
# percent-escaped markup.  URL extraction therefore cannot stop at `)` or
# `;`: Wiley DOI targets such as `10.1002/(SICI)…;2-S` would otherwise be
# silently truncated in the generated index and JSON gate.
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9%<>]+", re.IGNORECASE)
# Bare HTTP locators stop at Markdown delimiters. Markdown-link targets are
# parsed separately because DOI paths may themselves contain parentheses.
URL_RE = re.compile(r"https?://[^\s<>\]]+", re.IGNORECASE)

sys.path.insert(0, str(HERE))
from build_methods import SCIENTIFIC_FIELDS, scientific_content  # noqa: E402


def _trim(value: str) -> str:
    """Trim prose/Markdown delimiters while retaining balanced DOI syntax."""
    value = value.rstrip(".,;:!?")
    # Markdown links contribute one closing parenthesis after the URL.  A DOI
    # itself can contain balanced parentheses, so remove only surplus closers.
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value


def _markdown_targets(value: str) -> list[tuple[int, int, str]]:
    """Return Markdown link targets with balanced parenthesis preserved.

    Regex alone cannot distinguish the closing Markdown parenthesis from the
    valid balanced parenthesis in a Wiley DOI path.  The small scanner starts
    after every ``](http`` and closes only after nested pairs balance.
    """
    targets: list[tuple[int, int, str]] = []
    marker = re.compile(r"\]\((https?://)", re.IGNORECASE)
    for match in marker.finditer(value):
        start = match.start(1)
        index = start
        depth = 0
        while index < len(value):
            char = value[index]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    targets.append((start, index, value[start:index]))
                    break
                depth -= 1
            index += 1
    return targets


def _urls(value: str) -> list[str]:
    targets = _markdown_targets(value)
    protected = [(start, end) for start, end, _url in targets]
    urls = {url for _start, _end, url in targets}
    for match in URL_RE.finditer(value):
        if any(start <= match.start() < end for start, end in protected):
            continue
        urls.add(_trim(match.group(0)))
    return sorted(urls)


def _source_digests() -> dict[str, str]:
    digests: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = HERE / name
        if path.exists():
            digests[f"docs/_extras/{name}"] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return digests


def _links(record: dict[str, str],
           fields: tuple[str, ...] = ("paper", "provenance")) -> list[dict[str, str]]:
    found: set[tuple[str, str]] = set()
    for field in fields:
        value = record[field]
        for doi in DOI_RE.findall(value):
            found.add(("doi", _trim(doi)))
        for url in _urls(value):
            found.add(("url", url))
            parsed = urlsplit(url)
            if parsed.hostname and parsed.hostname.lower() in {"doi.org", "dx.doi.org"}:
                # A doi.org URL is already a full locator. Keep its encoded
                # path: decoding `%3C` / `%3E` would introduce characters
                # which are valid in a URL but not in a bare DOI token.
                doi = parsed.path.lstrip("/")
                if doi and DOI_RE.fullmatch(doi):
                    found.add(("doi", doi))
    return [
        {"kind": kind, "value": value}
        for kind, value in sorted(found, key=lambda item: (item[0], item[1]))
    ]


def validate_records(records: dict[str, dict[str, str]]) -> list[str]:
    """Return validation errors for content and offline DOI/URL syntax."""
    errors: list[str] = []
    for stem, record in sorted(records.items()):
        for field in SCIENTIFIC_FIELDS:
            if not isinstance(record.get(field), str) or not record[field].strip():
                errors.append(f"{stem}: missing required scientific field {field}")
        if not any(_links(record)):
            errors.append(f"{stem}: paper/provenance contains no DOI or HTTP(S) URL")
        paper = record.get("paper", "")
        paper_lower = paper.lower()
        explicit_absence = any(marker in paper_lower for marker in (
            "no canonical", "no single", "no paper",
            "no unique", "no publication", "implementation-specific",
            "no verified canonical", "no implemented",
        ))
        if not explicit_absence and not _links(record, ("paper",)):
            errors.append(
                f"{stem}: named bibliographic source lacks a reviewed DOI or URL")
        for link in _links(record):
            if link["kind"] == "doi":
                if not DOI_RE.fullmatch(link["value"]):
                    errors.append(f"{stem}: malformed DOI {link['value']!r}")
            else:
                parsed = urlparse(link["value"])
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    errors.append(f"{stem}: malformed URL {link['value']!r}")
    return errors


def render_page(records: dict[str, dict[str, str]], digests: dict[str, str]) -> str:
    lines = [
        "# Current method-science reference index",
        "",
        "This generated index covers the scientific records rendered in the method pages. "
        "It complements the [lossless historical bibliography](../bibliography.md): the "
        "records below retain their source text and DOI/URL provenance, but no new BibTeX "
        "is emitted without a separately reviewed structured source.",
        "",
        f"- Curated records: **{len(records)}**",
        "- Required fields: bibliographic source, principle, uses, limits, implementation, and provenance.",
        "- Links are checked for offline DOI/URL syntax by this generator; they are not network-fetched.",
        "",
        "## Source inputs",
        "",
    ]
    for path, digest in sorted(digests.items()):
        lines.append(f"- `{path}` — SHA-256 `{digest}`")
    lines.extend(["", "## References by documentation page", ""])
    for stem, record in sorted(records.items()):
        lines.extend([
            f"### [`{stem}`]({stem}.md) — {record['title']}",
            "",
            record["paper"],
            "",
            "**Provenance:** " + record["provenance"],
            "",
        ])
    return "\n".join(lines)


def render_links(records: dict[str, dict[str, str]], digests: dict[str, str]) -> str:
    payload = {
        "schema": 1,
        "purpose": "offline syntax report for current method-science references",
        "source_digests": digests,
        "records": [
            {"page": stem, "links": _links(record)}
            for stem, record in sorted(records.items())
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail when committed artifacts differ from source records")
    args = parser.parse_args()
    records = scientific_content()
    errors = validate_records(records)
    if errors:
        raise SystemExit("scientific reference validation failed:\n- " + "\n- ".join(errors))
    digests = _source_digests()
    page = render_page(records, digests)
    links = render_links(records, digests)
    expected = ((OUT_PAGE, page), (OUT_LINKS, links))
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, content in expected
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit("stale scientific reference artifacts: " + ", ".join(stale))
        return
    for path, content in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(records)} current science references")


if __name__ == "__main__":
    main()
