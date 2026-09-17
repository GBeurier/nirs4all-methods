"""End-to-end gates for the generated method-science documentation corpus.

The method pages are generated artifacts, but reviewing only their checked-in
Markdown leaves two important failure modes invisible: a clean checkout can
depend on residual pages, and a source record can have headings without a
usable public binding.  These tests generate twice into empty temporary
directories and verify the catalog, binding, benchmark and science contracts
against source-level indexes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS_DIR = ROOT / "docs" / "methods"
BUILD_METHODS = HERE / "build_methods.py"
REFERENCE_INDEX = HERE / "generate_scientific_reference_index.py"

sys.path.insert(0, str(HERE))

from build_methods import (  # noqa: E402
    REGISTRY_PY,
    SCIENTIFIC_FIELDS,
    catalog_c_symbols,
    parse_methods_catalog,
    parse_registry,
    scientific_content,
    validate_scientific_records,
)
from generate_scientific_reference_index import _links, validate_records  # noqa: E402
from public_api_docs import (  # noqa: E402
    binding_for_catalog,
    cross_bindings_for_catalog,
    scan_matlab_public_api,
    scan_public_api,
    scan_r_public_api,
)


CATALOG_SIZE = 209
REGISTRY_SIZE = 73
PYTHON_VERIFIED = 178
PYTHON_C_ONLY = 31
R_VERIFIED = 70
MATLAB_VERIFIED = 69

SCIENTIFIC_HEADINGS = (
    "### Bibliographic source",
    "### Mathematical principle",
    "### Appropriate uses",
    "### Limits and validation",
    "### Implementation",
    "### Sources and provenance",
)

# These are deliberately maintained historical/HPO landing pages.  The
# catalog renderer must neither overwrite them nor silently remove them from
# the methods toctree.  ``scientific-references.md`` is owned by the separate
# reference-index generator and is checked below with ``--check``.
MANUAL_METHOD_PAGES = frozenset({
    "aom_pop_ridge_active_superblock.md",
    "aom_pop_ridge_global.md",
    "aom_pop_ridge_mkl_superblock.md",
    "aom_pop_ridge_superblock.md",
    "asha.md",
    "cmaes.md",
    "ga_search.md",
    "gp_ei.md",
    "hyperband.md",
    "lhs.md",
    "median_pruner.md",
    "moment_stack.md",
    "moments.md",
    "optimization.md",
    "pso_search.md",
    "racing.md",
    "random.md",
    "ridge.md",
    "sobol.md",
    "sweep_run.md",
    "ternary.md",
    "tpe.md",
})
SEPARATE_GENERATED_PAGES = frozenset({
    "scientific-references.md",
    "_finetuning_bibliography.bib",
})
SCIENTIFIC_SOURCE_FILES = tuple(sorted(HERE.glob("scientific_*.py")))

_COVERAGE_ROW = re.compile(
    r"^\| `([^`]+)` \| \[([^]]+)\]\(([^)]+)\) \| (\w+) \|$", re.MULTILINE
)
_RETIRED_ROUTE = re.compile(
    r"(?:^\s*(?:from|import)\s+pls4all\b|library\(pls4all\)|"
    r"\bpls4all\.(?:python|sklearn|R(?:\.|\b)|cpp)\b|\bn4m\.sklearn\b|"
    r"bindings/r/pls4all|\+pls4all)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class GeneratedCorpus:
    first: Path
    second: Path


def _run_generator(out: Path) -> None:
    process = subprocess.run(
        [sys.executable, str(BUILD_METHODS), "--strict", "--no-bench", "--out", str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def _files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _coverage_pages(out: Path) -> dict[str, Path]:
    rows = _COVERAGE_ROW.findall((out / "coverage.md").read_text(encoding="utf-8"))
    assert len(rows) == CATALOG_SIZE
    assert len({method_id for method_id, _label, _page, _source in rows}) == CATALOG_SIZE
    assert all(source == "curated" for _method_id, _label, _page, source in rows)
    page_for = {method_id: out / page for method_id, _label, page, _source in rows}
    missing = [str(page.relative_to(out)) for page in page_for.values() if not page.is_file()]
    assert not missing, f"catalog pages absent from a clean generation: {missing}"
    return page_for


def _duplicate_literal_keys(path: Path) -> list[tuple[str, int, int]]:
    """Return duplicate literal dict keys without importing an overlay module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    duplicates: list[tuple[str, int, int]] = []
    for mapping in ast.walk(tree):
        if not isinstance(mapping, ast.Dict):
            continue
        first_line: dict[str, int] = {}
        for key in mapping.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            line = getattr(key, "lineno", 0)
            original = first_line.setdefault(key.value, line)
            if original != line:
                duplicates.append((key.value, original, line))
    return duplicates


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> GeneratedCorpus:
    first = tmp_path_factory.mktemp("methods-clean-a")
    second = tmp_path_factory.mktemp("methods-clean-b")
    _run_generator(first)
    _run_generator(second)
    assert _files(first) == _files(second), "method generation is not byte-idempotent"
    return GeneratedCorpus(first=first, second=second)


def test_negative_scientific_record_contract_rejects_missing_placeholder_and_controls() -> None:
    """Schema validation must reject hollow prose and escaped TeX controls."""
    valid = {
        "title": "Validated record",
        "paper": "A reviewed paper with https://doi.org/10.1000/example.",
        "principle": "A concrete mathematical principle with declared variables.",
        "use_cases": "A bounded spectroscopy use case with a stated decision.",
        "limitations": "A validation limit and an external-assessment condition.",
        "implementation": "The current ABI-2 implementation and source are named.",
        "provenance": "https://example.org/provenance",
    }
    assert validate_scientific_records({"valid": valid})["valid"] == valid

    missing = dict(valid, paper="")
    with pytest.raises(ValueError, match="missing required fields: paper"):
        validate_scientific_records({"missing": missing})

    placeholder = dict(valid, principle="Standard spectroscopic operator")
    with pytest.raises(ValueError, match="placeholders"):
        validate_scientific_records({"placeholder": placeholder})

    for control in ("\x08", "\t"):
        malformed = dict(valid, principle=f"TeX escape became {control}ar x.")
        with pytest.raises(ValueError, match="forbidden C0 controls"):
            validate_scientific_records({"control": malformed})


def test_scientific_overlay_sources_have_no_duplicate_literal_keys() -> None:
    """A duplicate dict key silently restores stale prose at import time.

    This deliberately parses source instead of importing it, so the check has
    no module-cache side effects and can identify both lines that need merging.
    Ruff F601 runs alongside it in the docs workflow for an independent rule.
    """
    assert SCIENTIFIC_SOURCE_FILES
    duplicates = {
        path.name: _duplicate_literal_keys(path)
        for path in SCIENTIFIC_SOURCE_FILES
    }
    duplicates = {path: rows for path, rows in duplicates.items() if rows}
    assert not duplicates, "duplicate scientific dict keys overwrite corrections: " + repr(duplicates)


def test_reference_link_parser_retains_wiley_doi_and_punctuated_url() -> None:
    """A DOI/URL extractor must not truncate legal parentheses or semicolons."""
    wiley = "10.1002/(SICI)1097-0312(199612)49:12<2511::AID-CPA6>3.0.CO;2-P"
    url = "https://example.org/ga(date;2026-09-17)#figure-2"
    record = {
        "paper": f"See [Wiley record](https://doi.org/{wiley}).",
        "provenance": f"Implementation ({url}).",
    }
    links = {(item["kind"], item["value"]) for item in _links(record)}
    assert ("doi", wiley) in links
    assert ("url", f"https://doi.org/{wiley}") in links
    assert ("url", url) in links
    complete = {
        "title": "Link record",
        "principle": "A mathematical statement long enough to be substantive.",
        "use_cases": "A concrete use case that names the intended decision.",
        "limitations": "A concrete limitation that calls for independent validation.",
        "implementation": "A current ABI-2 implementation statement.",
        **record,
    }
    assert validate_records({"link-record": complete}) == []


def test_current_wiley_reference_records_round_trip_without_fragments() -> None:
    """Regression for real DOI strings which previously lost their suffix."""
    expected = {
        "mb_pls": "10.1002/(SICI)1099-128X(199809/10)12:5%3C301::AID-CEM515%3E3.0.CO;2-S",
        "n_pls": "10.1002/(SICI)1099-128X(199601)10:1%3C47::AID-CEM400%3E3.0.CO;2-C",
        "ga_select": "10.1002/1099-128X(200009/12)14:5/6%3C643::AID-CEM621%3E3.0.CO;2-E",
    }
    records = scientific_content()
    for stem, doi in expected.items():
        links = {(item["kind"], item["value"]) for item in _links(records[stem])}
        assert ("doi", doi) in links
        assert ("url", f"https://doi.org/{doi}") in links


def test_curated_records_are_substantive_and_link_valid() -> None:
    records = scientific_content()
    assert len(records) >= CATALOG_SIZE
    assert validate_records(records) == []
    minimum_length = {
        "title": 3,
        "paper": 40,
        "principle": 60,
        "use_cases": 40,
        "limitations": 60,
        "implementation": 12,
        "provenance": 40,
    }
    for stem, record in records.items():
        for field in SCIENTIFIC_FIELDS:
            assert len(record[field].strip()) >= minimum_length[field], (
                f"{stem}: {field} is too short to be a scientific record"
            )


def test_runtime_science_overlays_preserve_corrected_algorithm_claims() -> None:
    """Protect corrections whose wrong text could otherwise be reintroduced."""
    records = scientific_content()
    pcr = records["pcr"]["principle"]
    assert r"\mathbf V_k" in pcr and r"\mathbf U_k" in pcr

    wvc = records["wvc_select"]["principle"]
    assert "No singular-value decomposition" in wvc

    weighted = records["weighted_pls"]["implementation"]
    assert "replayable" in weighted and "coefficient" in weighted

    glm = records["pls_glm"]["principle"] + " " + records["pls_glm"]["implementation"]
    assert "does not construct GLM working responses" in glm
    assert "reweighted least squares" in glm
    assert "not fitted by a Poisson likelihood" in glm


def test_clean_catalog_corpus_has_complete_science_and_current_bindings(
        corpus: GeneratedCorpus) -> None:
    out = corpus.first
    page_for = _coverage_pages(out)
    catalog = parse_methods_catalog(ROOT / "catalog" / "methods.yaml")
    assert len(catalog) == CATALOG_SIZE

    public = scan_public_api()
    r_public = scan_r_public_api()
    matlab_public = scan_matlab_public_api()
    python_bindings = {
        method["method_id"]: binding_for_catalog(method, public)
        for method in catalog
    }
    assert sum(binding is not None for binding in python_bindings.values()) == PYTHON_VERIFIED
    assert sum(binding is None for binding in python_bindings.values()) == PYTHON_C_ONLY

    rendered_r = 0
    rendered_matlab = 0
    for method in catalog:
        method_id = method["method_id"]
        text = page_for[method_id].read_text(encoding="utf-8")
        for heading in SCIENTIFIC_HEADINGS:
            assert heading in text, f"{method_id}: missing {heading}"
        assert (
            "### API and bindings" in text
            or "## API and bindings" in text
            or "## API surface" in text
            or "### API" in text
        ), method_id

        binding = python_bindings[method_id]
        if binding is None:
            assert "Python (verified public re-export)" not in text, method_id
            assert "cpp/include/n4m/" in text, f"{method_id}: C-only surface lacks header link"
        else:
            assert binding["import"] in text, f"{method_id}: missing verified Python import"
            assert f"{binding['symbol']}{binding['signature'] or '(...)'}" in text, (
                f"{method_id}: missing exact Python signature"
            )

        for symbol in catalog_c_symbols(method):
            assert symbol in text, f"{method_id}: catalog C ABI symbol omitted: {symbol}"

        cross = cross_bindings_for_catalog(method, r_public, matlab_public)
        for key, label in (("r", "R (source-verified)"), ("matlab", "MATLAB / Octave (source-verified)")):
            binding = cross.get(key)
            if binding is None:
                continue
            assert label in text, f"{method_id}: {key} source-verified binding omitted"
            assert f"{binding['symbol']}{binding['signature'] or '(...)'}" in text, (
                f"{method_id}: {key} signature drift"
            )
            if key == "r":
                rendered_r += 1
            else:
                rendered_matlab += 1

        pre_benchmark = text.split("### Benchmarks", maxsplit=1)[0]
        assert not _RETIRED_ROUTE.search(pre_benchmark), (
            f"{method_id}: retired route appears outside benchmark provenance"
        )

    assert rendered_r == R_VERIFIED
    assert rendered_matlab == MATLAB_VERIFIED


def test_no_bench_generation_preserves_all_registry_snapshots(
        corpus: GeneratedCorpus) -> None:
    registry = parse_registry(REGISTRY_PY)
    assert len(registry) == REGISTRY_SIZE
    for spec in registry:
        path = corpus.first / f"{spec['name']}.md"
        assert path.is_file(), f"registry page missing: {spec['name']}"
        assert "### Benchmarks" in path.read_text(encoding="utf-8"), (
            f"--no-bench dropped committed benchmark snapshot: {spec['name']}"
        )

    # Snapshot tables retain their recorded raw IDs and measurements, but
    # their captions must not advertise a retired package as a live binding.
    pls = (corpus.first / "pls.md").read_text(encoding="utf-8")
    assert "**Archived measurement identity.**" in pls
    assert "Python · archived pls4all benchmark" in pls
    assert "<code>pls4all.sklearn</code>" in pls
    assert "1.97 ms" in pls


def test_union_parameter_annotations_do_not_split_markdown_table_cells(
        corpus: GeneratedCorpus) -> None:
    """A union annotation must remain one cell when Sphinx parses the table."""
    pcr = (corpus.first / "pcr.md").read_text(encoding="utf-8")
    row = next(line for line in pcr.splitlines() if line.startswith("| `center_x` |"))
    # Markdown tables use ``|`` as a cell separator.  The source annotation
    # is ``bool | None`` and therefore needs an escaped pipe in generated
    # Markdown; checking the parsed cells catches visually shifted HTML rows.
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]]
    assert cells == [
        "`center_x`",
        r"`bool \| None`",
        "`True`",
        "current public binding signature",
    ]


def test_committed_corpus_matches_clean_generation_and_preserves_manual_pages(
        corpus: GeneratedCorpus) -> None:
    fresh = _files(corpus.first)
    committed = _files(METHODS_DIR)
    generated_names = set(fresh)
    committed_names = set(committed)
    expected_manual = {Path(name) for name in MANUAL_METHOD_PAGES | SEPARATE_GENERATED_PAGES}
    assert committed_names - generated_names == expected_manual
    assert all((METHODS_DIR / page).is_file() for page in expected_manual)
    stale = [str(page) for page, content in fresh.items() if committed.get(page) != content]
    assert not stale, "checked-in method pages differ from clean generation: " + ", ".join(stale[:12])

    index = (METHODS_DIR / "index.md").read_text(encoding="utf-8")
    assert ":glob:" in index and "\n*\n" in index, (
        "methods toctree no longer exposes preserved HPO/AOM/reference pages"
    )


def test_checked_in_scientific_reference_artifacts_are_current() -> None:
    process = subprocess.run(
        [sys.executable, str(REFERENCE_INDEX), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
