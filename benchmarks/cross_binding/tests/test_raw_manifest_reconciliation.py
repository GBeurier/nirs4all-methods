"""Raw-binding manifest gate (bindings/SPEC.md) — structural + NON-STRICT.

The manifest is a catalog projection; its `abi_symbols` are largely
auto-discovered guesses. This test validates the manifest's shape and that the
per-method real/guessed partition is exhaustive, and REPORTS the catalog↔ABI
coverage (the Phase-B cleanup debt) — it does NOT gate on coverage (a ctypes
binding can address any symbol, so symbol-coverage proves nothing; the real
conformance gates are behavioral — see test_donor_binding_specs.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def manifest() -> dict:
    sys.path.insert(0, str(REPO / "catalog" / "scripts"))
    try:
        from render_api import render_manifest
    except Exception as exc:  # pragma: no cover - host guard
        pytest.skip(f"render_api import failed: {exc}")
    return render_manifest()


def test_manifest_shape(manifest):
    assert manifest["method_count"] == len(manifest["methods"]) > 0
    assert manifest.get("source", "").startswith("catalog projection")
    r = manifest["reconciliation"]
    assert {"real_abi_symbol_count", "catalog_symbol_count", "catalog_real",
            "catalog_guessed", "catalog_coverage_pct", "unmapped_real_count",
            "unmapped_real_abi_symbols"} <= set(r)


def test_per_method_real_guessed_partition_is_exhaustive(manifest):
    for m in manifest["methods"]:
        syms = set(m["abi_symbols"])
        assert set(m["real_abi_symbols"]) | set(m["guessed_abi_symbols"]) == syms
        assert not (set(m["real_abi_symbols"]) & set(m["guessed_abi_symbols"]))


def test_reconciliation_is_reported_not_gated(manifest):
    """Always passes — surfaces the debt in the CI log; strict generation is
    gated on this reaching ~100% per bindings/SPEC.md."""
    r = manifest["reconciliation"]
    print(f"\ncatalog↔ABI coverage: {r['catalog_coverage_pct']}% "
          f"({r['catalog_real']}/{r['catalog_symbol_count']} catalog symbols real); "
          f"{r['unmapped_real_count']} real ABI symbols unmapped by the catalog "
          f"(Phase-B cleanup debt).")
    assert r["catalog_real"] + r["catalog_guessed"] == r["catalog_symbol_count"]
