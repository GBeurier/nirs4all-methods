"""Phase-C-minimal gate: the parity comparator summary is well-formed and its
totals are consistent with the per-method verdicts. Validates the committed
`parity/results/latest.json` (no result CSVs needed); also smoke-tests the
generator when the result CSVs are present.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LATEST = REPO / "parity" / "results" / "latest.json"
RESULTS = REPO / "benchmarks" / "cross_binding" / "results"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not LATEST.exists():
        pytest.skip(f"{LATEST} not present (run `make parity`)")
    return json.loads(LATEST.read_text())


def test_summary_shape(summary):
    assert "methods" in summary and "totals" in summary
    t = summary["totals"]
    assert {"methods", "reference", "binding"} <= set(t)
    assert t["methods"] == len(summary["methods"])
    for sc in summary["methods"].values():
        assert {"reference", "binding", "divergence"} <= set(sc)


def test_totals_match_per_method(summary):
    ref, bind = Counter(), Counter()
    for sc in summary["methods"].values():
        ref.update(sc["reference"])
        bind.update(sc["binding"])
    # totals are only the all-methods rollup when the summary isn't filtered
    if summary["totals"]["methods"] == len(summary["methods"]):
        assert dict(ref) == summary["totals"]["reference"]
        assert dict(bind) == summary["totals"]["binding"]


def test_generator_runs_if_csvs_present():
    if not (RESULTS / "full_matrix.csv").exists():
        pytest.skip("result CSVs not present")
    sys.path.insert(0, str(REPO / "parity" / "comparator"))
    sys.path.insert(0, str(REPO / "docs" / "_extras"))
    import run as comparator  # parity/comparator/run.py
    s = comparator.build_summary(RESULTS)
    assert s["totals"]["methods"] == len(s["methods"]) > 0
    assert "reference" in s["totals"] and "binding" in s["totals"]
