"""D-min contract gate: the shipped dashboard payload conforms to its schema
and its per-method score cards are self-consistent with the row cells.

This guards the canonical `dashboard.json` contract (docs/dashboard.schema.json)
that the current dashboard and the future SPA both consume. It validates the
committed artifact (`docs/_static/bench-data.json`), so it needs no result CSVs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PAYLOAD = REPO / "docs" / "_static" / "bench-data.json"
SCHEMA = REPO / "docs" / "dashboard.schema.json"

# The canonical-reference mirror column is excluded from score aggregation
# (it duplicates another column's verdict); keep this in sync with build_landing.
try:
    sys.path.insert(0, str(REPO / "docs" / "_extras"))
    from build_landing import REF_COL_ID  # type: ignore
except Exception:  # pragma: no cover - fallback if build_landing import fails
    REF_COL_ID = "__reference"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(f"{PAYLOAD} not present")
    return json.loads(PAYLOAD.read_text())


def test_payload_matches_schema(payload):
    schema = json.loads(SCHEMA.read_text())
    try:
        import jsonschema
    except ImportError:
        pytest.skip("jsonschema not installed; structural checks still run elsewhere")
    jsonschema.validate(payload, schema)


def test_method_scores_cover_exactly_the_rows(payload):
    row_algos = {r["algo"] for r in payload["rows"]}
    assert set(payload["method_scores"]) == row_algos


def test_method_scores_are_self_consistent(payload):
    """Recompute the score cards from the row cells and assert they match
    what build_landing emitted — so the scores can never silently drift from
    the underlying verdicts."""
    expect: dict = {}
    for r in payload["rows"]:
        e = expect.setdefault(r["algo"], {"reference": Counter(), "binding": Counter(),
                                          "rd": 0, "bd": 0})
        for cid, c in r["cells"].items():
            if cid == REF_COL_ID:
                continue
            if c.get("reference_parity"):
                e["reference"][c["reference_parity"]] += 1
            if c.get("binding_parity"):
                e["binding"][c["binding_parity"]] += 1
            d = c.get("divergence")
            if isinstance(d, (int, float)):
                if c.get("divergence_basis") == "reference":
                    e["rd"] += 1
                else:
                    e["bd"] += 1
    for algo, sc in payload["method_scores"].items():
        assert sc["reference"] == dict(expect[algo]["reference"]), f"{algo} reference"
        assert sc["binding"] == dict(expect[algo]["binding"]), f"{algo} binding"
        assert sc["divergence"]["reference"]["n"] == expect[algo]["rd"], f"{algo} ref div n"
        assert sc["divergence"]["binding"]["n"] == expect[algo]["bd"], f"{algo} bind div n"


def test_divergence_stats_shape(payload):
    for algo, sc in payload["method_scores"].items():
        for basis in ("reference", "binding"):
            st = sc["divergence"][basis]
            assert set(st) >= {"max", "median", "n"}
            assert isinstance(st["n"], int) and st["n"] >= 0
            if st["n"] == 0:
                assert st["max"] is None and st["median"] is None
            else:
                assert st["max"] is not None and st["median"] is not None
