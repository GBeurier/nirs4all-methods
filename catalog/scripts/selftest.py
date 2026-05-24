#!/usr/bin/env python3
"""Self-tests for the stdlib-only catalog parser/emitter path."""

from __future__ import annotations

from catalog_loader import load_legacy_methods, normalize_method, render_method
from validate_catalog import parse_yaml


def assert_round_trip(method: dict) -> None:
    expected = normalize_method(method)
    parsed = parse_yaml(render_method(method))
    if parsed != expected:
        method_id = method.get("method_id", "<unknown>")
        raise AssertionError(f"{method_id}: render/parse round-trip drift")


def main() -> int:
    methods = load_legacy_methods()
    for method in methods:
        assert_round_trip(method)

    multiline = dict(methods[0])
    multiline["notes"] = "line one\nline two"
    parity = dict(multiline.get("parity") or {})
    parity["tolerances"] = [
        {
            "row_key": "selftest",
            "reference": "parser",
            "abs_tol": "1e-12",
            "rel_tol": "1e-12",
            "comparison_set": "all\nfields",
            "notes": "nested\nstring",
        }
    ]
    multiline["parity"] = parity
    assert_round_trip(multiline)

    print(f"PASS: catalog parser/emitter round-trip ({len(methods)} methods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
