#!/usr/bin/env python3
"""Self-tests for the stdlib-only catalog parser/emitter path."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from unittest.mock import patch

from catalog_loader import load_legacy_methods, normalize_method, render_method
from validate import validate_methods
from validate_catalog import parse_yaml


def assert_round_trip(method: dict) -> None:
    expected = normalize_method(method)
    parsed = parse_yaml(render_method(method))
    # Compared as JSON: NaN defaults (unused optional values) are not equal to
    # themselves as floats.
    if json.dumps(parsed, sort_keys=True) != json.dumps(expected, sort_keys=True):
        method_id = method.get("method_id", "<unknown>")
        raise AssertionError(f"{method_id}: render/parse round-trip drift")


def assert_abi_ownership(methods: list[dict]) -> None:
    """Qualify shared entry points without weakening dedicated-symbol checks."""
    by_id = {m["method_id"]: normalize_method(m) for m in methods}
    shared = [by_id["models.pls.pls_regression"], by_id["models.pls.opls"]]
    generic = "n4m_model_fit"
    dedicated = "n4m_transform_savitzky_golay_create"
    missing = "n4m_missing_export"

    def check(rows, expected, infra, failures, message):
        output = StringIO()
        with (
            patch("validate.expected_abi_symbols", return_value=expected),
            patch("validate.infra_abi_symbols", return_value=infra),
            redirect_stdout(output),
        ):
            result = validate_methods(rows, strict_abi=True)
        assert result == (failures, 0), (result, output.getvalue())
        assert message in output.getvalue(), output.getvalue()

    check(shared, {generic}, {generic}, 0, "0 method + 1 infra = 1/1")
    check(list(reversed(shared)), {generic}, {generic}, 0, "= 1/1")

    collision = deepcopy(shared)
    for method in collision:
        method["abi_symbols"] = [dedicated]
        method["c_surface"] = [dedicated]
    check(collision, {generic, dedicated}, {generic}, 1, "also claimed by")

    repeated = deepcopy(shared)
    repeated[0]["abi_symbols"] = [generic, generic]
    repeated[0]["c_surface"] = [generic, generic]
    check(repeated, {generic}, {generic}, 1, "also claimed by")

    unknown = deepcopy(shared)
    unknown[0]["abi_symbols"] = [missing]
    unknown[0]["c_surface"] = [missing]
    check(unknown, {generic}, {generic}, 1, "ABI symbol not in expected snapshot")
    check(shared, {generic, dedicated}, {generic}, 1, "real ABI symbol unaccounted")
    check(shared, {generic}, {generic, missing}, 1, "infra ABI symbol not in expected snapshot")

    mismatched = deepcopy(shared)
    mismatched[0]["c_surface"] = "none"
    check(mismatched, {generic}, {generic}, 1, "c_surface != abi_symbols")


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

    assert_abi_ownership(methods)
    print(f"PASS: catalog parser/emitter round-trip ({len(methods)} methods)")
    print("PASS: shared ABI ownership, unique coverage and negative ownership checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
