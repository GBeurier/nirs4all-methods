#!/usr/bin/env python3
"""Phase B catalog validation gate."""

from __future__ import annotations

import argparse
import fnmatch
import json
from typing import Any

from catalog_loader import (
    CATALOG,
    CATALOG_VERSION,
    REPO,
    load_methods,
    load_yaml_file,
)
from validate_catalog import validate_schema

SCHEMA_DIR = CATALOG / "schema"
SUBSET_DIR = CATALOG / "subsets"


def expected_abi_symbols() -> set[str]:
    path = REPO / "cpp" / "abi" / "expected_symbols_linux.txt"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        symbol = line.strip().split("@@", 1)[0]
        if symbol:
            out.add(symbol)
    return out


def infra_abi_symbols() -> set[str]:
    """Non-method support symbols (catalog/abi_infra.yaml) — the explicit bucket
    that, together with all method abi_symbols, must cover the full ABI snapshot."""
    path = CATALOG / "abi_infra.yaml"
    if not path.exists():
        return set()
    data = load_yaml_file(path) or {}
    return set(data.get("infra_abi_symbols") or [])


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def check_existing_path(path_value: str, *, suffix: str = "") -> bool:
    path = path_value.split(":", 1)[0] if suffix == "tu" else path_value
    return (REPO / path).exists()


def subset_closure(
    subset: dict[str, Any],
    methods: list[dict[str, Any]],
) -> tuple[set[str], list[str]]:
    errors: list[str] = []
    method_index = {m["method_id"]: m for m in methods}
    by_category: dict[str, set[str]] = {}
    for method in methods:
        by_category.setdefault(method["category"], set()).add(method["method_id"])

    selected: set[str] = set()
    includes = subset.get("includes") or {}
    for category in includes.get("categories") or []:
        ids = by_category.get(category, set())
        if not ids:
            errors.append(f"active subset includes empty category {category!r}")
        selected.update(ids)
    for method_id in includes.get("method_ids") or []:
        if method_id not in method_index:
            errors.append(f"active subset includes unknown method_id {method_id!r}")
        else:
            selected.add(method_id)
    for pattern in includes.get("patterns") or []:
        selected.update(
            method_id
            for method_id in method_index
            if fnmatch.fnmatch(method_id, pattern)
        )

    rejected: set[str] = set()
    excludes = subset.get("excludes") or {}
    for category in excludes.get("categories") or []:
        rejected.update(by_category.get(category, set()))
    for method_id in excludes.get("method_ids") or []:
        rejected.add(method_id)
    for pattern in excludes.get("patterns") or []:
        rejected.update(
            method_id
            for method_id in method_index
            if fnmatch.fnmatch(method_id, pattern)
        )
    return selected - rejected, errors


def validate_methods(
    methods: list[dict[str, Any]],
    *,
    strict_abi: bool,
) -> tuple[int, int]:
    method_schema = load_schema("method.json")
    expected_symbols = expected_abi_symbols()
    seen: set[str] = set()
    sym_owner: dict[str, str] = {}  # ABI symbol -> owning method (duplicate guard)
    fail_count = 0
    warn_count = 0
    warning_samples: list[str] = []

    for index, method in enumerate(methods):
        where = method.get("method_id", f"index:{index}")
        for error in validate_schema(method, method_schema, f"$methods[{index}]"):
            print(f"  SCHEMA: {where}: {error}")
            fail_count += 1

        method_id = method.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            print(f"  FAIL: methods[{index}] has no method_id")
            fail_count += 1
            continue
        if method_id in seen:
            print(f"  FAIL: duplicate method_id {method_id!r}")
            fail_count += 1
        seen.add(method_id)

        if method.get("catalog_version") != CATALOG_VERSION:
            print(f"  FAIL: {method_id}: catalog_version must be {CATALOG_VERSION}")
            fail_count += 1

        for tu in method.get("tu") or []:
            if not check_existing_path(str(tu), suffix="tu"):
                print(f"  FAIL: {method_id}: TU path missing: {tu}")
                fail_count += 1
        for header in method.get("headers") or []:
            if not check_existing_path(str(header)):
                print(f"  FAIL: {method_id}: header path missing: {header}")
                fail_count += 1
        for fixture in (method.get("parity") or {}).get("fixtures") or []:
            if not check_existing_path(str(fixture)):
                print(f"  FAIL: {method_id}: fixture path missing: {fixture}")
                fail_count += 1
        for symbol in method.get("abi_symbols") or []:
            if symbol not in expected_symbols:
                message = f"{method_id}: ABI symbol not in expected snapshot: {symbol}"
                if strict_abi:
                    print(f"  FAIL: {message}")
                    fail_count += 1
                else:
                    if len(warning_samples) < 20:
                        warning_samples.append(message)
                    warn_count += 1
            if strict_abi and symbol in sym_owner:
                print(f"  FAIL: {method_id}: ABI symbol {symbol} also claimed by {sym_owner[symbol]}")
                fail_count += 1
            sym_owner[symbol] = method_id

    # Coverage gate (strict only): every REAL exported symbol must be owned by a
    # method OR listed in the explicit infra bucket. Catches API exported by
    # libn4m but missing from the catalog (the Phase-B reconciliation invariant).
    if strict_abi and expected_symbols:
        infra = infra_abi_symbols()
        # Only the exported FUNCTION symbols (n4m_*) need a method/infra home; the
        # ELF version-script node (N4M_1) is not a function symbol.
        real_funcs = {s for s in expected_symbols if s.startswith("n4m_")}
        unaccounted = sorted(real_funcs - set(sym_owner) - infra)
        for symbol in unaccounted[:20]:
            print(f"  FAIL: real ABI symbol unaccounted (no method, not infra): {symbol}")
        if unaccounted:
            if len(unaccounted) > 20:
                print(f"  FAIL: ... {len(unaccounted) - 20} more unaccounted ABI symbol(s)")
            fail_count += len(unaccounted)
        covered = len(set(sym_owner)) + len(infra)
        print(f"  ABI coverage: {len(set(sym_owner))} method + {len(infra)} infra "
              f"= {covered}/{len(real_funcs)} exported n4m_* symbols")

    for warning in warning_samples:
        print(f"  WARN: {warning}")
    if warn_count > len(warning_samples):
        print(
            f"  WARN: ... {warn_count - len(warning_samples)} additional ABI warning(s)"
        )
    return fail_count, warn_count


def validate_subsets(methods: list[dict[str, Any]]) -> int:
    subset_schema = load_schema("subset_v1.json")
    fail_count = 0
    subset_files = sorted(SUBSET_DIR.glob("*.yaml"))
    print(f"  subsets - {len(subset_files)} descriptors")
    for path in subset_files:
        subset = load_yaml_file(path)
        if not isinstance(subset, dict):
            print(f"  FAIL: {path.name}: top-level is not a mapping")
            fail_count += 1
            continue
        for error in validate_schema(subset, subset_schema, f"${path.name}"):
            print(f"  SCHEMA: {path.name}: {error}")
            fail_count += 1
        if subset.get("status") != "active":
            print(
                f"  {path.name}: status={subset.get('status')} (closure check skipped)"
            )
            continue
        ids, errors = subset_closure(subset, methods)
        for error in errors:
            print(f"  FAIL: {path.name}: {error}")
            fail_count += 1
        if not ids:
            print(f"  FAIL: {path.name}: active subset closure is EMPTY after excludes")
            fail_count += 1
        else:
            print(f"  {path.name}: closure OK ({len(ids)} methods after excludes)")
    return fail_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-abi",
        action="store_true",
        help="fail on snapshot ABI mismatches",
    )
    args = parser.parse_args(argv)

    print(f"== validate.py ==  repo={REPO}")
    try:
        methods = load_methods(prefer_split=True)
    except Exception as exc:
        print(f"FAIL: cannot load methods: {exc}")
        return 1
    print(f"  methods - {len(methods)} entries")

    fail_count, warn_count = validate_methods(methods, strict_abi=args.strict_abi)
    fail_count += validate_subsets(methods)
    print("--")
    if fail_count == 0:
        suffix = f" ({warn_count} warning(s))" if warn_count else ""
        print(f"PASS: catalog validation green{suffix}")
        return 0
    print(f"FAIL: {fail_count} catalog validation error(s), {warn_count} warning(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
