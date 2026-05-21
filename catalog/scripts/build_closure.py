#!/usr/bin/env python3
"""Compute the transitive closure of methods + TUs + headers + fixtures
needed by a given subset YAML.

Reads catalog/methods.yaml and the requested catalog/subsets/<name>.yaml.
Emits a JSON manifest describing exactly what the vendored-source
distribution must ship.

Usage:
    python catalog/scripts/build_closure.py <subset-name>
    python catalog/scripts/build_closure.py --all

Examples:
    python catalog/scripts/build_closure.py nirs4all_methods
    python catalog/scripts/build_closure.py pls4all

Outputs JSON to stdout (or under build/closure/<subset>.json with --emit).
"""

from __future__ import annotations
import argparse
import fnmatch
import json
import sys
from pathlib import Path

# Reuse the YAML parser from validate_catalog.py via direct import-by-path.
import importlib.util

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"

_spec = importlib.util.spec_from_file_location(
    "_validate_catalog",
    Path(__file__).parent / "validate_catalog.py",
)
_vc = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(_vc)  # type: ignore[union-attr]
parse_yaml = _vc.parse_yaml


def load_subset(name: str) -> dict:
    path = CATALOG / "subsets" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"subset not found: {path}")
    return parse_yaml(path.read_text(encoding="utf-8"))


def load_methods() -> tuple[list[dict], dict[str, dict]]:
    doc = parse_yaml((CATALOG / "methods.yaml").read_text(encoding="utf-8"))
    methods: list[dict] = doc["methods"]
    index = {m["method_id"]: m for m in methods}
    return methods, index


def closure_for(subset: dict, methods: list[dict], index: dict[str, dict]) -> set[str]:
    inc = subset.get("includes") or {}
    exc = subset.get("excludes") or {}

    by_category: dict[str, set[str]] = {}
    for m in methods:
        by_category.setdefault(m["category"], set()).add(m["method_id"])

    selected: set[str] = set()
    for cat in inc.get("categories") or []:
        selected.update(by_category.get(cat, set()))
    for mid in inc.get("method_ids") or []:
        if mid in index:
            selected.add(mid)
    for pat in inc.get("patterns") or []:
        for mid in index:
            if fnmatch.fnmatch(mid, pat):
                selected.add(mid)

    rejected: set[str] = set()
    for cat in exc.get("categories") or []:
        rejected.update(by_category.get(cat, set()))
    for mid in exc.get("method_ids") or []:
        rejected.add(mid)
    for pat in exc.get("patterns") or []:
        for mid in index:
            if fnmatch.fnmatch(mid, pat):
                rejected.add(mid)

    return selected - rejected


def collect_artifacts(method_ids: set[str], index: dict[str, dict]) -> dict:
    tus: set[str] = set()
    headers: set[str] = set()
    fixtures: set[str] = set()
    abi_symbols: set[str] = set()
    feature_flags: set[str] = set()
    vendored_licenses: set[str] = set()
    bench_ids: set[str] = set()
    categories: set[str] = set()
    by_category: dict[str, list[str]] = {}

    for mid in sorted(method_ids):
        m = index[mid]
        categories.add(m["category"])
        by_category.setdefault(m["category"], []).append(mid)
        for tu in m.get("tu", []):
            tus.add(tu.split(":", 1)[0])
        for h in m.get("headers", []) or []:
            headers.add(h)
        for ctu in m.get("carry_along_tu", []) or []:
            tus.add(ctu)
        f = m.get("parity") or {}
        for fx in f.get("fixtures", []) or []:
            fixtures.add(fx)
        for sym in m.get("abi_symbols", []) or []:
            abi_symbols.add(sym)
        for ff in m.get("feature_flags", []) or []:
            feature_flags.add(ff)
        for lic in m.get("vendored_licenses", []) or []:
            vendored_licenses.add(lic)
        b = m.get("bench") or {}
        if b.get("bench_id"):
            bench_ids.add(b["bench_id"])

    return {
        "method_count": len(method_ids),
        "category_count": len(categories),
        "categories": sorted(categories),
        "method_ids_by_category": {k: sorted(v) for k, v in sorted(by_category.items())},
        "translation_units": sorted(tus),
        "headers": sorted(headers),
        "fixtures": sorted(fixtures),
        "abi_symbols": sorted(abi_symbols),
        "feature_flags": sorted(feature_flags),
        "vendored_licenses": sorted(vendored_licenses),
        "bench_ids": sorted(bench_ids),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subset", nargs="?", help="subset name (file stem under catalog/subsets/)")
    ap.add_argument("--all", action="store_true", help="run for every subset and emit one file each")
    ap.add_argument("--emit", action="store_true", help="write build/closure/<subset>.json instead of stdout")
    args = ap.parse_args(argv)

    if not args.all and not args.subset:
        ap.error("provide a subset name or --all")

    methods, index = load_methods()
    subsets = [args.subset] if args.subset else sorted(p.stem for p in (CATALOG / "subsets").glob("*.yaml"))

    for name in subsets:
        s = load_subset(name)
        if s is None:
            print(f"{name}: empty YAML", file=sys.stderr)
            return 1
        ids = closure_for(s, methods, index)
        artifacts = collect_artifacts(ids, index)
        report = {
            "subset":        name,
            "package":       s.get("package"),
            "status":        s.get("status"),
            "since_release": s.get("since_release"),
            "abi":           s.get("abi"),
            "publish":       s.get("publish"),
            "closure":       artifacts,
        }
        rendered = json.dumps(report, indent=2, sort_keys=False)
        if args.emit:
            out_dir = REPO / "build" / "closure"
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{name}.json"
            target.write_text(rendered + "\n")
            print(f"emitted {target.relative_to(REPO)} ({len(ids)} methods)")
        else:
            print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
