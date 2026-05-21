#!/usr/bin/env python3
"""Render the vendored-source distribution for a subset.

Reads catalog/methods.yaml + catalog/subsets/<name>.yaml, computes the
closure via build_closure.py, then emits a target source tree under
build/render/<name>/ containing:

    build/render/<name>/
        subset_manifest.json         (the closure report, for audit)
        VENDORED.md                   (human-readable summary)
        sources/                      (copied TUs under their final paths)
        include/n4m/                  (copied public headers)

Usage:
    python catalog/scripts/render_subset.py <subset>
    python catalog/scripts/render_subset.py <subset> --copy

Without --copy, only the manifest + VENDORED.md are emitted (dry run).
With --copy, the actual source files are tar-style copied into sources/.

This is a STAGING tool — not a build system. The Python / R packaging in
M10 / M11 of the merge plan consumes this tree to assemble the slim wheel
and CRAN source tarball.

Limitation at M2: many TUs the catalog references live under
_donor/nirs4all-methods/cpp/src/core/... (donor side) or at pls4all's
flat cpp/src/core/ (pls4all side). After M3 (donor source move) and M4
(pls4all model.cpp split), TU paths will collapse to the final layout
under cpp/src/core/<category>/...; --copy at that point will produce a
clean vendored tree without _donor/ ambiguity.
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"
BUILD = REPO / "build" / "render"

_spec_bc = importlib.util.spec_from_file_location(
    "_build_closure",
    Path(__file__).parent / "build_closure.py",
)
_bc = importlib.util.module_from_spec(_spec_bc)  # type: ignore[arg-type]
assert _spec_bc and _spec_bc.loader
_spec_bc.loader.exec_module(_bc)  # type: ignore[union-attr]


def write_manifest(target: Path, report: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")


def write_human_summary(target: Path, name: str, report: dict) -> None:
    c = report["closure"]
    lines = [
        f"# Vendored source manifest — `{name}`",
        "",
        f"**Package**: `{report['package']}`",
        f"**Status**: {report['status']}",
        f"**Since release**: {report.get('since_release') or 'n/a'}",
        f"**ABI prefix**: `{(report.get('abi') or {}).get('symbol_prefix') or '?'}`",
        "",
        "## Closure",
        "",
        f"- Methods: **{c['method_count']}**",
        f"- Categories ({c['category_count']}): {', '.join(c['categories'])}",
        f"- Translation units: {len(c['translation_units'])}",
        f"- Public headers: {len(c['headers'])}",
        f"- ABI symbols: {len(c['abi_symbols'])}",
        f"- Parity fixtures: {len(c['fixtures'])}",
        "",
        "## Method IDs by category",
        "",
    ]
    for cat, ids in c["method_ids_by_category"].items():
        lines.append(f"### `{cat}` ({len(ids)})")
        lines.append("")
        for mid in ids:
            lines.append(f"- `{mid}`")
        lines.append("")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))


def copy_sources(report: dict, root: Path) -> tuple[int, list[str]]:
    sources = root / "sources"
    include = root / "include" / "n4m"
    sources.mkdir(parents=True, exist_ok=True)
    include.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    copied = 0
    for tu in report["closure"]["translation_units"]:
        src = REPO / tu
        if not src.exists():
            missing.append(tu)
            continue
        # Map source path to a flat path under sources/, preserving directory structure.
        dest = sources / tu
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    for h in report["closure"]["headers"]:
        src = REPO / h
        if src.exists():
            shutil.copy2(src, include / src.name)
    return copied, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subset", help="subset name (file stem under catalog/subsets/)")
    ap.add_argument("--copy", action="store_true", help="actually copy TUs + headers (default: manifest-only)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: build/render/<subset>)")
    args = ap.parse_args(argv)

    s = _bc.load_subset(args.subset)
    if s is None:
        print(f"FAIL: subset {args.subset} could not be parsed")
        return 1
    methods, index = _bc.load_methods()
    ids = _bc.closure_for(s, methods, index)
    artifacts = _bc.collect_artifacts(ids, index)
    report = {
        "subset":        args.subset,
        "package":       s.get("package"),
        "status":        s.get("status"),
        "since_release": s.get("since_release"),
        "abi":           s.get("abi"),
        "publish":       s.get("publish"),
        "closure":       artifacts,
    }

    out_root = args.out if args.out else BUILD / args.subset
    write_manifest(out_root / "subset_manifest.json", report)
    write_human_summary(out_root / "VENDORED.md", args.subset, report)
    rel = out_root.relative_to(REPO) if out_root.is_relative_to(REPO) else out_root
    print(f"manifest: {rel}/subset_manifest.json")
    print(f"summary:  {rel}/VENDORED.md")
    print(f"methods:  {artifacts['method_count']}    categories: {artifacts['category_count']}    TUs: {len(artifacts['translation_units'])}    headers: {len(artifacts['headers'])}")

    if args.copy:
        copied, missing = copy_sources(report, out_root)
        print(f"sources copied: {copied}")
        if missing:
            print(f"MISSING_TUS: {len(missing)} translation units could not be found:")
            for m in missing[:10]:
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
