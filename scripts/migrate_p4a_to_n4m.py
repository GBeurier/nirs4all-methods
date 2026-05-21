#!/usr/bin/env python3
"""M5: mechanical p4a_* → n4m_* rename across the unified repo.

The pls4all-side sources use `p4a_*` / `P4A_*` / `pls4all::*` symbols and
the `pls4all/p4a.h` umbrella header. M5 unifies onto donor's `n4m_*` /
`N4M_*` / `n4m::*` convention with `n4m/n4m.h`. The donor sources already
use the target naming, so the rename only touches pls4all-origin files.

Mapping (case-sensitive, word-boundary aware):

| Pattern (regex)            | Replacement      | Notes                          |
|----------------------------|------------------|--------------------------------|
| \\bp4a_                    | n4m_             | function / type / variable names |
| \\bP4A_                    | N4M_             | macros, enum constants         |
| pls4all::core              | n4m::core        | C++ namespace ref              |
| pls4all::cuda_dispatch     | n4m::cuda_dispatch |                              |
| namespace\\s+pls4all       | namespace n4m    | namespace declaration          |
| ::pls4all::                | ::n4m::          | fully qualified                |
| pls4all/p4a.h              | n4m/n4m.h        | include path                   |
| pls4all/p4a_export.h       | n4m/n4m_export.h |                                |
| pls4all/p4a_version.h      | n4m/n4m_version.h |                              |
| libp4a                     | libn4m           | library / DLL name             |
| pls4all_core               | n4m_core         | CMake target                   |
| pls4all_c\\b               | n4m_c            | CMake target                   |
| pls4all_c_static           | n4m_c_static     |                                |
| pls4all_tests              | n4m_tests        |                                |
| pls4all_cli                | n4m_cli          |                                |

Things this script intentionally does NOT touch:

  - The string "pls4all" in user-facing docs (README.md, CHANGELOG.md, etc.) —
    those describe the historical package name and the slim subset that
    still ships as `pls4all` on PyPI/CRAN. Renaming would erase project
    identity.
  - `bindings/r/pls4all/` and `bindings/python/src/pls4all/` directory names
    — those host the SLIM `pls4all` subset packages; only their internal
    p4a_* symbols rename to n4m_*.
  - The `pls4all-final/v1.0.0-pre-merge` git tag (already pushed).
  - The catalog YAMLs (catalog/subsets/pls4all.yaml etc.) reference the
    slim package name; those stay.
  - Files under `_donor/` (already empty post-M3; nothing to rename).

Usage:
    python scripts/migrate_p4a_to_n4m.py --dry-run [paths...]
    python scripts/migrate_p4a_to_n4m.py --apply --stage core
    python scripts/migrate_p4a_to_n4m.py --apply --stage bindings
    python scripts/migrate_p4a_to_n4m.py --apply --stage docs

Stages:
    core      → cpp/src/, cpp/include/pls4all/, cpp/tests/, cpp/cli/
    bindings  → bindings/* (python, r, js, julia, matlab, go, rust, dotnet,
                ruby, lua, nim, jni, octave, android)
    parity    → parity/
    benchmarks → benchmarks/
    docs      → docs/
    all       → everything above (large; use after staged smoke tests)
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Order matters: more-specific patterns must run before more-general ones.
REPLACEMENTS: list[tuple[re.Pattern[str], str, str]] = [
    # CMake target names (most specific)
    (re.compile(r"\bpls4all_c_static\b"),  "n4m_c_static",   "CMake target rename"),
    (re.compile(r"\bpls4all_core\b"),      "n4m_core",       "CMake target rename"),
    (re.compile(r"\bpls4all_tests\b"),     "n4m_tests",      "CMake target rename"),
    (re.compile(r"\bpls4all_cli\b"),       "n4m_cli",        "CMake target rename"),
    (re.compile(r"\bpls4all_c\b"),         "n4m_c",          "CMake target rename"),
    # Include paths (must run before bare pls4all:: substitutions to avoid breaking)
    (re.compile(r'"pls4all/p4a_export\.h"'), '"n4m/n4m_export.h"',  "include path"),
    (re.compile(r'"pls4all/p4a_version\.h"'), '"n4m/n4m_version.h"', "include path"),
    (re.compile(r'"pls4all/p4a\.h"'),       '"n4m/n4m.h"',         "include path"),
    (re.compile(r"<pls4all/p4a_export\.h>"), "<n4m/n4m_export.h>",  "include path (angle)"),
    (re.compile(r"<pls4all/p4a_version\.h>"), "<n4m/n4m_version.h>", "include path (angle)"),
    (re.compile(r"<pls4all/p4a\.h>"),       "<n4m/n4m.h>",         "include path (angle)"),
    # Library names
    (re.compile(r"\blibp4a\b"),            "libn4m",         "library name"),
    # C++ namespace
    (re.compile(r"\bpls4all::core\b"),     "n4m::core",      "C++ namespace ref"),
    (re.compile(r"\bpls4all::cuda_dispatch\b"), "n4m::cuda_dispatch", "C++ namespace ref"),
    (re.compile(r"\b::pls4all::"),         "::n4m::",        "C++ fully qualified"),
    (re.compile(r"\bnamespace\s+pls4all\b"), "namespace n4m", "namespace declaration"),
    # Token-level renames
    (re.compile(r"\bp4a_"),                "n4m_",           "function / type / variable prefix"),
    (re.compile(r"\bP4A_"),                "N4M_",           "macro / enum constant prefix"),
]

STAGE_PATHS = {
    "core": [
        "cpp/src",
        "cpp/include/pls4all",
        "cpp/tests",
        "cpp/cli",
    ],
    "bindings": [
        "bindings",
    ],
    "parity": ["parity"],
    "benchmarks": ["benchmarks"],
    "docs": ["docs"],
}

# Skip these files even if matched by stage globs.
SKIP_FILES = {
    # User-facing docs that describe the historical pls4all project — renaming
    # these would erase project identity. These references are intentional.
    "README.md",
    "CHANGELOG.md",
    "ARCHITECTURE.md",
    "DISTRIBUTION.md",
    "Direction_Technique.md",
    "Overview.md",
    "Backlog.md",
    "CITATION.cff",
    "ROADMAP.md",
    "MERGE_ROADMAP.md",
}

# Patterns to skip directory-wise (e.g. don't touch the catalog YAMLs that
# intentionally reference the slim 'pls4all' package).
SKIP_DIRS = {
    "_donor",   # already empty post-M3; we never rename anything still under here
    "build",
    ".git",
    "__pycache__",
    "node_modules",
    ".reference_oracles",   # generated artifacts
    ".x_target",            # generated artifacts
    ".predictions",         # generated artifacts
    ".ruff_cache",
}

# File extensions we consider text-source for token replacement.
TEXT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".in",
    ".py", ".pyi",
    ".r", ".R", ".Rd",
    ".m",  # matlab/octave
    ".jl",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".cs", ".rb", ".lua", ".nim",
    ".java", ".kt", ".swift",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".cmake", ".txt", ".md",
    ".sh", ".ps1",
    ".def", ".ld",
}


def should_visit(p: Path) -> bool:
    parts = set(p.parts)
    if parts & SKIP_DIRS:
        return False
    if p.name in SKIP_FILES:
        return False
    return True


def is_text_file(p: Path) -> bool:
    return p.suffix in TEXT_EXTS


def apply_to_file(p: Path, dry_run: bool) -> tuple[int, dict[str, int]]:
    """Return (total_substitutions, per_rule_counts)."""
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return (0, {})
    out = text
    counts: dict[str, int] = {}
    total = 0
    for pat, repl, label in REPLACEMENTS:
        out, n = pat.subn(repl, out)
        if n:
            counts[label] = counts.get(label, 0) + n
            total += n
    if total > 0 and not dry_run:
        p.write_text(out, encoding="utf-8")
    return total, counts


def walk(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            if should_visit(root) and is_text_file(root):
                files.append(root)
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if not should_visit(p):
                continue
            if not is_text_file(p):
                continue
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    ap.add_argument("--apply", action="store_true", help="Actually rewrite files.")
    ap.add_argument("--stage", choices=list(STAGE_PATHS) + ["all"], default="core",
                    help="Which subtree to rename in this run (default: core).")
    ap.add_argument("--path", action="append", default=[],
                    help="Additional path(s) to include (relative to repo root or absolute).")
    args = ap.parse_args(argv)

    if not (args.dry_run or args.apply):
        ap.error("provide --dry-run or --apply")

    # Resolve paths
    roots: list[Path] = []
    if args.stage == "all":
        for s in STAGE_PATHS.values():
            roots.extend(REPO / p for p in s)
    else:
        roots.extend(REPO / p for p in STAGE_PATHS[args.stage])
    for extra in args.path:
        p = Path(extra)
        if not p.is_absolute():
            p = REPO / p
        roots.append(p)

    print(f"stage: {args.stage} | dry-run: {args.dry_run} | apply: {args.apply}")
    print(f"roots: {[str(p.relative_to(REPO)) for p in roots if p.is_relative_to(REPO)]}")

    files = walk(roots)
    print(f"text files matched: {len(files)}")

    total = 0
    files_modified = 0
    rule_counts: dict[str, int] = {}
    for f in files:
        n, per = apply_to_file(f, dry_run=args.dry_run)
        if n:
            files_modified += 1
            total += n
            for k, v in per.items():
                rule_counts[k] = rule_counts.get(k, 0) + v

    print(f"\nSummary: {total} substitutions across {files_modified} file(s)")
    if rule_counts:
        for label in sorted(rule_counts, key=lambda k: -rule_counts[k]):
            print(f"  {rule_counts[label]:>6}  {label}")

    if args.dry_run:
        print("(dry run — no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
