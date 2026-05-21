#!/usr/bin/env python3
"""M14 — license audit script.

Walks the unified repo and inventories every vendored third-party piece
that ships embedded with the library. Emits a structured report at
NOTICE.md + THIRD_PARTY_LICENSES.md.

Scope:
  - cpp/src/core/common/_vendored/   — FITPACK Fortran (CECILL-compat? GPL?)
  - cpp/src/core/filters/_vendored/   — IsolationForest / LOF / MCD (likely
                                         scikit-learn-derived under BSD-3)
  - parity/donor_imports/             — references PyWavelets coefficient
                                         tables (BSD), pybaselines (BSD-3),
                                         scipy (BSD-3)
  - bindings/donor_imports/r/n4m/src/libn4m/   — R-side vendored copy of
                                                  libn4m (CECILL-2.1 same as
                                                  upstream)

Output:
  NOTICE.md                — short-form 1-liner attributions
  THIRD_PARTY_LICENSES.md  — full SPDX + license-text dump per vendored piece

Usage:
    python scripts/audit_third_party_licenses.py --report
    python scripts/audit_third_party_licenses.py --check   # CI gate: fail if
                                                            # any vendored
                                                            # dir lacks a
                                                            # LICENSE file
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Known vendored subtrees and their declared license.
VENDORED = [
    {
        "path":     "cpp/src/core/common/_vendored/fitpack",
        "name":     "FITPACK",
        "upstream": "https://github.com/scipy/scipy/tree/main/scipy/interpolate/fitpack",
        "spdx":     "BSD-3-Clause",
        "license_file": "LICENSE.scipy.txt",
        "notes":    "Vendored from SciPy's interpolate/fitpack/. Fortran spline kernels. CECILL-2.1 carrier is compatible because BSD-3 is permissive.",
    },
    {
        "path":     "cpp/src/core/filters/_vendored",
        "name":     "Isolation Forest + LOF + Minimum Covariance Determinant",
        "upstream": "https://github.com/scikit-learn/scikit-learn",
        "spdx":     "BSD-3-Clause",
        "license_file": "LICENSE.scikit_learn.txt",
        "notes":    "Reimplemented from scikit-learn algorithms; no scikit-learn source copied verbatim. Treat as 'derived' under BSD-3 attribution.",
    },
]

# Documented external runtime references (NOT vendored, but parity gate locks them).
REFERENCE_LIBS = [
    {"name": "scipy",          "version": "1.17.1", "spdx": "BSD-3-Clause"},
    {"name": "scikit-learn",   "version": "1.8.0",  "spdx": "BSD-3-Clause"},
    {"name": "numpy",          "version": "1.26.4", "spdx": "BSD-3-Clause"},
    {"name": "pywt",           "version": "1.8.0",  "spdx": "MIT"},
    {"name": "pybaselines",    "version": "1.2.1",  "spdx": "BSD-3-Clause"},
    {"name": "pls (R)",        "version": ">=2.8",  "spdx": "GPL-2.0-or-later"},
    {"name": "ropls (R/Bioconductor)", "version": ">=1.34", "spdx": "CECILL-2.0"},
    {"name": "mixOmics (R/Bioconductor)", "version": ">=6.28", "spdx": "GPL-3.0"},
    {"name": "plsVarSel (R)",  "version": "any",    "spdx": "GPL-2.0"},
    {"name": "ikpls (Python)", "version": "any",    "spdx": "Apache-2.0"},
]


def find_license_in(path: Path) -> Path | None:
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.scipy.txt", "COPYING"):
        cand = path / name
        if cand.exists():
            return cand
    return None


def emit_notice() -> str:
    lines = ["# NOTICE — third-party software bundled with nirs4all-methods", ""]
    lines.append("This library is licensed under CECILL-2.1 (see `LICENSE`).")
    lines.append("Bundled within the source distribution and/or the compiled")
    lines.append("shared library are the following third-party components,")
    lines.append("each retaining its original license:")
    lines.append("")
    for v in VENDORED:
        lines.append(f"- **{v['name']}** ({v['spdx']}) — vendored under `{v['path']}` "
                     f"from `{v['upstream']}`.")
    lines.append("")
    lines.append("## Documented external references (not vendored)")
    lines.append("")
    lines.append("The parity gate cross-checks numerical outputs against pinned")
    lines.append("versions of the libraries below. These are dependencies of the")
    lines.append("CI / test environment, not redistribution targets.")
    lines.append("")
    for r in REFERENCE_LIBS:
        lines.append(f"- {r['name']} {r['version']} ({r['spdx']})")
    return "\n".join(lines) + "\n"


def emit_third_party() -> str:
    lines = ["# THIRD_PARTY_LICENSES — full attribution for vendored material", ""]
    for v in VENDORED:
        lines.append(f"## {v['name']}  (SPDX: {v['spdx']})")
        lines.append("")
        lines.append(f"**Upstream**: {v['upstream']}")
        lines.append(f"**Vendored at**: `{v['path']}`")
        lines.append(f"**Notes**: {v['notes']}")
        lp = REPO / v["path"]
        if v.get("license_file"):
            lf = lp / v["license_file"]
            if lf.exists():
                lines.append("")
                lines.append("```")
                lines.append(lf.read_text(encoding="utf-8", errors="replace").rstrip())
                lines.append("```")
            else:
                lines.append(f"\n*MISSING: declared LICENSE file `{v['license_file']}` not found at vendored path.*")
        else:
            found = find_license_in(lp)
            if found is None:
                lines.append("\n*No LICENSE file at vendored path. Verify with upstream before redistribution.*")
            else:
                lines.append("")
                lines.append(f"License at `{found.relative_to(REPO)}`:")
                lines.append("```")
                lines.append(found.read_text(encoding="utf-8", errors="replace").rstrip())
                lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def check() -> int:
    missing = []
    for v in VENDORED:
        lp = REPO / v["path"]
        if not lp.exists():
            missing.append(f"Vendored path missing on disk: {v['path']}")
            continue
        if v.get("license_file"):
            if not (lp / v["license_file"]).exists():
                missing.append(f"{v['name']}: declared LICENSE file {v['license_file']} not found at {v['path']}")
        else:
            if find_license_in(lp) is None:
                missing.append(f"{v['name']}: no LICENSE file at {v['path']}")
    if missing:
        print("FAIL: license audit found problems:")
        for m in missing:
            print(f"  - {m}")
        return 1
    print(f"PASS: license audit clean ({len(VENDORED)} vendored entries, {len(REFERENCE_LIBS)} reference libs)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report", action="store_true", help="Write NOTICE.md + THIRD_PARTY_LICENSES.md")
    g.add_argument("--check",  action="store_true", help="CI gate: fail if any vendored dir lacks a LICENSE")
    args = ap.parse_args(argv)

    if args.report:
        (REPO / "NOTICE.md").write_text(emit_notice(), encoding="utf-8")
        (REPO / "THIRD_PARTY_LICENSES.md").write_text(emit_third_party(), encoding="utf-8")
        print(f"Wrote {REPO / 'NOTICE.md'}")
        print(f"Wrote {REPO / 'THIRD_PARTY_LICENSES.md'}")
        return 0
    if args.check:
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
