"""One-shot Phase-A10 rename: pls4all -> n4m inside bindings/r/n4m/.

Token-level only. Walks the R binding tree and rewrites pls4all_*
function names, useDynLib() target, R_init symbol, Makevars cxx source
references, and the DESCRIPTION/NAMESPACE package name. Leaves the
historical project URL on github (github.com/GBeurier/...) alone — the
repo rename is out of scope here.

Idempotent: re-running is a no-op once tokens are migrated.

Removed from the repo after Phase A lands.
"""

from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "bindings" / "r" / "n4m"

# Order matters: more-specific patterns first.
REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # filename references
    (re.compile(r"\bpls4all_cxx\.cpp\b"), "n4m_cxx.cpp"),
    (re.compile(r"\bpls4all_cxx\.o\b"),   "n4m_cxx.o"),
    (re.compile(r"\bpls4all\.so\b"),      "n4m.so"),
    # R-init / namespace
    (re.compile(r"\bR_init_pls4all\b"),         "R_init_n4m"),
    (re.compile(r"useDynLib\(pls4all\b"),       "useDynLib(n4m"),
    (re.compile(r"^Package:\s*pls4all\b", re.MULTILINE), "Package: n4m"),
    # S3 class strings of the form summary.pls4all_*
    (re.compile(r"\bsummary\.pls4all_"),         "summary.n4m_"),
    # C-side helper symbols r_pls4all_*
    (re.compile(r"\br_pls4all_"),               "r_n4m_"),
    # Generic pls4all_<ident> (functions, S3 classes, type names)
    (re.compile(r"\bpls4all_"),                 "n4m_"),
    # pls4all-package (dash form, used in roxygen tags and Rd metadata)
    (re.compile(r"\bpls4all-package\b"),        "n4m-package"),
    # roxygen / Rd alias + name when referenced standalone
    (re.compile(r"@aliases\s+pls4all\b"),       "@aliases n4m"),
    (re.compile(r"@useDynLib\s+pls4all\b"),     "@useDynLib n4m"),
    (re.compile(r"\\alias\{pls4all\}"),          r"\\alias{n4m}"),
    (re.compile(r"\\name\{pls4all-package\}"),  r"\\name{n4m-package}"),
    # .Call(..., PACKAGE = "pls4all")
    (re.compile(r'PACKAGE\s*=\s*"pls4all"'),    'PACKAGE = "n4m"'),
    # Strings inside source markers
    (re.compile(r'"pls4all-cran"'),             '"n4m-cran"'),
    # Namespace-qualified test/doc references
    (re.compile(r"\bpls4all::"),                "n4m::"),
    # Loaded-DLL keys + bare "pls4all" string keys (NOT inside github URLs)
    (re.compile(r'\["pls4all"\]'),              '["n4m"]'),
    (re.compile(r'getLoadedDLLs\(\)\[\["pls4all"\]\]'), 'getLoadedDLLs()[["n4m"]]'),
]

# Bare-`pls4all` final sweep applied after the other rules. Skips lines
# that look like the historical project URL on GitHub, since the repo
# rename is out of scope for this script.
BARE_PLS4ALL = re.compile(r"\bpls4all\b")


def replace_bare(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if "github.com/GBeurier/pls4all" in line:
            out_lines.append(line)
            continue
        if '"pls4all", "contributors"' in line:
            # historical Authors@R contributor tag, kept intentionally
            out_lines.append(line)
            continue
        out_lines.append(BARE_PLS4ALL.sub("n4m", line))
    return "".join(out_lines)

TEXT_EXTS = {
    ".R", ".r", ".Rd", ".Rmd",
    ".c", ".h", ".cpp", ".hpp",
    ".md", ".txt",
    "",  # DESCRIPTION, NAMESPACE
}

SKIP_DIRS = {".git", "__pycache__", "build"}


def is_text(p: Path) -> bool:
    if p.suffix in TEXT_EXTS:
        return True
    if p.name in {"DESCRIPTION", "NAMESPACE", "Makevars", "Makevars.win"}:
        return True
    return False


def main() -> int:
    files = [p for p in ROOT.rglob("*") if p.is_file() and is_text(p)
             and not set(p.parts) & SKIP_DIRS]
    total = 0
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = src
        for pat, repl in REPLACEMENTS:
            new = pat.sub(repl, new)
        new = replace_bare(new)
        if new != src:
            f.write_text(new, encoding="utf-8")
            total += 1
    print(f"rewrote {total} files under {ROOT.relative_to(ROOT.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
