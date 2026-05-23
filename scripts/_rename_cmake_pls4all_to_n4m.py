"""One-shot Phase-A12/A13 rename: bare `pls4all` / `PLS4ALL` / `Pls4all`
identifiers in CMake files → `n4m` / `N4M` / `N4m`.

The first migration pass (`scripts/migrate_p4a_to_n4m.py`) only touched
CMake *target* names (`pls4all_core`, `pls4all_c`, etc.). The CMake
plumbing also has:

  - cache variables (`PLS4ALL_WITH_BLAS`, `PLS4ALL_BUILD_SHARED`, ...)
  - utility function names (`pls4all_add_warnings`, ...)
  - project name (`project(pls4all ...)`)
  - export sets (`EXPORT pls4all_targets`)
  - install include dir (`<prefix>/include/pls4all`)
  - the `pls4all`-prefixed local variables (`_pls4all_*`)

This script walks the CMake plumbing and rewrites all of those. It
skips the historical project URL on GitHub.

Removed from the repo after Phase A lands.
"""

from __future__ import annotations
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FILES = [
    REPO / "CMakeLists.txt",
    REPO / "CMakePresets.json",
    REPO / "cmake" / "N4mConfig.cmake.in",
    REPO / "cmake" / "N4mOptions.cmake",
    REPO / "cmake" / "SanitizersToolchain.cmake",
    REPO / "cmake" / "CompilerWarnings.cmake",
    REPO / "cpp" / "src" / "CMakeLists.txt",
    REPO / "cpp" / "src" / "n4m_targets.cmake",
    REPO / "cpp" / "tests" / "CMakeLists.txt",
]

REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # Cache variable prefix (must run before bare token rename)
    (re.compile(r"\bPLS4ALL_"), "N4M_"),
    # CamelCase config / module prefix used in cmake/Pls4all*.cmake filenames
    (re.compile(r"\bPls4all"), "N4m"),
    # lowercase identifier prefix: function names, variable names
    (re.compile(r"\bpls4all_"), "n4m_"),
    (re.compile(r"\b_pls4all_"), "_n4m_"),
    # Bare project token / paths
    (re.compile(r"\bpls4all\b"), "n4m"),
]

URL_KEEP = "github.com/GBeurier/pls4all"


def transform(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        # Preserve historical GitHub URL substring.
        if URL_KEEP in line:
            head, _, tail = line.partition(URL_KEEP)
            new_head = head
            new_tail = tail
            for pat, repl in REPLACEMENTS:
                new_head = pat.sub(repl, new_head)
                new_tail = pat.sub(repl, new_tail)
            out.append(f"{new_head}{URL_KEEP}{new_tail}")
            continue
        new = line
        for pat, repl in REPLACEMENTS:
            new = pat.sub(repl, new)
        out.append(new)
    return "".join(out)


def main() -> int:
    total = 0
    for f in FILES:
        if not f.is_file():
            continue
        src = f.read_text(encoding="utf-8")
        new = transform(src)
        if new != src:
            f.write_text(new, encoding="utf-8")
            total += 1
    print(f"rewrote {total} CMake file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
