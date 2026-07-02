# SPDX-License-Identifier: CECILL-2.1
"""Guard: user-facing binding READMEs must document the CURRENT C ABI version.

The JS and Python binding READMEs show example ``version()`` / ``abi_version()``
output. Those illustrative values silently rot on an ABI bump — they were found
stale at ``abi.1.9.0`` while the engine had already moved to ``2.0.0``. This
test pins every ``+abi.<major>.<minor>.<patch>`` string and the documented ABI
triple in those READMEs to the canonical macros in
``cpp/include/n4m/n4m_version.h``, so a future ABI bump forces the docs to move
with it.

Pure file parsing — no libn4m / NumPy required, so it runs in every gate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HEADER = REPO / "cpp/include/n4m/n4m_version.h"

# READMEs that document version()/abi_version() example output.
BINDING_READMES = [
    REPO / "bindings/js/README.md",
    REPO / "bindings/python/README.md",
]


def _header_abi() -> tuple[int, int, int]:
    text = HEADER.read_text(encoding="utf-8")

    def macro(name: str) -> int:
        match = re.search(rf"#define\s+{name}\s+(\d+)", text)
        assert match is not None, f"{name} not found in {HEADER}"
        return int(match.group(1))

    return (
        macro("N4M_ABI_VERSION_MAJOR"),
        macro("N4M_ABI_VERSION_MINOR"),
        macro("N4M_ABI_VERSION_PATCH"),
    )


@pytest.mark.parametrize("readme", BINDING_READMES, ids=lambda p: p.parent.name)
def test_readme_abi_string_matches_header(readme: Path) -> None:
    """Every ``+abi.<x.y.z>`` string in the README must equal the header."""
    major, minor, patch = _header_abi()
    expected = f"{major}.{minor}.{patch}"
    text = readme.read_text(encoding="utf-8")
    abi_strings = re.findall(r"abi\.(\d+\.\d+\.\d+)", text)
    assert abi_strings, f"{readme} documents no `+abi.<x.y.z>` version string"
    stale = sorted({s for s in abi_strings if s != expected})
    assert not stale, (
        f"{readme} documents stale ABI version(s) {stale}; "
        f"cpp/include/n4m/n4m_version.h is {expected}"
    )


@pytest.mark.parametrize("readme", BINDING_READMES, ids=lambda p: p.parent.name)
def test_readme_abi_triple_matches_header(readme: Path) -> None:
    """The documented ``abi_version()``/``abiVersion()`` triple must match."""
    major, minor, patch = _header_abi()
    text = readme.read_text(encoding="utf-8")
    # Target the abiVersion()/abi_version() example line specifically, so an
    # unrelated 3-tuple elsewhere in the README cannot trip (or hide) the check.
    match = re.search(
        r"abi_?[vV]ersion\(\)[^\n]*?[\[(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
        text,
    )
    assert match is not None, (
        f"{readme} documents no abiVersion()/abi_version() example triple"
    )
    triple = tuple(int(x) for x in match.groups())
    assert triple == (major, minor, patch), (
        f"{readme} documents ABI triple {triple}; "
        f"cpp/include/n4m/n4m_version.h is {(major, minor, patch)}"
    )
