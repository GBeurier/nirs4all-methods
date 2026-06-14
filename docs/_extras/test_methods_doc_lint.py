"""Doc-lint: no ABI-1 C symbol *or family prefix* may survive in a page.

The namespace migration renamed every public C symbol to ABI-2
(`n4m_<role>_<leaf><tail>`). The generated `docs/methods/*.md` pages must only
ever name ABI-2 symbols. This test fails if either of the following survives in
any generated method page:

  * an exact ABI-1 function symbol — any column-1 (`old_symbol`) entry of
    `proposals/namespace/_rename_map.tsv`, matched on a word boundary;
  * an ABI-1 family / method *prefix* and its `<prefix>_*` wildcard form — the
    per-method prefix obtained by stripping the trailing operation verb from
    each old symbol, plus the coarse legacy families (`n4m_pp_`, `n4m_aug_`,
    `n4m_split_`, `n4m_aom_`). The operator pages publish their "C ABI" line as
    such a wildcard, which the exact-symbol check alone would miss.

`*_t` / `*_result_t` struct TYPE names legitimately kept their ABI-1 spelling
in the ABI-2 headers (they are types, not function symbols) and are
allow-listed.

CI runs this through pytest; locally `build_methods.py --strict` performs the
same assertion during regeneration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_methods import (  # noqa: E402
    METHODS_DIR,
    RENAME_MAP_TSV,
    build_old_prefix_set,
    lint_generated_pages,
    load_symbol_old_to_new,
)


def test_no_abi1_symbols_in_method_pages() -> None:
    old_to_new = load_symbol_old_to_new(RENAME_MAP_TSV)
    assert old_to_new, f"empty/missing rename map: {RENAME_MAP_TSV}"

    findings = lint_generated_pages(METHODS_DIR, old_to_new)

    msg = "ABI-1 C symbols/prefixes survive in generated method pages:\n" + (
        "\n".join(
            f"  {page}: {', '.join(syms)}" for page, syms in findings.items()
        )
    )
    assert not findings, msg


def test_lint_flags_wildcard_prefix(tmp_path: Path) -> None:
    """The strengthened lint must catch an injected ABI-1 `<prefix>_*` wildcard
    (not just exact symbols) and must allow-list `*_t` type names."""
    old_to_new = load_symbol_old_to_new(RENAME_MAP_TSV)

    # A page advertising a dead ABI-1 family prefix as a wildcard.
    bad = tmp_path / "split_kennard_stone.md"
    bad.write_text(
        "# kennard stone\n_C ABI_: `n4m_split_kennard_stone_*`\n",
        encoding="utf-8",
    )
    findings = lint_generated_pages(tmp_path, old_to_new)
    assert "split_kennard_stone.md" in findings
    assert any(
        "n4m_split_kennard_stone" in hit
        for hit in findings["split_kennard_stone.md"]
    )

    # `*_t` / `*_result_t` type names must NOT be flagged.
    typ = tmp_path / "type_name.md"
    typ.write_text(
        "n4m_aom_global_result_t* res = NULL;\n", encoding="utf-8"
    )
    findings = lint_generated_pages(tmp_path, old_to_new)
    assert "type_name.md" not in findings


def test_old_prefix_set_covers_families() -> None:
    """The forbidden-prefix set includes both per-method prefixes (from the
    rename map) and the coarse legacy families."""
    old_to_new = load_symbol_old_to_new(RENAME_MAP_TSV)
    prefixes = build_old_prefix_set(old_to_new)
    # coarse families
    for fam in ("n4m_pp", "n4m_aug", "n4m_split", "n4m_aom"):
        assert fam in prefixes
    # a per-method prefix derived by stripping the trailing verb
    assert "n4m_split_kennard_stone" in prefixes
    assert "n4m_pp_snv" in prefixes
