#!/usr/bin/env python3
# SPDX-License-Identifier: CECILL-2.1
"""Deterministically build the Phase-2 C-ABI symbol rename map from the catalog.

Codex Phase-2 convention (proposals/namespace/codex_phaseR_phase2.md, section B/D):

    old method symbol  ->  n4m_<role>_<leaf><tail>

where
    role = method["namespace"].split(".")[0]   (one of the 12 top-level roles)
    leaf = method["leaf"]
    tail = the operation suffix of the OLD symbol, preserved byte-for-byte
           (e.g. _fit / _create / _destroy / _transform / _select / _split /
            _result_get_best_score / _output_cols / ...).

The hard part is deriving <tail> from each old symbol. The old "subject" (the
part replaced by <role>_<leaf>) is NOT the leaf, so it must be discovered:

  * Multi-symbol method  -> subject = longest common token-prefix of its symbols
    (e.g. n4m_pp_snv for snv create/destroy/transform; n4m_interval for the
    interval generator set; the trailing differing tokens become the tail).
  * Single-symbol method -> subject = symbol with its trailing operation verb
    removed (VERB_TAILS), so the verb becomes the tail. The no-verb utility
    functions and the `pls_fit_simple` irregular are handled by IRREGULAR.

The 10 `c_surface: "none"` methods are skipped entirely: no declaration, no
exported symbol, no snapshot entry, no stub.

Output:
  proposals/namespace/_rename_map.tsv
      old_symbol \t new_symbol \t method_id \t role \t leaf \t tail

Validation (fail-closed):
  * every method symbol in cpp/abi/expected_symbols_linux.txt is mapped exactly once
  * all new symbols are unique (no <role,leaf,tail> collision)
  * symbol count preserved (566)
  * every (non-empty) tail begins with '_'
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog" / "methods.yaml"
LINUX_SNAP = REPO / "cpp" / "abi" / "expected_symbols_linux.txt"
OUT_TSV = REPO / "proposals" / "namespace" / "_rename_map.tsv"

EXPECTED_TOTAL = 566

# Trailing operation verbs for single-symbol methods. The verb is stripped from
# the old symbol to obtain the subject; it then becomes the tail. Order matters:
# longest match first so e.g. `_fit_simple` is tried before `_fit`.
VERB_TAILS = ["_select", "_compute", "_run", "_rank", "_fit"]

# Explicit per-method irregulars Codex named. Maps method_id -> (subject, new_tail).
# `subject` is the leading slice replaced by n4m_<role>_<leaf>; `new_tail` is the
# canonical tail emitted verbatim (overriding any literal old suffix).
#   * pls_fit_simple: drop the "simple" implementation word -> canonical _fit.
#   * the three no-verb utility fns: whole symbol is the subject -> empty tail.
IRREGULAR: dict[str, tuple[str, str]] = {
    "models.pls.pls_fit_simple": ("n4m_pls_fit_simple", "_fit"),
    "utilities.hotelling_t2": ("n4m_util_hotelling_t2", ""),
    "utilities.q_residuals": ("n4m_util_q_residuals", ""),
    "utilities.signal_type_detector": ("n4m_signal_detect", ""),
}


def load_methods() -> list[dict]:
    with open(CATALOG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["methods"]


def lcp_tokens(symbols: list[str]) -> str:
    toks = [s.split("_") for s in symbols]
    n = min(len(t) for t in toks)
    pref: list[str] = []
    for i in range(n):
        col = {t[i] for t in toks}
        if len(col) == 1:
            pref.append(toks[0][i])
        else:
            break
    return "_".join(pref)


def strip_verb(symbol: str) -> tuple[str, str]:
    """Return (subject, tail) for a single-symbol method via trailing-verb strip."""
    for v in VERB_TAILS:
        if symbol.endswith(v):
            return symbol[: -len(v)], v
    raise SystemExit(
        f"ERROR: single-symbol {symbol!r} has no recognized verb tail "
        f"({VERB_TAILS}); add an IRREGULAR entry"
    )


def map_method(method: dict) -> list[tuple[str, str, str, str, str, str]]:
    mid = method["method_id"]
    role = method["namespace"].split(".")[0]
    leaf = method["leaf"]
    syms = method["c_surface"]
    rows: list[tuple[str, str, str, str, str, str]] = []

    if mid in IRREGULAR:
        subject, forced_tail = IRREGULAR[mid]
        if len(syms) != 1:
            raise SystemExit(f"ERROR: IRREGULAR {mid!r} expects 1 symbol, got {len(syms)}")
        old = syms[0]
        if not old.startswith(subject):
            raise SystemExit(f"ERROR: IRREGULAR subject {subject!r} not a prefix of {old!r}")
        new = f"n4m_{role}_{leaf}{forced_tail}"
        return [(old, new, mid, role, leaf, forced_tail)]

    if len(syms) == 1:
        subject, _ = strip_verb(syms[0])
    else:
        subject = lcp_tokens(syms)
        if not subject:
            raise SystemExit(f"ERROR: empty LCP subject for {mid!r} symbols={syms}")

    for old in syms:
        if not old.startswith(subject):
            raise SystemExit(f"ERROR: {mid}: subject {subject!r} not a prefix of {old!r}")
        tail = old[len(subject):]
        if tail and not tail.startswith("_"):
            raise SystemExit(
                f"ERROR: {mid}: tail {tail!r} (from {old!r} - {subject!r}) lacks leading '_'"
            )
        new = f"n4m_{role}_{leaf}{tail}"
        rows.append((old, new, mid, role, leaf, tail))
    return rows


def catalog_is_premigration() -> bool:
    """True while the catalog still carries the OLD terse symbols (the input this
    generator derives from). Once Phase 2 applied the rename, catalog `c_surface`
    holds the new `n4m_<role>_<leaf>...` names, so the OLD->NEW derivation no
    longer has an OLD side to read — the committed `_rename_map.tsv` is then the
    artifact of record and is re-read instead of regenerated."""
    for m in load_methods():
        if m["c_surface"] != "none":
            return not str(m["c_surface"][0]).startswith("n4m_" + m["namespace"].split(".")[0] + "_")
    return True


def load_committed_map() -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    with open(OUT_TSV, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            c = line.rstrip("\n").split("\t")
            rows.append((c[0], c[1], c[2], c[3], c[4], c[5]))
    return rows


def build() -> list[tuple[str, str, str, str, str, str]]:
    if not catalog_is_premigration():
        # Phase 2 rename already applied: the catalog no longer carries the OLD
        # symbol side. Return the committed artifact so re-runs are idempotent
        # and still emit the validated 566-row map.
        print(
            "NOTE: catalog already migrated to ABI-2 symbols; re-reading the "
            "committed _rename_map.tsv instead of regenerating from OLD symbols.",
            file=sys.stderr,
        )
        return load_committed_map()
    rows: list[tuple[str, str, str, str, str, str]] = []
    for m in load_methods():
        if m["c_surface"] == "none":
            continue
        rows.extend(map_method(m))
    return rows


def validate(rows: list[tuple[str, str, str, str, str, str]]) -> None:
    olds = [r[0] for r in rows]
    news = [r[1] for r in rows]

    # The catalog side to compare against the snapshot is OLD pre-migration and
    # NEW post-migration; pick the column that matches the current catalog.
    pre = catalog_is_premigration()
    active = olds if pre else news
    active_label = "old" if pre else "new"

    catalog_syms: set[str] = set()
    for m in load_methods():
        if m["c_surface"] != "none":
            catalog_syms.update(m["c_surface"])
    with open(LINUX_SNAP, encoding="utf-8") as fh:
        snap = {ln.strip() for ln in fh if ln.strip()}
    snap_method = snap & catalog_syms

    fail: list[str] = []
    if len(active) != len(set(active)):
        seen: dict[str, int] = {}
        for o in active:
            seen[o] = seen.get(o, 0) + 1
        fail.append(f"duplicate {active_label} symbols: {[k for k, v in seen.items() if v > 1]}")
    missing = snap_method - set(active)
    if missing:
        fail.append(f"snapshot method symbols not mapped ({len(missing)}): {sorted(missing)}")
    extra = set(active) - snap_method
    if extra:
        fail.append(f"mapped {active_label} symbols absent from snapshot: {sorted(extra)}")
    if len(news) != len(set(news)):
        seen2: dict[str, int] = {}
        for n in news:
            seen2[n] = seen2.get(n, 0) + 1
        fail.append(f"NEW-SYMBOL COLLISION: {{k: v for k, v in ...}} -> {[k for k, v in seen2.items() if v > 1]}")
    if len(rows) != EXPECTED_TOTAL:
        fail.append(f"expected {EXPECTED_TOTAL} mapped symbols, got {len(rows)}")
    if fail:
        for f in fail:
            print("VALIDATION FAIL:", f, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    rows = build()
    validate(rows)
    rows_sorted = sorted(rows, key=lambda r: r[0])
    with open(OUT_TSV, "w", encoding="utf-8") as fh:
        fh.write("old_symbol\tnew_symbol\tmethod_id\trole\tleaf\ttail\n")
        for r in rows_sorted:
            fh.write("\t".join(r) + "\n")
    n_irregular = len(IRREGULAR)
    print(
        f"OK: {len(rows)} symbols mapped, "
        f"{len(set(r[1] for r in rows))} unique new symbols, 0 collisions."
    )
    print(f"    irregulars handled explicitly: {n_irregular}")
    print(f"    wrote {OUT_TSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
