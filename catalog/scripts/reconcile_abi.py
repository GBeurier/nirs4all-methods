#!/usr/bin/env python3
"""Reconcile catalog method ABI symbols against the authoritative ABI snapshot.

The catalog's `abi_symbols` were M2 auto-discovered GUESSES (wrong-by-convention,
e.g. `n4m_augmentation_spline_x_simplification_create` vs the real
`n4m_aug_spline_x_simplify_create`). This tool replaces them with the REAL
exported symbols from `cpp/abi/expected_symbols_linux.txt`, partitioned into:

  - **method symbols** — attributed to exactly one catalog method via an
    explicit, reviewed `{method_id: prefix}` table (`catalog/abi_method_map.yaml`).
    Assignment is by LONGEST-prefix match, so `n4m_split_spxy_fold_*` goes to the
    SPXYFold method, not the SPXY method.
  - **infra/support symbols** — context/config/array/status/version/rng/matrix/…
    that belong to no method, listed explicitly in `catalog/abi_infra.yaml`.

Source of truth (per the review): the ABI snapshot + `n4m.h`/`pls.h` declare the
exported set; the mapping table is a human-reviewed artifact (NOT a runtime
naming heuristic). `--seed` bootstraps a candidate table to curate; `--check`
gates that every real symbol is accounted for; `--write` rewrites
`catalog/methods.yaml` then regenerates the split files.

Usage:
  reconcile_abi.py --seed      # write a candidate abi_method_map.yaml to curate
  reconcile_abi.py --check     # gate: every real symbol attributed (method|infra)
  reconcile_abi.py --write     # apply the curated map to the catalog
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog_loader import REPO, load_methods  # noqa: E402

SNAPSHOT = REPO / "cpp" / "abi" / "expected_symbols_linux.txt"
MAP_YAML = REPO / "catalog" / "abi_method_map.yaml"
MISSING_YAML = REPO / "catalog" / "abi_missing_methods.yaml"
INFRA_YAML = REPO / "catalog" / "abi_infra.yaml"
METHODS_YAML = REPO / "catalog" / "methods.yaml"

# Non-method support surface: context/config/marshalling/version/rng/etc. These
# belong to no operator. Prefix-matched; kept explicit so --check can prove the
# union of (method symbols, infra symbols) equals the full exported set.
INFRA_PREFIXES = (
    "n4m_context_", "n4m_config_", "n4m_array_", "n4m_status_", "n4m_version_",
    "n4m_check_abi_", "n4m_rng_", "n4m_matrix_", "n4m_get_abi_", "n4m_get_version",
    "n4m_get_build", "n4m_get_git", "n4m_backend_", "n4m_dtype_", "n4m_operator_bank_",
    "n4m_signal_type_", "n4m_validation_plan_", "n4m_gating_", "n4m_pipeline_",
    "n4m_serialization_", "n4m_method_result_", "n4m_status_to_string",
    # Shared handles used by ALL methods of a family (not per-method operators):
    "n4m_model_",         # the generic trained-model handle (fit/predict/transform/export/import)
    "n4m_split_result_",  # the shared splitter result handle
)


def _dump(data) -> str:
    """Dump YAML the vendored catalog parser can read: block style with INDENTED
    sequences (`key:\\n  - item`). PyYAML's default emits indentless sequences
    (`key:\\n- item`), which the vendored parser does not associate with the key.
    """
    import yaml

    class _IndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    return yaml.dump(data, Dumper=_IndentDumper, sort_keys=False,
                     default_flow_style=False, width=1000, allow_unicode=True)


def real_symbols() -> list[str]:
    return sorted({ln.strip() for ln in SNAPSHOT.read_text().splitlines()
                   if ln.strip().startswith("n4m_")})


def is_infra(sym: str) -> bool:
    return sym.startswith(INFRA_PREFIXES)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def seed_candidate(methods: list[dict], method_syms: list[str]) -> dict[str, str | None]:
    """Best-effort {method_id: prefix} candidate for curation (NOT authoritative)."""
    prefixes: set[str] = set()
    for s in method_syms:
        parts = s.split("_")
        for k in range(2, len(parts)):
            prefixes.add("_".join(parts[:k]))
    out: dict[str, str | None] = {}
    for m in methods:
        leaf = m["method_id"].split(".")[-1]
        tu = (m.get("tu") or [""])[0]
        keys = {_norm(leaf), _norm(Path(tu).stem if tu else "")}
        keys = {k for k in keys if k}
        best, best_len = None, -1
        for p in prefixes:
            pn = _norm(p.replace("n4m_", ""))
            if any(k in pn or pn.endswith(k) for k in keys):
                cov = sum(1 for s in method_syms if s == p or s.startswith(p + "_"))
                if cov > 0 and len(p) > best_len:
                    best, best_len = p, len(p)
        out[m["method_id"]] = best
    return out


def load_missing() -> list[dict]:
    """New catalog methods (libn4m exports them; M2 discovery missed them)."""
    import yaml
    if not MISSING_YAML.exists():
        return []
    data = yaml.safe_load(MISSING_YAML.read_text()) or {}
    out = []
    for e in data.get("new_methods") or []:
        e = dict(e)
        # Mirror the catalog id convention: 3-segment (cat.sub.name) for
        # augmentation/models/preprocessing; 2-segment (cat.name) otherwise.
        if e["category"] in ("augmentation", "models", "preprocessing"):
            e["method_id"] = f"{e['category']}.{e['subcategory']}.{e['name']}"
        else:
            e["method_id"] = f"{e['category']}.{e['name']}"
        e["family"] = f"{e['category']}.{e['subcategory']}"
        out.append(e)
    return out


def load_map() -> dict[str, str]:
    """Merged {method_id: prefix} from the curated map + the missing-methods table."""
    import yaml  # PyYAML is available in the catalog dev env
    if not MAP_YAML.exists():
        raise SystemExit(f"{MAP_YAML} missing — run `reconcile_abi.py --seed` and curate it.")
    data = yaml.safe_load(MAP_YAML.read_text()) or {}
    # Drop nulls (curated as "no public method symbol" / helper stub).
    mapping = {k: v for k, v in (data.get("method_prefix") or {}).items() if v}
    for e in load_missing():
        mapping[e["method_id"]] = e["prefix"]
    return mapping


def derive_tu(prefix: str) -> list[str]:
    """Implementation source file for a symbol family (a real, existing path —
    validate.py checks it exists). Some symbols are macro-generated (the literal
    symbol never appears in source), so search the prefix, then the operator name,
    in c_api then core; return the defining file path.
    """
    import subprocess
    op = "_".join(prefix.split("_")[2:]) or prefix  # drop n4m_<cat>_
    for needle in (prefix, op):
        for d in ("cpp/src/c_api", "cpp/src/core"):
            r = subprocess.run(["grep", "-rl", needle, str(REPO / d)],
                               capture_output=True, text=True).stdout.strip().splitlines()
            if r:
                return [str(Path(r[0]).relative_to(REPO))]
    return ["cpp/src/c_api/c_api_method_result.cpp"]


def assign(method_syms: list[str], mapping: dict[str, str]) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Longest-prefix assignment of method symbols to method_ids.

    Returns (method_id→symbols, unassigned, conflicts).
    """
    # Build prefix→method_id; longest prefix wins per symbol.
    pref_items = sorted(mapping.items(), key=lambda kv: len(kv[1]), reverse=True)
    by_method: dict[str, list[str]] = {mid: [] for mid in mapping}
    unassigned: list[str] = []
    for s in method_syms:
        match = None
        for mid, pfx in pref_items:
            if s == pfx or s.startswith(pfx + "_"):
                match = mid
                break
        if match is None:
            unassigned.append(s)
        else:
            by_method[match].append(s)
    conflicts: list[str] = []  # prefixes that are themselves prefixes of another (ambiguous)
    prefixes = list(mapping.values())
    for p in prefixes:
        for q in prefixes:
            if p != q and q.startswith(p + "_") and p not in conflicts:
                pass  # nested prefixes are fine (longest-match handles them)
    return by_method, unassigned, conflicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", action="store_true", help="write a candidate abi_method_map.yaml")
    g.add_argument("--check", action="store_true", help="gate: every real symbol accounted for")
    g.add_argument("--write", action="store_true", help="apply the map to catalog/methods.yaml")
    g.add_argument("--add-missing", action="store_true",
                   help="append the abi_missing_methods.yaml entries to catalog/methods.yaml")
    args = ap.parse_args(argv)

    real = real_symbols()
    infra = [s for s in real if is_infra(s)]
    method_syms = [s for s in real if not is_infra(s)]
    methods = load_methods(prefer_split=True)

    if args.seed:
        import yaml
        cand = seed_candidate(methods, method_syms)
        body = {"# NOTE": "Reviewed mapping: method_id -> ABI symbol prefix. null = no public "
                          "method symbol (helper stub / config-side). Curate by hand.",
                "method_prefix": {k: cand[k] for k in sorted(cand)}}
        MAP_YAML.write_text(yaml.safe_dump(body, sort_keys=False, default_flow_style=False))
        miss = [k for k, v in cand.items() if not v]
        print(f"seeded {MAP_YAML} ({len(cand)} methods, {len(miss)} need manual prefix)")
        return 0

    mapping = load_map()
    by_method, unassigned, _ = assign(method_syms, mapping)
    empty = [mid for mid, syms in by_method.items() if not syms]

    if args.add_missing:
        import yaml
        missing = load_missing()
        data = yaml.safe_load(METHODS_YAML.read_text())
        existing = {m["method_id"] for m in data["methods"]}
        added = 0
        for e in missing:
            if e["method_id"] in existing:
                continue
            data["methods"].append({
                "method_id": e["method_id"], "family": e["family"],
                "category": e["category"], "subcategory": e["subcategory"],
                "since_abi": "1.0.0",
                "abi_symbols": sorted(by_method.get(e["method_id"], [])),
                "tu": derive_tu(e["prefix"]),
                "headers": ["cpp/include/n4m/n4m.h"],
                "carry_along_tu": ["cpp/src/core/common/context.cpp",
                                   "cpp/src/core/common/status.cpp",
                                   "cpp/src/core/common/matrix_view.cpp",
                                   "cpp/src/core/common/version.cpp"],
                "parity": {"fixtures": [], "tolerance_row": "tbd",
                           "divergence_id": None, "references": [], "tolerances": []},
                "bench": {"bench_id": e["method_id"], "registry_entry": None},
                "feature_flags": [], "vendored_licenses": [], "bindings": {},
                "publications": [],
                "notes": "Added in Phase-B ABI reconciliation — exported by libn4m, "
                         "missed by the M2 auto-discovery (see catalog/ABI_RECONCILE_GAP.md).",
            })
            added += 1
        data["method_count"] = len(data["methods"])
        METHODS_YAML.write_text(_dump(data))
        print(f"added {added} methods; method_count={data['method_count']}")
        print("next: split_legacy_methods.py --write && reconcile_abi.py --check && validate.py --strict-abi")
        return 0

    if args.check:
        ok = True
        if unassigned:
            ok = False
            print(f"FAIL: {len(unassigned)} method symbols unattributed:", file=sys.stderr)
            for s in unassigned[:40]:
                print(f"  {s}", file=sys.stderr)
        if empty:
            ok = False
            print(f"FAIL: {len(empty)} mapped methods captured no symbol: {empty}", file=sys.stderr)
        covered = sum(len(v) for v in by_method.values())
        print(f"method symbols: {covered} attributed / {len(method_syms)} total; "
              f"infra: {len(infra)}; union vs snapshot: {covered + len(infra)}/{len(real)}")
        if covered + len(infra) != len(real):
            ok = False
            print("FAIL: union(method, infra) != snapshot", file=sys.stderr)
        return 0 if ok else 1

    if args.write:
        import yaml
        INFRA_YAML.write_text(_dump({"infra_abi_symbols": infra}))
        # Rewrite EVERY method's abi_symbols from the assignment: real methods get
        # their symbols, helper/concept stubs (null-mapped) get [] — clearing the
        # old M2 guesses so no unreal symbol remains. Then regenerate split files.
        data = yaml.safe_load(METHODS_YAML.read_text())
        kept, dropped = [], []
        for entry in data["methods"]:
            syms = sorted(by_method.get(entry["method_id"], []))
            if not syms:
                # Helper/concept stub with no public ABI symbol — REMOVE it
                # (the schema requires abi_symbols minItems=1; Codex: remove
                # helper stubs rather than keep symbol-less entries).
                dropped.append(entry["method_id"])
                continue
            entry["abi_symbols"] = syms
            kept.append(entry)
        data["methods"] = kept
        data["method_count"] = len(kept)
        METHODS_YAML.write_text(_dump(data))
        print(f"wrote {len(kept)} methods with real abi_symbols; dropped "
              f"{len(dropped)} helper stubs: {dropped}")
        print("now run: python catalog/scripts/split_legacy_methods.py --write")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
