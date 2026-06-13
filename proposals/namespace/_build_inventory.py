#!/usr/bin/env python3
"""Build the machine-readable method inventory for namespace design.

Walks catalog/methods/*.yaml (208 files), parses the dotted-taxonomy filename
into {id, category, subcategory, method}, and extracts the available fields.

The catalog schema has NO explicit top-level `kind`/`domain`/`status`/`aliases`
fields. We therefore derive:
  - `kind`   : from the dominant ABI-symbol verb suffix (create/transform/fit/
               select/split/apply/...) — the operational shape of the method.
               Falls back to the category when no symbols exist.
  - `domain` : the C ABI header the method lands in (cpp/include/n4m/<x>.h),
               which is the de-facto domain grouping today.
  - `aliases`: bindings.python.legacy_aliases (the only alias source present).
Anything underivable is set to null; no method is ever dropped.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

import yaml

CAT_DIR = "/home/delete/nirs4all/nirs4all-methods/catalog/methods"
OUT_JSON = "/home/delete/nirs4all/nirs4all-methods/proposals/namespace/_method_inventory.json"
OUT_MD = "/home/delete/nirs4all/nirs4all-methods/proposals/namespace/_method_inventory.md"

# ABI-symbol verb suffix -> operational "kind". Checked in order against the
# trailing token(s) of each n4m_* symbol; first matching symbol-group wins.
VERB_KIND = [
    ("transform", "stateful_transformer"),  # *_create + *_transform (+ *_destroy)
    ("apply", "applier"),                    # augmenters/filters: *_apply
    ("split", "splitter"),
    ("select", "selector"),
    ("fit", "estimator"),
]


def derive_kind(symbols: list[str], category: str) -> str | None:
    """Operational kind from the set of ABI symbol verb suffixes."""
    suffixes = set()
    for s in symbols:
        # last underscore-delimited token, e.g. n4m_pp_snv_transform -> transform
        suffixes.add(s.rsplit("_", 1)[-1])
    # stateful transformer = has both create and transform
    if "transform" in suffixes:
        return "stateful_transformer"
    if "apply" in suffixes and "create" in suffixes:
        return "applier"
    if "split" in suffixes:
        return "splitter"
    if "select" in suffixes:
        return "selector"
    if "fit" in suffixes:
        return "estimator"
    # metric_* / util_* pure functions, or getters-only result objects
    if symbols and all(s.startswith("n4m_metric_") for s in symbols):
        return "metric"
    if symbols and all(s.startswith("n4m_util_") or s.startswith("n4m_transfer_") for s in symbols):
        return "utility_fn"
    if not symbols:
        return None  # AOM superblocks etc. that are Python-only / not yet exported
    # fall back to category-level grouping
    return {
        "diagnostics": "metric",
        "utilities": "utility_fn",
        "aom_pop": "aom_operator",
    }.get(category)


def header_to_domain(headers: list[str]) -> str | None:
    """The C ABI header file the method lands in == today's de-facto domain."""
    if not headers:
        return None
    # strip path + .h
    base = os.path.basename(headers[0])
    return re.sub(r"\.h$", "", base)


def parse_filename(fname: str) -> tuple[str, str, str | None, str | None]:
    """category.subcategory.method.yaml -> (id, category, subcategory, method)."""
    stem = re.sub(r"\.yaml$", "", fname)
    parts = stem.split(".")
    category = parts[0]
    if len(parts) >= 3:
        subcategory = parts[1]
        method = ".".join(parts[2:])
    elif len(parts) == 2:
        subcategory = None
        method = parts[1]
    else:
        subcategory = None
        method = None
    return stem, category, subcategory, method


def main() -> None:
    files = sorted(glob.glob(os.path.join(CAT_DIR, "*.yaml")))
    records = []
    for path in files:
        fname = os.path.basename(path)
        with open(path) as fh:
            doc = yaml.safe_load(fh)

        fid, fcat, fsub, fmeth = parse_filename(fname)

        # Prefer the explicit method_id if present; else the filename stem.
        method_id = doc.get("method_id") or fid
        # category/subcategory: prefer explicit fields, fall back to filename parse.
        category = doc.get("category") or fcat
        subcategory = doc.get("subcategory") or fsub

        symbols = list(doc.get("abi_symbols") or [])
        headers = list(doc.get("headers") or [])

        # aliases: only source present is bindings.python.legacy_aliases
        aliases = None
        bindings = doc.get("bindings") or {}
        py = bindings.get("python") if isinstance(bindings, dict) else None
        if isinstance(py, dict) and py.get("legacy_aliases"):
            aliases = list(py["legacy_aliases"])

        py_class = py.get("class") if isinstance(py, dict) else None
        py_module = py.get("module") if isinstance(py, dict) else None

        parity = doc.get("parity") or {}
        bench = doc.get("bench") or {}

        rec = {
            "id": method_id,
            "category": category,
            "subcategory": subcategory,
            "method": fmeth,
            "family": doc.get("family"),
            "kind": derive_kind(symbols, category),
            "domain": header_to_domain(headers),
            "abi_symbol": symbols[0] if symbols else None,
            "abi_symbols": symbols,
            "abi_symbol_count": len(symbols),
            "header": headers[0] if headers else None,
            "since_abi": doc.get("since_abi"),
            "aliases": aliases,
            "python_module": py_module,
            "python_class": py_class,
            "has_python_binding": bool(py),
            "references": list(parity.get("references") or []) or None,
            "divergence_id": parity.get("divergence_id"),
            "publications": list(doc.get("publications") or []) or None,
            "bench_registry_entry": bench.get("registry_entry"),
            "filename": fname,
            "notes": doc.get("notes"),
        }
        records.append(rec)

    records.sort(key=lambda r: r["id"])

    with open(OUT_JSON, "w") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # ---- readable tree ----
    tree: dict[str, dict[str, list[dict]]] = collections.OrderedDict()
    for r in records:
        c = r["category"]
        s = r["subcategory"] or "(none)"
        tree.setdefault(c, collections.OrderedDict()).setdefault(s, []).append(r)

    cat_counts = {c: sum(len(v) for v in subs.values()) for c, subs in tree.items()}

    lines = []
    lines.append("# Method catalog inventory")
    lines.append("")
    lines.append(f"Total methods: **{len(records)}** "
                 f"(parsed from `catalog/methods/*.yaml`).")
    lines.append("")
    lines.append("Generated by `_build_inventory.py`. Machine-readable twin: "
                 "`_method_inventory.json`.")
    lines.append("")
    lines.append("## Counts by top-level category")
    lines.append("")
    lines.append("| category | methods |")
    lines.append("|---|---:|")
    for c in sorted(cat_counts, key=lambda k: -cat_counts[k]):
        lines.append(f"| `{c}` | {cat_counts[c]} |")
    lines.append(f"| **total** | **{len(records)}** |")
    lines.append("")
    lines.append("## category . subcategory . method tree")
    lines.append("")
    for c in sorted(tree, key=lambda k: -cat_counts[k]):
        lines.append(f"### `{c}` ({cat_counts[c]})")
        lines.append("")
        for s in sorted(tree[c]):
            members = sorted(tree[c][s], key=lambda r: r["id"])
            lines.append(f"- **{c}.{s}** ({len(members)})")
            for r in members:
                kind = r["kind"] or "?"
                dom = r["domain"] or "?"
                sym = r["abi_symbol"] or "(no abi symbol)"
                alias = ""
                if r["aliases"]:
                    alias = f" — alias: {', '.join(r['aliases'])}"
                lines.append(
                    f"  - `{r['method']}` — kind=`{kind}` domain=`{dom}` "
                    f"abi=`{sym}` (+{r['abi_symbol_count']} syms){alias}"
                )
        lines.append("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines))

    # ---- console summary for the agent ----
    print(f"records written: {len(records)}")
    print(f"JSON: {OUT_JSON}")
    print(f"MD:   {OUT_MD}")
    print("category counts:", json.dumps(cat_counts, sort_keys=True))
    kinds = collections.Counter(r["kind"] for r in records)
    print("kind distribution:", dict(kinds))
    domains = collections.Counter(r["domain"] for r in records)
    print("domain (header) distribution:", dict(domains))
    null_kind = [r["id"] for r in records if r["kind"] is None]
    print("null-kind ids:", null_kind)
    # sanity: every record has id/category and no method dropped
    assert len(records) == len(files), "method count mismatch!"
    assert all(r["id"] for r in records)
    assert all(r["category"] for r in records)


if __name__ == "__main__":
    main()
