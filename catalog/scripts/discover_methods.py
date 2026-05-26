#!/usr/bin/env python3
"""Walk both source trees and emit catalog/methods.yaml.

Runs once per release-prep cycle; it is not a build-time tool. Output is a
hand-curated starting point — maintainers may refine entries afterwards
(adding bench_ids, references, publications). Re-running this script does
NOT clobber such refinements: it writes catalog/methods.discovered.yaml
which then has to be merge-promoted into catalog/methods.yaml by a human
(or by catalog/scripts/merge_discovered.py later).
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]  # repo root
DONOR_PREFIX = "_donor/nirs4all-methods"


def file_to_method_id(rel_path: str) -> tuple[str | None, str, str, str]:
    """(method_id, category, subcategory, family) or (None, ...) if not a method file.

    Static rules:
    - cpp/src/core/<category>/<sub?>/<name>.{c,cpp} → donor-style nested layout
      (this is where moved donor sources live post-M3).
    - cpp/src/core/<name>.cpp (flat) → pls4all-style, mapped via PLS4ALL_MAP.

    Note: the historical `_donor/nirs4all-methods/` prefix is no longer
    expected — M3 lifted everything out. Any leftover path under
    `_donor/.../cpp/src/core/` is treated as archeology (skipped).
    """
    name = Path(rel_path).name
    stem = Path(rel_path).stem

    # Skip donor archeology (anything still living under _donor/)
    if rel_path.startswith(f"{DONOR_PREFIX}/"):
        return (None, "", "", "")

    if not rel_path.startswith("cpp/src/core/"):
        return (None, "", "", "")

    parts = rel_path.split("/")
    # parts = ['cpp','src','core','<category>',('<sub>',)'<file>'] for nested layout
    # parts = ['cpp','src','core','<file>']                         for flat layout

    rest = parts[3:]  # everything from <category> onward
    if not rest:
        return (None, "", "", "")

    # Skip vendored utility subtrees at any depth (not public methods).
    if any(p == "_vendored" for p in rest):
        return (None, "", "", "")

    if len(rest) == 1:
        # Flat pls4all-style file at cpp/src/core/<file>.cpp
        m = PLS4ALL_MAP.get(stem)
        if m is None:
            return (None, "", "", "")
        category, sub, mid_override = m
        if mid_override is None:
            return (None, "", "", "")
        family = f"{category}.{sub}" if sub else category
        return (mid_override, category, sub, family)

    # Nested donor-style file under cpp/src/core/<category>/...
    head = rest[0]
    # Defensive — the M3 move was supposed to singularize. If something is
    # still under augmentations/ here, treat it as a path bug.
    if head == "augmentations":
        head = "augmentation"
    if head == "common":
        # common files don't expose public methods directly
        return (None, "common", "", "")

    if len(rest) == 3:
        # <category>/<sub>/<file>
        sub = rest[1]
        return (
            f"{head}.{sub}.{stem}",
            head,
            sub,
            f"{head}.{sub}",
        )
    if len(rest) == 2:
        # <category>/<file>
        return (f"{head}.{stem}", head, "", head)
    # nested too deep — skip
    return (None, "", "", "")


# pls4all flat source mapping (built by hand from the canonical method list)
# value = (category, subcategory, method_id_override_or_None)
PLS4ALL_MAP: dict[str, tuple[str, str, str | None]] = {
    # AOM family
    "aom_operators":      ("aom_pop", "aom",     "aom_pop.aom_operators"),
    "aom_preprocessing":  ("aom_pop", "aom",     "aom_pop.aom_preprocessing"),
    "aom_selection":      ("aom_pop", "aom",     "aom_pop.aom_pls"),
    "operator_bank":      ("aom_pop", "aom",     "aom_pop.operator_bank"),
    "gating_strategy":    ("aom_pop", "aom",     "aom_pop.gating_strategy"),
    # Variable selection (24 files → selection.<name>)
    "bipls_selection":          ("selection", "wrapper",    "selection.bipls"),
    "bve_selection":            ("selection", "wrapper",    "selection.bve"),
    "cars_selection":           ("selection", "wrapper",    "selection.cars"),
    "emcuve_selection":         ("selection", "wrapper",    "selection.emcuve"),
    "ga_selection":             ("selection", "wrapper",    "selection.ga"),
    "interval_selection":       ("selection", "wrapper",    "selection.interval"),
    "ipw_selection":            ("selection", "wrapper",    "selection.ipw"),
    "irf_selection":            ("selection", "wrapper",    "selection.irf"),
    "iriv_selection":           ("selection", "wrapper",    "selection.iriv"),
    "pso_selection":            ("selection", "wrapper",    "selection.pso"),
    "random_frog_selection":    ("selection", "wrapper",    "selection.random_frog"),
    "randomization_selection":  ("selection", "wrapper",    "selection.randomization"),
    "rep_selection":            ("selection", "wrapper",    "selection.rep"),
    "scars_selection":          ("selection", "wrapper",    "selection.scars"),
    "shaving_selection":        ("selection", "wrapper",    "selection.shaving"),
    "sipls_selection":          ("selection", "wrapper",    "selection.sipls"),
    "spa_selection":            ("selection", "wrapper",    "selection.spa"),
    "st_selection":             ("selection", "wrapper",    "selection.st"),
    "stability_selection":      ("selection", "wrapper",    "selection.stability"),
    "t2_selection":             ("selection", "wrapper",    "selection.t2"),
    "uve_selection":            ("selection", "wrapper",    "selection.uve"),
    "vip_spa_selection":        ("selection", "wrapper",    "selection.vip_spa"),
    "vissa_selection":          ("selection", "wrapper",    "selection.vissa"),
    "wvc_selection":            ("selection", "wrapper",    "selection.wvc"),
    "variable_selection":       ("selection", "ranker",     "selection.variable_select"),
    # Diagnostics
    "pls_diagnostics":          ("diagnostics", "pls",     "diagnostics.pls_diagnostics"),
    "pls_monitoring":           ("diagnostics", "pls",     "diagnostics.pls_monitoring"),
    "component_coefficients":   ("diagnostics", "pls",     "diagnostics.component_coefficients"),
    "variable_importance":      ("diagnostics", "pls",     "diagnostics.variable_importance"),
    "model_selection":          ("diagnostics", "validation", "diagnostics.model_selection"),
    "validation":               ("diagnostics", "validation", "diagnostics.validation"),
    "cross_validation":         ("diagnostics", "validation", "diagnostics.cross_validation"),
    "metrics":                  ("diagnostics", "validation", "diagnostics.regression_metrics"),
    "classification_metrics":   ("diagnostics", "validation", "diagnostics.classification_metrics"),
    # Models — multiblock / local / classification / specialized
    "mb_pls":                   ("models", "multiblock",  "models.multiblock.mb_pls"),
    "multiblock_extensions":    ("models", "multiblock",  "models.multiblock.extensions"),
    "lw_pls":                   ("models", "local",       "models.local.lw_pls"),
    "pls_lda":                  ("models", "classification", "models.classification.pls_lda"),
    "pls_logistic":             ("models", "classification", "models.classification.pls_logistic"),
    "gpr_pls":                  ("models", "specialized", "models.specialized.gpr_pls"),
    "kernel_pls":               ("models", "pls",         "models.pls.kernel"),
    "tensor_pls":               ("models", "specialized", "models.specialized.tensor_pls"),
    "recursive_pls":            ("models", "specialized", "models.specialized.recursive"),
    "ecr":                      ("models", "specialized", "models.specialized.ecr"),
    # Model.cpp + extra_pls.cpp are the PLS aggregate — split in M4
    "model":                    ("models", "pls",         "models.pls._aggregate_v097"),
    "extra_pls":                ("models", "pls",         "models.pls._extra_aggregate_v097"),
    # Pipeline (split in M4)
    "pipeline":                 ("pipeline", "",          "pipeline._aggregate_v097"),
    # Common (no method_id — these are infra files, not public methods)
    "context":                  ("common", "",            None),  # skipped — not a public method
    "status":                   ("common", "",            None),
    "version":                  ("common", "",            None),
    "matrix_view":              ("common", "",            None),
    "config":                   ("common", "",            None),
    "cuda_dispatch":            ("common", "",            None),
}


CATEGORY_TO_ABI_PREFIX = {
    "preprocessing":  "n4m_preprocessing",
    "augmentation":   "n4m_augmentation",
    "splitters":      "n4m_splitter",
    "filters":        "n4m_filter",
    "utilities":      "n4m_utility",
    "models":         "n4m_models",
    "selection":      "n4m_selection",
    "diagnostics":    "n4m_diag",
    "aom_pop":        "n4m_aom_pop",
    "transfer":       "n4m_transfer",
    "pipeline":       "n4m_pipeline",
}


def guess_abi_symbols(method_id: str, category: str, subcategory: str) -> list[str]:
    """Produce plausible n4m_* symbol names for the catalog scaffolding.

    The final symbols land via M5/M6 ABI rename. These scaffolded names follow
    the convention n4m_<category>_<verb> with a per-category prefix.
    """
    prefix = CATEGORY_TO_ABI_PREFIX.get(category, f"n4m_{category}")
    # operator = last dotted segment, sanitized
    operator = method_id.split(".")[-1]
    operator = re.sub(r"[^a-z0-9_]", "_", operator)
    # category default verb sets
    if category == "preprocessing":
        return [
            f"{prefix}_{operator}_fit",
            f"{prefix}_{operator}_transform",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "augmentation":
        return [
            f"{prefix}_{operator}_create",
            f"{prefix}_{operator}_apply",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "splitters":
        return [
            f"{prefix}_{operator}_create",
            f"{prefix}_{operator}_split",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "filters":
        return [
            f"{prefix}_{operator}_fit",
            f"{prefix}_{operator}_apply",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "utilities":
        return [
            f"{prefix}_{operator}",
        ]
    if category == "models":
        # subcategory baked into the prefix
        sub_pref = f"n4m_{category}_{subcategory}" if subcategory else f"n4m_{category}"
        return [
            f"{sub_pref}_{operator}_fit",
            f"{sub_pref}_{operator}_predict",
            f"{sub_pref}_{operator}_destroy",
        ]
    if category == "selection":
        return [
            f"{prefix}_{operator}_run",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "diagnostics":
        return [
            f"{prefix}_{operator}",
        ]
    if category == "aom_pop":
        return [
            f"{prefix}_{operator}_fit",
            f"{prefix}_{operator}_predict",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "transfer":
        return [
            f"{prefix}_{operator}_calibrate",
            f"{prefix}_{operator}_apply",
            f"{prefix}_{operator}_destroy",
        ]
    if category == "pipeline":
        return [
            f"{prefix}_{operator}_create",
            f"{prefix}_{operator}_apply",
            f"{prefix}_{operator}_destroy",
        ]
    return [f"{prefix}_{operator}"]


def header_for_category(category: str) -> str:
    if category == "pipeline":
        return "cpp/include/n4m/n4m.h"
    return f"cpp/include/n4m/{category}.h"


def emit_entry(method_id: str, category: str, subcategory: str, family: str, tu_path: str) -> dict:
    return {
        "method_id":   method_id,
        "family":      family,
        "category":    category,
        "subcategory": subcategory or "general",
        "since_abi":   "1.0.0",
        "abi_symbols": guess_abi_symbols(method_id, category, subcategory),
        "tu":          [tu_path],
        "headers":     [header_for_category(category)],
        "carry_along_tu": [
            "cpp/src/core/common/context.cpp",
            "cpp/src/core/common/status.cpp",
            "cpp/src/core/common/matrix_view.cpp",
            "cpp/src/core/common/version.cpp",
        ],
        "parity": {
            "fixtures":      [],
            "tolerance_row": "tbd",
            "divergence_id": None,
            "references":    [],
        },
        "bench": {
            "bench_id":       method_id,
            "registry_entry": None,
        },
        "feature_flags":     [],
        "vendored_licenses": [],
        "bindings":          {},
        "publications":      [],
        "notes": "Auto-discovered from source tree in M2. Refine in subsequent phases (fixtures, references, bindings, publications).",
    }


def walk_pls4all() -> list[dict]:
    """Flat pls4all sources at cpp/src/core/*.cpp (M4 will split these)."""
    entries: list[dict] = []
    for cpp in sorted((REPO / "cpp" / "src" / "core").glob("*.cpp")):
        rel = cpp.relative_to(REPO).as_posix()
        mid, cat, sub, fam = file_to_method_id(rel)
        if mid is None:
            continue
        entries.append(emit_entry(mid, cat, sub, fam, rel))
    return entries


def walk_donor() -> list[dict]:
    """Donor sources lifted into cpp/src/core/<category>/... by M3."""
    entries: list[dict] = []
    core_root = REPO / "cpp" / "src" / "core"
    for sub_root_name in ("preprocessing", "augmentation", "splitters", "filters", "utilities"):
        sub_root = core_root / sub_root_name
        if not sub_root.is_dir():
            continue
        for c in sorted(sub_root.rglob("*.c")):
            rel = c.relative_to(REPO).as_posix()
            mid, cat, sub, fam = file_to_method_id(rel)
            if mid is None:
                continue
            entries.append(emit_entry(mid, cat, sub, fam, rel))
    return entries


def emit_yaml(entries: list[dict]) -> str:
    """Stdlib YAML emitter — avoid the PyYAML dep."""
    # We emit a list-of-mappings with stable key order.
    lines: list[str] = []
    lines.append("# Auto-discovered catalog of n4m methods.")
    lines.append("# Generated by catalog/scripts/discover_methods.py.")
    lines.append("# Do not edit by hand: refinements should land via a future merge_discovered.py.")
    lines.append("# Schema: catalog/schema/method_v1.json")
    lines.append("")
    lines.append("schema_version: 1")
    lines.append(f"method_count: {len(entries)}")
    lines.append("methods:")

    def emit_value(v, indent: int) -> list[str]:
        pad = " " * indent
        if isinstance(v, str):
            # Use plain scalar if safe, else quote
            if v and not any(c in v for c in [":", "#", "\n", '"', "'"]) and not v.strip() != v:
                return [v]
            return ['"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"']
        if v is None:
            return ["null"]
        if isinstance(v, bool):
            return ["true" if v else "false"]
        if isinstance(v, (int, float)):
            return [str(v)]
        if isinstance(v, list):
            if not v:
                return ["[]"]
            out = [""]
            for item in v:
                rendered = emit_value(item, indent + 2)
                if len(rendered) == 1 and not rendered[0].startswith(("{", "[")):
                    out.append(f"{pad}- {rendered[0]}")
                else:
                    out.append(f"{pad}-")
                    for line in rendered:
                        out.append(f"{pad}  {line}" if line else "")
            return out
        if isinstance(v, dict):
            if not v:
                return ["{}"]
            out = [""]
            for k, val in v.items():
                rendered = emit_value(val, indent + 2)
                if len(rendered) == 1 and not rendered[0].startswith(("\n",)):
                    out.append(f"{pad}{k}: {rendered[0]}")
                else:
                    out.append(f"{pad}{k}:")
                    for line in rendered:
                        if line:
                            out.append(f"{pad}  {line}" if not line.startswith(pad) else line)
                        else:
                            out.append("")
            return out
        return [str(v)]

    for entry in entries:
        lines.append("  - method_id:   " + entry["method_id"])
        lines.append("    family:      " + entry["family"])
        lines.append("    category:    " + entry["category"])
        lines.append("    subcategory: " + entry["subcategory"])
        lines.append("    since_abi:   \"" + entry["since_abi"] + "\"")
        lines.append("    abi_symbols:")
        for s in entry["abi_symbols"]:
            lines.append(f"      - {s}")
        lines.append("    tu:")
        for t in entry["tu"]:
            lines.append(f"      - {t}")
        lines.append("    headers:")
        for h in entry["headers"]:
            lines.append(f"      - {h}")
        lines.append("    carry_along_tu:")
        for c in entry["carry_along_tu"]:
            lines.append(f"      - {c}")
        lines.append("    parity:")
        lines.append("      fixtures: []")
        lines.append("      tolerance_row: tbd")
        lines.append("      divergence_id: null")
        lines.append("      references: []")
        lines.append("    bench:")
        lines.append(f"      bench_id: {entry['bench']['bench_id']}")
        lines.append("      registry_entry: null")
        lines.append("    feature_flags: []")
        lines.append("    vendored_licenses: []")
        lines.append("    bindings: {}")
        lines.append("    publications: []")
        notes_text = entry["notes"].replace('"', '\\"')
        lines.append(f"    notes: \"{notes_text}\"")
    return "\n".join(lines) + "\n"


def main() -> int:
    pls = walk_pls4all()
    don = walk_donor()
    all_entries = pls + don
    out = emit_yaml(all_entries)
    target = REPO / "catalog" / "methods.discovered.yaml"
    target.write_text(out)
    print(f"Wrote {target.relative_to(REPO)} — {len(all_entries)} methods")
    # Promote to methods.yaml on initial generation if absent:
    methods_yaml = REPO / "catalog" / "methods.yaml"
    if not methods_yaml.exists():
        methods_yaml.write_text(out)
        print(f"Initial methods.yaml created (same content) — {len(all_entries)} methods")
    else:
        print("methods.yaml already exists; merge with discovered file via human review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
