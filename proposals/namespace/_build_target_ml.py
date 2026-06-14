#!/usr/bin/env python3
"""Regenerate the FINAL Scheme-B (ML/DL) namespace mapping after the Codex review.

Starts from the reviewed base mapping (`_mappings.json['ml']`, id -> namespace) and the
ground-truth inventory (id -> leaf/method, abi_symbol), then applies the Codex change-list as
explicit overrides. Validates 208/208 coverage and zero fully-qualified-name collisions, and emits
`_target_ml_table.tsv` plus a printed tree summary. Single source of truth for the migration.
"""
import json
import collections
import pathlib

HERE = pathlib.Path(__file__).parent
base = json.load(open(HERE / "_mappings.json"))["ml"]          # id -> namespace (no n4m., no leaf)
inv = {r["id"]: r for r in json.load(open(HERE / "_method_inventory.json"))}

assert len(base) == 208 and len(inv) == 208, (len(base), len(inv))

# --- Codex change-list: namespace overrides (id -> new namespace) ----------------------------
NS = {
    # 1. split the 14-member estimators.regression.linear catch-all
    "models.pls.cppls": "estimators.regression.latent",
    "models.pls.pcr": "estimators.regression.latent",
    "models.pls.pls_fit_simple": "estimators.regression.latent",
    "models.regularized.continuum_regression": "estimators.regression.latent",
    "models.specialized.ecr": "estimators.regression.latent",
    "models.specialized.missing_aware_nipals": "estimators.regression.latent",
    "models.regularized.ridge": "estimators.regression.regularized",
    "models.regularized.ridge_pls": "estimators.regression.regularized",
    "models.regularized.robust_pls": "estimators.regression.robust",
    "models.regularized.weighted_pls": "estimators.regression.robust",
    "models.pls.kernel": "estimators.regression.kernel",
    "models.specialized.gpr_pls": "estimators.regression.kernel",
    "models.specialized.tensor_pls": "estimators.regression.tensor",
    "models.specialized.recursive": "estimators.regression.online",
    # fold estimators.local into regression.local
    "models.local.lw_pls": "estimators.regression.local",
    # 2. glm / survival split
    "models.heads.pls_glm": "estimators.regression.glm",
    "models.heads.pls_cox": "estimators.survival",
    # 3+4. domain_adaptation restructure (+ EPO moves in; OSC stays in transform)
    "models.transfer.di_pls": "domain_adaptation.invariant",
    "models.transfer.ds": "domain_adaptation.standardization",
    "models.transfer.pds": "domain_adaptation.standardization",
    "preprocessing.transfer.direct_standardization": "domain_adaptation.standardization",
    "preprocessing.transfer.piecewise_direct_standardization": "domain_adaptation.standardization",
    "preprocessing.transfer.robust_direct_standardization": "domain_adaptation.standardization",
    "preprocessing.transfer.slope_bias": "domain_adaptation.standardization",
    "utilities.transfer_metrics": "domain_adaptation.metrics",
    "preprocessing.orthogonalization.epo": "domain_adaptation.orthogonalization",
    # 5. redistribute the 18 aom_pop.* (search/campaign -> model_selection; superblock -> compose; blend/stack -> ensemble)
    "aom_pop.aom_sweep": "model_selection.aom_search",
    "aom_pop.aom_chain_sweep": "model_selection.aom_search",
    "aom_pop.aom_chain_fixed_fit": "model_selection.aom_search",
    "aom_pop.aom_chain_ridge_pls": "model_selection.aom_search",
    "aom_pop.aom_preprocessing": "model_selection.aom_search",
    "aom_pop.aom_pls": "model_selection.aom_search",
    "aom_pop.pop_pls": "model_selection.aom_search",
    "aom_pop.robust_hpo": "model_selection.aom_search",
    "aom_pop.ridge_global": "model_selection.aom_search",
    "aom_pop.aom_chain_screen_refit": "model_selection.aom_campaign",
    "aom_pop.aom_staged_chain_campaign": "model_selection.aom_campaign",
    "aom_pop.aom_pls_superblock": "compose.aom_superblock",
    "aom_pop.aom_ridge_pls_superblock": "compose.aom_superblock",
    "aom_pop.ridge_superblock": "compose.aom_superblock",
    "aom_pop.ridge_active_superblock": "compose.aom_superblock",
    "aom_pop.ridge_mkl_superblock": "compose.aom_superblock",
    "aom_pop.ridge_blender": "ensemble",
    "aom_pop.operator_pls_stack": "ensemble",
    # 6. model_selection.split -> model_selection.splitters
    "splitters.binned_strat_group_kfold": "model_selection.splitters",
    "splitters.kbins_stratified": "model_selection.splitters",
    "splitters.kennard_stone": "model_selection.splitters",
    "splitters.kmeans": "model_selection.splitters",
    "splitters.split_splitter": "model_selection.splitters",
    "splitters.spxy": "model_selection.splitters",
    "splitters.spxy_fold": "model_selection.splitters",
    "splitters.spxy_g_fold": "model_selection.splitters",
    "splitters.systematic_circular": "model_selection.splitters",
    # 9. dissolve the utils grab-bag ENTIRELY (Codex roadmap gate): no top-level utils
    "utilities.sweep": "model_selection.sweep",
    "utilities.moments": "lowlevel.moments",
    "utilities.signal_type_detector": "transform.signal_conversion",
}

# --- Codex change-list: leaf renames (id -> new leaf), the clean-break moment ------------------
LEAF = {
    "models.pls.pls_fit_simple": "pls",
    "models.pls.kernel": "kernel_pls",
    "models.specialized.tensor_pls": "n_pls",
    "models.specialized.recursive": "recursive_pls",
    "preprocessing.scaling.baseline": "baseline_center",
    "preprocessing.derivatives.derivate": "derivative",
    "splitters.split_splitter": "data_twinning",
    "diagnostics.model_selection": "one_se_rule",
    # flatten ensemble.aom (Codex roadmap gate) — provenance kept in the leaf name
    "aom_pop.ridge_blender": "aom_ridge_blender",
    "aom_pop.operator_pls_stack": "aom_operator_pls_stack",
}

# Explicit C-symbol disambiguation where the inventory single-column is ambiguous (Codex).
ABI = {"selection.wvc": "n4m_wvc_select"}

# Phase-R catalog SPLITS that create genuinely new public rows (Codex change 1): WVC carries two
# real public surfaces. (id, namespace, leaf, abi_symbol, legacy_of)
SYNTH = [
    ("selection.wvc_threshold", "feature_selection.wrapper", "wvc_threshold",
     "n4m_wvc_threshold_select", "selection.wvc"),
]

rows = []
fq_seen = {}
for mid in base:
    ns = NS.get(mid, base[mid])
    leaf = LEAF.get(mid, inv[mid]["method"])
    fq = f"n4m.{ns}.{leaf}"
    rows.append((mid, ns, leaf, fq, ABI.get(mid, inv[mid].get("abi_symbol") or "")))
    fq_seen.setdefault(fq, []).append(mid)
for sid, ns, leaf, abi, _legacy in SYNTH:
    fq = f"n4m.{ns}.{leaf}"
    rows.append((sid, ns, leaf, fq, abi))
    fq_seen.setdefault(fq, []).append(sid)

# --- validation -------------------------------------------------------------------------------
collisions = {fq: ids for fq, ids in fq_seen.items() if len(ids) > 1}
covered = {r[0] for r in rows}
missing = set(inv) - covered
assert not missing, f"uncovered ids: {missing}"
assert len(rows) == 208 + len(SYNTH), len(rows)
print("rows:", len(rows), "| unique fq:", len(fq_seen), "| collisions:", len(collisions))
if collisions:
    for fq, ids in collisions.items():
        print("  COLLISION", fq, ids)

# --- tree summary -----------------------------------------------------------------------------
top = collections.Counter(r[1].split(".")[0] for r in rows)
leafns = collections.Counter(r[1] for r in rows)
print("\nTOP-LEVEL (", len(top), "):")
for k, v in sorted(top.items(), key=lambda x: -x[1]):
    print(f"  {k:22s} {v}")
print("\nLEAF NAMESPACES (", len(leafns), "):")
for k, v in sorted(leafns.items()):
    print(f"  {k:42s} {v}")

# --- emit -------------------------------------------------------------------------------------
out = HERE / "_target_ml_table.tsv"
with open(out, "w") as f:
    f.write("catalog_id\tnamespace\tleaf\tfq_name\tabi_symbol\n")
    for r in sorted(rows, key=lambda r: (r[1], r[2])):
        f.write("\t".join(r) + "\n")
print("\nwrote", out)
