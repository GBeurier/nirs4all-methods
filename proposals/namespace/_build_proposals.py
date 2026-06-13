#!/usr/bin/env python3
"""Build the two namespace-mapping tables for NAMESPACE_PROPOSALS.md.

Reads _method_inventory.json (ground truth, 208 records) and emits, for each of
the two schemes, a complete id -> fully-qualified-namespace mapping. Validates
that every id is mapped exactly once and that both schemes total 208.

Run:  python3 _build_proposals.py            # prints validation + markdown tables
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INV = json.loads((HERE / "_method_inventory.json").read_text())
IDS = [r["id"] for r in INV]
BY_ID = {r["id"]: r for r in INV}
assert len(IDS) == len(set(IDS)) == 208, "inventory not 208 unique ids"

# ---------------------------------------------------------------------------
# Display-name helper (humanized short name used in tables)
# ---------------------------------------------------------------------------
def short(id_: str) -> str:
    return id_.rsplit(".", 1)[-1]

# ===========================================================================
# SCHEME 1 — STATS / CHEMOMETRICS
# Top level mirrors the chemometrics literature & R packages (pls, mdatools,
# ChemoSpec, prospectr): the major *analysis families* are the top tier
# (regression, discriminant, multiblock, preprocessing, variable_selection,
# transfer, validation, simulation, optimization). Keyed on method id.
# Namespaces below use that flat-family top level (no single umbrella root).
# ===========================================================================
STATS: dict[str, str] = {}

def S(ns: str, *ids: str) -> None:
    for i in ids:
        assert i in BY_ID, f"unknown id {i}"
        assert i not in STATS, f"double-mapped (stats): {i}"
        STATS[i] = ns

# --- chemometrics.regression.latent (latent-variable regression) ---
S("regression.latent",
  "models.pls.pls_fit_simple", "models.pls.pcr", "models.pls.kernel",
  "models.pls.cppls", "models.regularized.ridge_pls",
  "models.regularized.continuum_regression", "models.specialized.gpr_pls",
  "models.specialized.tensor_pls", "models.local.lw_pls",
  "models.specialized.recursive", "models.specialized.missing_aware_nipals",
  "models.specialized.ecr")

# --- chemometrics.regression.penalized (penalized / shrinkage) ---
S("regression.penalized",
  "models.regularized.ridge", "models.regularized.weighted_pls",
  "models.regularized.robust_pls")

# --- chemometrics.regression.sparse (sparse / structured-sparse LV regression) ---
S("regression.sparse",
  "models.sparse.sparse_simpls", "models.sparse.fused_sparse_pls",
  "models.sparse.group_sparse_pls")

# --- chemometrics.regression.ensemble (resampling ensembles of LV regressors) ---
S("regression.ensemble",
  "models.ensembles.bagging_pls", "models.ensembles.boosting_pls",
  "models.ensembles.random_subspace_pls", "models.ensembles.moment_stack")

# --- chemometrics.discriminant (discriminant analysis & classification) ---
S("discriminant",
  "models.classification.pls_lda", "models.classification.pls_qda",
  "models.classification.pls_logistic", "models.sparse.sparse_pls_da")

# --- chemometrics.glm_survival (generalized-response & survival heads) ---
S("regression.glm_survival",
  "models.heads.pls_glm", "models.heads.pls_cox")

# --- chemometrics.multiblock (multiblock & multiway) ---
S("multiblock",
  "models.multiblock.mb_pls", "models.multiblock.so_pls",
  "models.multiblock.on_pls", "models.multiblock.o2pls",
  "models.multiblock.rosa", "models.multiblock.mir_pls")

# --- chemometrics.calibration_transfer (instrument / standardization transfer) ---
S("transfer",
  "models.transfer.ds", "models.transfer.pds", "models.transfer.di_pls",
  "preprocessing.transfer.direct_standardization",
  "preprocessing.transfer.piecewise_direct_standardization",
  "preprocessing.transfer.robust_direct_standardization",
  "preprocessing.transfer.slope_bias",
  "preprocessing.orthogonalization.epo")
# transfer diagnostics live here too
S("transfer", "utilities.transfer_metrics")

# --- chemometrics.variable_selection (wavelength / variable selection) ---
SEL_WRAPPERS = [r["id"] for r in INV
                if r["category"] == "selection" and r["subcategory"] == "wrapper"
                and r["method"] != "interval"]
S("variable_selection", *SEL_WRAPPERS)
# variable_select (ranker) and interval (interval generator) round out the family.
S("variable_selection",
  "selection.variable_select", "selection.interval")

# --- chemometrics.preprocessing.* (signal preprocessing) ---
# scatter correction
S("preprocessing.scatter",
  "preprocessing.scatter.snv", "preprocessing.scatter.robust_snv",
  "preprocessing.scatter.local_snv", "preprocessing.scatter.piecewise_snv",
  "preprocessing.scatter.weighted_snv", "preprocessing.scatter.msc",
  "preprocessing.scatter.localized_msc", "preprocessing.scatter.piecewise_msc",
  "preprocessing.scatter.emsc", "preprocessing.scatter.vsn",
  "preprocessing.scatter.area_normalization",
  "preprocessing.scatter.local_centering")
# orthogonal signal correction (OSC is preprocessing-side; EPO already in transfer)
S("preprocessing.osc", "preprocessing.orthogonalization.osc")
# baseline
S("preprocessing.baseline",
  *[r["id"] for r in INV if r["family"] == "preprocessing.baselines"])
# smoothing & derivatives
S("preprocessing.smoothing_derivative",
  *[r["id"] for r in INV if r["family"] in ("preprocessing.derivatives",
                                            "preprocessing.smoothing")])
# wavelets
S("preprocessing.wavelet",
  *[r["id"] for r in INV if r["family"] == "preprocessing.wavelets"])
# scaling / normalization
S("preprocessing.scaling",
  *[r["id"] for r in INV if r["family"] == "preprocessing.scaling"])
# alignment / warping
S("preprocessing.alignment",
  *[r["id"] for r in INV if r["family"] == "preprocessing.alignment"])
# resampling / binning of the wavelength axis
S("preprocessing.resampling",
  *[r["id"] for r in INV if r["family"] == "preprocessing.resampling"])
# radiometric / unit conversion
S("preprocessing.signal_conversion",
  *[r["id"] for r in INV if r["family"] == "preprocessing.signal_conversion"])
# specialized fck
S("preprocessing.specialized", "preprocessing.specialized.fck_static")
# latent-projection feature extractors (PCA/SVD style preprocessing)
S("decomposition",
  "preprocessing.feature_selection.flexible_pca",
  "preprocessing.feature_selection.flexible_svd")

# --- chemometrics.outliers (sample / variable screening, leverage, Q/T2) ---
S("outliers",
  "filters.x_outlier", "filters.y_outlier", "filters.high_leverage",
  "filters.spectral_quality", "filters.composite",
  "filters.correlation", "filters.variance",
  "utilities.hotelling_t2", "utilities.q_residuals")

# --- chemometrics.resampling (sample selection & data splitting / CV design) ---
S("resampling",
  *[r["id"] for r in INV if r["category"] == "splitters"])

# --- chemometrics.diagnostics (model diagnostics, validation metrics) ---
S("validation",
  "diagnostics.pls_diagnostics", "diagnostics.pls_monitoring",
  "diagnostics.approximate_press", "diagnostics.model_selection",
  "diagnostics.regression_metrics",
  "utilities.signal_type_detector", "utilities.moments", "utilities.sweep")

# --- chemometrics.optimization.aom (AOM / POP operator optimization) ---
S("optimization.aom",
  *[r["id"] for r in INV if r["category"] == "aom_pop"])

# --- chemometrics.simulation (data perturbation / spectral simulation) ---
# All augmentation lives here, grouped by physical/statistical origin.
S("simulation.scatter",
  *[r["id"] for r in INV if r["family"] == "augmentation.scattering"])
S("simulation.baseline_drift",
  *[r["id"] for r in INV if r["family"] == "augmentation.drift"])
S("simulation.instrument",
  *[r["id"] for r in INV if r["family"] in ("augmentation.edge_artifacts",
                                            "augmentation.environmental")])
S("simulation.noise",
  *[r["id"] for r in INV if r["family"] == "augmentation.noise"])
S("simulation.wavelength",
  *[r["id"] for r in INV if r["family"] == "augmentation.wavelength"])
S("simulation.intensity",
  *[r["id"] for r in INV if r["family"] in ("augmentation.spectral",
                                            "augmentation.splines")])
S("simulation.mixing",
  *[r["id"] for r in INV if r["family"] in ("augmentation.mixup",
                                            "augmentation.random")])

# ===========================================================================
# SCHEME 2 — ML / DL PIPELINE (sklearn module layout)
# ===========================================================================
ML: dict[str, str] = {}

def M(ns: str, *ids: str) -> None:
    for i in ids:
        assert i in BY_ID, f"unknown id {i}"
        assert i not in ML, f"double-mapped (ml): {i}"
        ML[i] = ns

# --- transform.preprocessing.* (sklearn-style stateless/stateful transforms) ---
M("transform.scatter",
  *[r["id"] for r in INV if r["family"] == "preprocessing.scatter"])
M("transform.baseline",
  *[r["id"] for r in INV if r["family"] == "preprocessing.baselines"])
M("transform.smoothing",
  *[r["id"] for r in INV if r["family"] in ("preprocessing.derivatives",
                                            "preprocessing.smoothing")])
M("transform.wavelet",
  *[r["id"] for r in INV if r["family"] == "preprocessing.wavelets"])
M("transform.scaling",
  *[r["id"] for r in INV if r["family"] == "preprocessing.scaling"])
M("transform.alignment",
  *[r["id"] for r in INV if r["family"] == "preprocessing.alignment"])
M("transform.resampling",
  *[r["id"] for r in INV if r["family"] == "preprocessing.resampling"])
M("transform.signal_conversion",
  *[r["id"] for r in INV if r["family"] == "preprocessing.signal_conversion"])
M("transform.orthogonalization",
  "preprocessing.orthogonalization.osc", "preprocessing.orthogonalization.epo")
M("transform.specialized", "preprocessing.specialized.fck_static")

# --- decomposition (unsupervised projection transformers) ---
M("decomposition",
  "preprocessing.feature_selection.flexible_pca",
  "preprocessing.feature_selection.flexible_svd")

# --- feature_selection (variable selection: wrappers, filters, ranker) ---
M("feature_selection.wrapper",
  *[r["id"] for r in INV if r["category"] == "selection"
    and r["subcategory"] == "wrapper" and r["method"] != "interval"])
M("feature_selection.filter",
  "filters.correlation", "filters.variance")
M("feature_selection.ranking",
  "selection.variable_select")
M("feature_selection.interval",
  "selection.interval")

# --- augmentation.* (DL-style data augmentation appliers) ---
M("augmentation.noise",
  *[r["id"] for r in INV if r["family"] == "augmentation.noise"])
M("augmentation.scattering",
  *[r["id"] for r in INV if r["family"] == "augmentation.scattering"])
M("augmentation.drift",
  *[r["id"] for r in INV if r["family"] == "augmentation.drift"])
M("augmentation.instrument",
  *[r["id"] for r in INV if r["family"] in ("augmentation.edge_artifacts",
                                            "augmentation.environmental")])
M("augmentation.wavelength",
  *[r["id"] for r in INV if r["family"] == "augmentation.wavelength"])
M("augmentation.spectral",
  *[r["id"] for r in INV if r["family"] == "augmentation.spectral"])
M("augmentation.splines",
  *[r["id"] for r in INV if r["family"] == "augmentation.splines"])
M("augmentation.mixup",
  *[r["id"] for r in INV if r["family"] in ("augmentation.mixup",
                                            "augmentation.random")])

# --- estimators (regressors / classifiers, by sklearn estimator role) ---
M("estimators.regression.linear",
  "models.pls.pls_fit_simple", "models.pls.pcr", "models.pls.kernel",
  "models.pls.cppls", "models.regularized.ridge", "models.regularized.ridge_pls",
  "models.regularized.continuum_regression", "models.regularized.weighted_pls",
  "models.regularized.robust_pls", "models.specialized.gpr_pls",
  "models.specialized.tensor_pls", "models.specialized.missing_aware_nipals",
  "models.specialized.recursive", "models.specialized.ecr")
M("estimators.regression.sparse",
  "models.sparse.sparse_simpls", "models.sparse.fused_sparse_pls",
  "models.sparse.group_sparse_pls")
M("estimators.classification",
  "models.classification.pls_lda", "models.classification.pls_qda",
  "models.classification.pls_logistic", "models.sparse.sparse_pls_da")
M("estimators.glm_survival",
  "models.heads.pls_glm", "models.heads.pls_cox")
M("estimators.multiblock",
  "models.multiblock.mb_pls", "models.multiblock.so_pls",
  "models.multiblock.on_pls", "models.multiblock.o2pls",
  "models.multiblock.rosa", "models.multiblock.mir_pls")
M("estimators.local",
  "models.local.lw_pls")
M("ensemble",
  "models.ensembles.bagging_pls", "models.ensembles.boosting_pls",
  "models.ensembles.random_subspace_pls", "models.ensembles.moment_stack")

# --- model_selection (CV splitters / sample selection design) ---
M("model_selection.split",
  *[r["id"] for r in INV if r["category"] == "splitters"])

# --- compose / pipeline (AOM/POP staged campaigns + superblocks) ---
AOM_IDS = [r["id"] for r in INV if r["category"] == "aom_pop"]
# split AOM into operator search, superblocks, campaigns
COMPOSE_SEARCH = {"aom_pop.aom_pls", "aom_pop.pop_pls", "aom_pop.aom_sweep",
                  "aom_pop.aom_chain_sweep", "aom_pop.aom_chain_fixed_fit",
                  "aom_pop.aom_preprocessing", "aom_pop.robust_hpo",
                  "aom_pop.ridge_global"}
COMPOSE_SUPERBLOCK = {"aom_pop.aom_pls_superblock",
                      "aom_pop.aom_ridge_pls_superblock",
                      "aom_pop.ridge_superblock", "aom_pop.ridge_active_superblock",
                      "aom_pop.ridge_mkl_superblock", "aom_pop.ridge_blender",
                      "aom_pop.operator_pls_stack", "aom_pop.aom_chain_ridge_pls"}
COMPOSE_CAMPAIGN = {"aom_pop.aom_chain_screen_refit",
                    "aom_pop.aom_staged_chain_campaign"}
for i in AOM_IDS:
    if i in COMPOSE_SEARCH:
        M("compose.aom_search", i)
    elif i in COMPOSE_SUPERBLOCK:
        M("compose.aom_superblock", i)
    elif i in COMPOSE_CAMPAIGN:
        M("compose.aom_campaign", i)
    else:
        raise SystemExit(f"unrouted aom id {i}")

# --- domain_adaptation (calibration transfer = sklearn 'domain adaptation') ---
M("domain_adaptation",
  "models.transfer.ds", "models.transfer.pds", "models.transfer.di_pls",
  "preprocessing.transfer.direct_standardization",
  "preprocessing.transfer.piecewise_direct_standardization",
  "preprocessing.transfer.robust_direct_standardization",
  "preprocessing.transfer.slope_bias",
  "utilities.transfer_metrics")

# --- outlier_detection (sklearn 'outlier/novelty detection' + sample filters) ---
M("outlier_detection",
  "filters.x_outlier", "filters.y_outlier", "filters.high_leverage",
  "filters.spectral_quality", "filters.composite",
  "utilities.hotelling_t2", "utilities.q_residuals")

# --- metrics (scoring + diagnostics) ---
M("metrics.scoring",
  "diagnostics.regression_metrics", "diagnostics.approximate_press",
  "diagnostics.model_selection")
M("metrics.diagnostics",
  "diagnostics.pls_diagnostics", "diagnostics.pls_monitoring")

# --- utils (low-level helpers that are not estimators/transformers/metrics) ---
M("utils",
  "utilities.signal_type_detector", "utilities.moments", "utilities.sweep")

# ===========================================================================
# VALIDATE
# ===========================================================================
def validate(tag: str, mapping: dict[str, str]) -> None:
    missing = [i for i in IDS if i not in mapping]
    extra = [i for i in mapping if i not in BY_ID]
    print(f"[{tag}] mapped={len(mapping)} of {len(IDS)} "
          f"missing={len(missing)} extra={len(extra)}")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for m in missing:
            print("   -", m)
    if extra:
        print("  EXTRA:", extra)
    assert not missing and not extra and len(mapping) == 208, f"{tag} INVALID"

validate("STATS", STATS)
validate("ML", ML)

# top-level namespaces
def toplevels(mapping):
    return sorted({ns.split(".")[0] for ns in mapping.values()})
print("STATS top-levels:", toplevels(STATS))
print("ML top-levels:", toplevels(ML))

# node histograms (members per leaf namespace)
def leaf_counts(mapping):
    c = Counter(mapping.values())
    return dict(sorted(c.items()))
print("\nSTATS leaf counts:")
for k, v in leaf_counts(STATS).items():
    print(f"  {v:3d}  {k}")
print("\nML leaf counts:")
for k, v in leaf_counts(ML).items():
    print(f"  {v:3d}  {k}")

# Dump the two mappings to JSON for the doc-writer
out = {
    "stats": {i: STATS[i] for i in IDS},
    "ml": {i: ML[i] for i in IDS},
}
(HERE / "_mappings.json").write_text(json.dumps(out, indent=2))
print("\nwrote _mappings.json")
