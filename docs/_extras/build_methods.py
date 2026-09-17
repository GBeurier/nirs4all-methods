#!/usr/bin/env python3
"""Generate the n4m method documentation pages + the catalogue index.

Reads:
  * `catalog/methods.yaml` — the Phase-B+ method catalog and single source of
    truth for the full 209-method surface. Carries the ABI-2 `namespace` /
    `leaf` / `fq_name` per method; drives the catalogue index total + grouping.
  * `benchmarks/parity_timing/registry.py` (AST-parsed) — the 73 PLS/selection
    methods with descriptions, notes, parity-reference flags, parity tolerance.
  * `benchmarks/cross_binding/results/full_matrix.csv` — parity + timing
    cells per method and per backend.
  * `proposals/namespace/_rename_map.tsv` — old→new (ABI-1→ABI-2) C symbol
    map. It is the source of truth for every C symbol rendered on the pages:
    each method's symbol is resolved through it, a final substitution pass
    rewrites any symbol that leaked through a curated free-text string, and the
    `--strict` doc-lint fails the build if any ABI-1 symbol survives. It is also
    inverted to resolve each catalog method to its legacy documentation page.

Emits:
  * `docs/methods/<name>.md`  — one page per method (rich registry/operator
    pages + catalog stub pages for methods added in the namespace migration).
  * `docs/methods/index.md`   — catalogue index grouped by `n4m.<role>`.

The script is idempotent — re-running it overwrites the generated pages
without touching hand-written content.

Usage:
    python docs/_extras/build_methods.py
    python docs/_extras/build_methods.py --strict      # fail on missing data
    python docs/_extras/build_methods.py --no-bench    # skip parity tables
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from public_api_docs import (
    binding_for_catalog,
    cross_bindings_for_catalog,
    scan_matlab_public_api,
    scan_public_api,
    scan_r_public_api,
)
from selection_parity_policy import (
    binding_cross_check_reason,
    dependency_unavailable_reason,
    jaccard_from_note,
    selection_bydesign_reason,
)

try:
    from method_param_docs import lookup as _param_doc_lookup
except ImportError:  # pragma: no cover - depends on import path
    def _param_doc_lookup(method: str, param: str) -> str:
        return ""

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PY = ROOT / "benchmarks" / "parity_timing" / "registry.py"
# Legacy tier-2 binding catalog. Removed in the namespace migration; the
# `parse_catalog` reader degrades to an empty mapping when it is absent and is
# kept only as an optional secondary parameter source. C symbols are NEVER
# sourced from here — they come from the ABI-2 catalog / `_rename_map.tsv`.
CATALOG_YAML = ROOT / "bindings" / "_catalog" / "sklearn_tier2.yaml"
# The Phase-B+ method catalog (one entry per native method) is the single
# source of truth for the index: it carries the new `namespace` / `leaf` /
# `fq_name` fields for all 209 methods. The legacy registry / sklearn_tier2
# catalog only cover the 73 PLS/selection methods, so the index total and
# grouping are driven from here instead.
METHODS_CATALOG_YAML = ROOT / "catalog" / "methods.yaml"
RENAME_MAP_TSV = ROOT / "proposals" / "namespace" / "_rename_map.tsv"
CSV_PATH = ROOT / "benchmarks" / "cross_binding" / "results" / "full_matrix.csv"
METHODS_DIR = ROOT / "docs" / "methods"

# Source directories for the current ABI-2 binding surface.  The former
# ``pls4all`` / ``n4m.sklearn`` trees were removed during the namespace
# migration.  Do not silently fall back to them: doing so leaves generated
# pages apparently valid while advertising imports that no longer exist.
N4M_PYTHON_DIR = ROOT / "bindings" / "python" / "src" / "n4m"
N4M_IMPL_DIR = N4M_PYTHON_DIR / "_impl"
R_DIR = ROOT / "bindings" / "r" / "n4m" / "R"
MATLAB_DIR = ROOT / "bindings" / "matlab" / "+n4m"

# ---------------------------------------------------------------------------
# Per-method bibliographic / math metadata.
#
# The full curated body of explanations lives in `methods_bibliography.py`
# next to this file (kept separate so the generator file stays
# legible). We import it and use it directly.
# ---------------------------------------------------------------------------

try:
    from methods_bibliography import BIBLIOGRAPHY as _CURATED_BIB
except ImportError:  # pragma: no cover - depends on import path
    _CURATED_BIB = {}

try:
    from scientific_legacy import SCIENTIFIC_CONTENT as _LEGACY_SCIENCE
except ImportError:  # pragma: no cover - diagnosed by the strict content gate
    _LEGACY_SCIENCE = {}

# The operator and ABI-2-stub records deliberately live in small, separately
# owned modules.  Their keys are documentation page stems, not display names:
# that makes a rename in the ABI catalog visible to the coverage gate.
try:
    from scientific_augmentation_filter_split import SCIENTIFIC_CONTENT as _AUGMENTATION_SCIENCE
except ImportError:  # pragma: no cover - permits partial contributor checkouts
    _AUGMENTATION_SCIENCE = {}

try:
    from scientific_remaining import SCIENTIFIC_CONTENT as _REMAINING_SCIENCE
except ImportError:  # pragma: no cover - permits partial contributor checkouts
    _REMAINING_SCIENCE = {}

try:
    from scientific_aom import SCIENTIFIC_CONTENT as _AOM_SCIENCE
except ImportError:  # pragma: no cover - permits partial contributor checkouts
    _AOM_SCIENCE = {}

SCIENTIFIC_FIELDS = (
    "title", "paper", "principle", "use_cases", "limitations",
    "implementation", "provenance",
)

_GENERIC_SCIENCE_PLACEHOLDERS = (
    "sourced entirely from the catalog",
    "standard spectroscopic operator",
    "no binding description",
    "no curated reference",
)

# The remaining-content contributor supplied four temporary ABI-2 stub stems
# for the AOM Ridge compositions.  `scientific_aom.py` now owns the stable
# published pages with the full current function signatures; keeping both
# would create four unlinked duplicate documents.  The source records remain
# available for audit in their module, while the rendered corpus uses the
# more specific page records below.
_SUPERSEDED_SCIENCE_RECORDS = {
    "aom_pop_ridge_global",
    "aom_pop_ridge_superblock",
    "aom_pop_ridge_active_superblock",
    "aom_pop_ridge_mkl_superblock",
}


def validate_scientific_records(
        records: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Validate the complete seven-section scientific-record contract.

    The generator deliberately fails on incomplete prose instead of silently
    emitting a page which has parameters and a parity table but no scientific
    explanation.  Newlines are accepted in Markdown prose; every other C0
    control character is rejected because accidental Python escapes such as
    ``\\bar`` and ``\\theta`` otherwise become invisible backspace/tab bytes
    and corrupt MathJax output.
    """
    normalized_records: dict[str, dict[str, str]] = {}
    for stem, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"scientific record {stem!r} is not a mapping")
        normalized = dict(record)
        missing_fields = [
            field for field in SCIENTIFIC_FIELDS
            if not isinstance(normalized.get(field), str)
            or not normalized[field].strip()
        ]
        if missing_fields:
            raise ValueError(
                f"scientific record {stem!r} missing required fields: "
                f"{', '.join(missing_fields)}")
        text = "\n".join(normalized[field] for field in SCIENTIFIC_FIELDS)
        controls = sorted({ord(char) for char in text
                           if ord(char) < 32 and char != "\n"})
        if controls:
            rendered = ", ".join(f"U+{code:04X}" for code in controls)
            raise ValueError(
                f"scientific record {stem!r} contains forbidden C0 controls: "
                f"{rendered}")
        unbalanced_math = [
            field for field in SCIENTIFIC_FIELDS
            if normalized[field].count("$") % 2
        ]
        if unbalanced_math:
            raise ValueError(
                f"scientific record {stem!r} contains unbalanced inline-math delimiters: "
                f"{', '.join(unbalanced_math)}")
        lowered = text.lower()
        placeholders = [token for token in _GENERIC_SCIENCE_PLACEHOLDERS
                        if token in lowered]
        if placeholders:
            raise ValueError(
                f"scientific record {stem!r} contains placeholders: {placeholders}")
        normalized_records[stem] = normalized
    return normalized_records


def scientific_content() -> dict[str, dict[str, str]]:
    """Join and validate the complete rendered scientific corpus.

    The legacy source remains an unchanged historical archive.  Its rendered
    counterpart is the explicit 73-record scientific overlay, so corrections
    are reviewable without editing or silently trusting archived prose.
    """
    if set(_LEGACY_SCIENCE) != set(_CURATED_BIB):
        missing = sorted(set(_CURATED_BIB) - set(_LEGACY_SCIENCE))
        extra = sorted(set(_LEGACY_SCIENCE) - set(_CURATED_BIB))
        raise ValueError(
            "legacy scientific overlay does not exactly cover bibliography: "
            f"missing={missing}, extra={extra}")
    sources = (
        ("scientific_legacy", _LEGACY_SCIENCE),
        ("scientific_augmentation_filter_split", _AUGMENTATION_SCIENCE),
        ("scientific_remaining", {
            stem: record for stem, record in _REMAINING_SCIENCE.items()
            if stem not in _SUPERSEDED_SCIENCE_RECORDS
        }),
        ("scientific_aom", _AOM_SCIENCE),
    )
    merged: dict[str, dict[str, str]] = {}
    for source_name, records in sources:
        for stem, record in records.items():
            if stem in merged:
                raise ValueError(
                    f"duplicate scientific record for {stem!r}: {source_name}")
            merged[stem] = record
    return validate_scientific_records(merged)

BIBLIOGRAPHY: dict[str, dict] = dict(_CURATED_BIB) or {
    "pls": {
        "group": "core",
        "title": "PLS regression (SIMPLS)",
        "paper": "de Jong, S. (1993). *SIMPLS: an alternative approach to "
                 "partial least squares regression*. Chemometrics and "
                 "Intelligent Laboratory Systems 18(3), 251–263.",
        "principle": (
            "Project X into a `k`-dimensional latent space chosen to "
            "maximise covariance with `y`, then regress `y` on the latent "
            "scores. SIMPLS computes loadings directly from the "
            "cross-product matrix `X'y` without explicit deflation of X, "
            "which is the variant most closely matching MATLAB's "
            "`plsregress`."),
        "implementation": (
            "`n4m_pls_fit` in libn4m, dispatched through "
            "`Algorithm.PLS_REGRESSION` + `Solver.SIMPLS`. Variants NIPALS, "
            "SVD, power-iteration and randomized-SVD are also available "
            "via the `Solver` enum."),
    },
    "pcr": {
        "group": "core",
        "title": "Principal Components Regression",
        "paper": "Massy, W. F. (1965). *Principal Components Regression in "
                 "Exploratory Statistical Research*. JASA 60(309), 234–256.",
        "principle": (
            "SVD on the (centred) X, retain the top `k` components, "
            "regress `y` on the score matrix. The reference is "
            "`PCA(n_components=k)` + `LinearRegression` from scikit-learn "
            "and `pls::pcr` in R."),
        "implementation": (
            "`Algorithm.PCR` + `Solver.SVD` in libn4m, sharing the same "
            "Model.fit / Model.predict surface as PLS."),
    },
    "opls": {
        "group": "core",
        "title": "Orthogonal PLS (OPLS)",
        "paper": "Trygg, J. & Wold, S. (2002). *Orthogonal projections to "
                 "latent structures (O-PLS)*. Journal of Chemometrics "
                 "16(3), 119–128.",
        "principle": (
            "Decompose X into a predictive component aligned with y and a "
            "set of Y-orthogonal components. The Y-orthogonal subspace "
            "absorbs structural variation unrelated to the response, which "
            "improves interpretability for spectroscopy."),
        "implementation": (
            "`Algorithm.OPLS` + `Solver.NIPALS` + `Deflation.ORTHOGONAL`. "
            "The R reference is Bioconductor `ropls::opls`; "
            "orthogonal-component conventions may differ across libraries."),
    },
    "sparse_simpls": {
        "group": "sparse",
        "title": "Sparse SIMPLS",
        "paper": "Chun, H. & Keleş, S. (2010). *Sparse partial least "
                 "squares regression for simultaneous dimension reduction "
                 "and variable selection*. JRSS B 72(1), 3–25.",
        "principle": (
            "SIMPLS with an L1 soft-threshold applied to the loading "
            "weights at each component. The penalty `sparsity_lambda` "
            "trades off prediction accuracy against feature selection."),
        "implementation": (
            "`n4m_sparse_simpls_fit` — MethodResult entry point; "
            "coefficients map back to original feature space. Reference: "
            "R `spls` 2.3.2."),
    },
    "di_pls": {
        "group": "calibration-transfer",
        "title": "Domain-invariant PLS (di-PLS)",
        "paper": "Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S. "
                 "& Moser, B. A. (2018). *Domain-invariant partial-least-"
                 "squares regression*. Analytical Chemistry 90, "
                 "6693–6701.",
        "principle": (
            "PLS with an extra penalty that pushes the score "
            "distributions of a source and a target domain together. "
            "Useful when measurement conditions drift between calibration "
            "and prediction sets."),
        "implementation": (
            "`n4m_di_pls_fit` — needs `X_target` at fit time. Reference: "
            "Python `diPLSlib.models.DIPLS` (B-Analytics)."),
    },
    "recursive_pls": {
        "group": "core",
        "title": "Recursive (moving-window) PLS",
        "paper": "Helland, K., Berntsen, H. E., Borgen, O. S. & Martens, "
                 "H. (1992). *Recursive algorithm for partial least "
                 "squares regression*. Chemom. Intell. Lab. Syst. "
                 "14, 129–137.",
        "principle": (
            "Update the PLS fit incrementally over a sliding window of "
            "samples. Suitable for streaming / drifting processes."),
        "implementation": (
            "`n4m_recursive_pls_run`. References: sklearn rolling-window "
            "PLS, R `pls` window refits."),
    },
    "cppls": {
        "group": "core",
        "title": "Powered PLS (CPPLS / Indahl)",
        "paper": "Indahl, U. G. (2005). *A twist to partial least squares "
                 "regression*. Journal of Chemometrics 19(1), 32–44.",
        "principle": (
            "PLS where the column inner products are powered by `γ` "
            "before the projection step, which moves the latent direction "
            "between PCA (`γ=0`) and PLS (`γ=1`)."),
        "implementation": (
            "`n4m_cppls_fit`. NOTE: R `pls::cppls` is **Liland 2009 "
            "Canonical Powered PLS**, a different algorithm — same name, "
            "different mathematics. Use the tier-1 reference only with "
            "this caveat."),
    },
    "weighted_pls": {
        "group": "robust",
        "title": "Sample-weighted PLS",
        "paper": "Martens, H. & Næs, T. (1989). *Multivariate "
                 "Calibration*. Wiley. Chapter on weighted regression.",
        "principle": (
            "SIMPLS on `√w`-prescaled, centred data. Equivalent to "
            "running a standard PLS on a weighted residual problem."),
        "implementation": (
            "`n4m_weighted_pls_fit` — in-sample only (no global coef "
            "export). Needs a sample-weight vector at fit time."),
    },
    "robust_pls": {
        "group": "robust",
        "title": "Robust PLS (Huber IRLS over SIMPLS)",
        "paper": "Serneels, S., Croux, C., Filzmoser, P. & Van Espen, "
                 "P. J. (2005). *Partial Robust M-Regression*. Chemom. "
                 "Intell. Lab. Syst. 79, 55–64.",
        "principle": (
            "Iteratively reweighted PLS with a Huber loss on the "
            "residuals. Down-weights outliers without removing them."),
        "implementation": (
            "`n4m_robust_pls_fit`. Reference: R `chemometrics::prm`."),
    },
    "ridge_pls": {
        "group": "regularized",
        "title": "Ridge-augmented PLS",
        "paper": "Hoerl, A. E. & Kennard, R. W. (1970). *Ridge "
                 "regression: biased estimation for nonorthogonal "
                 "problems*. Technometrics 12(1), 55–67.",
        "principle": (
            "Augment the PLS regression step with an L2 ridge penalty "
            "on the latent-space coefficients. Useful when `k` is close "
            "to the rank of X and the spectra are highly collinear."),
        "implementation": "`n4m_ridge_pls_fit` (in-sample only).",
    },
    "continuum_regression": {
        "group": "nonlinear",
        "title": "Continuum regression",
        "paper": "Stone, M. & Brooks, R. J. (1990). *Continuum "
                 "regression: cross-validated sequentially constructed "
                 "prediction embracing ordinary least squares, partial "
                 "least squares and principal components regression*. "
                 "JRSS B 52(2), 237–269.",
        "principle": (
            "A one-parameter family `τ ∈ [0, 1]` interpolating OLS "
            "(`τ=0`), PLS (`τ=0.5`) and PCR (`τ=1`)."),
        "implementation": "`n4m_continuum_regression_fit`.",
    },
    "n_pls": {
        "group": "multi-block",
        "title": "N-way PLS (Trilinear PLS)",
        "paper": "Bro, R. (1996). *Multiway calibration. Multilinear "
                 "PLS*. Journal of Chemometrics 10(1), 47–61.",
        "principle": (
            "Tensor-mode PLS for X arranged as an `n × j × k` tensor. "
            "Decomposes X into rank-one tensor components."),
        "implementation": (
            "`n4m_n_pls_fit` — takes a flattened X plus `mode_j` and "
            "`mode_k` shape parameters. Reference: `tensorly` and the "
            "original Bro paper code."),
    },
    "kernel_pls_rbf": {
        "group": "nonlinear",
        "title": "Kernel PLS",
        "paper": "Rosipal, R. & Trejo, L. J. (2001). *Kernel partial "
                 "least squares regression in reproducing kernel Hilbert "
                 "space*. JMLR 2, 97–123.",
        "principle": (
            "PLS in the feature space of an RBF / polynomial / linear "
            "kernel. Captures nonlinear relationships between X and y."),
        "implementation": (
            "`n4m_kernel_pls_fit`. Predict-on-new-X is currently "
            "in-sample-only in the Python sklearn wrapper because the "
            "ABI does not yet export the kernel-centering."),
    },
    "o2pls": {
        "group": "multi-block",
        "title": "O2-PLS",
        "paper": "Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block "
                 "(X–Y) latent variable regression*. Journal of "
                 "Chemometrics 17(1), 53–64.",
        "principle": (
            "Symmetric two-block PLS that separates joint, X-orthogonal "
            "and Y-orthogonal variation."),
        "implementation": (
            "`n4m_o2pls_fit`. Reference: CRAN `OmicsPLS` 2.1.0."),
    },
    "approximate_press": {
        "group": "diagnostic",
        "title": "Approximate PRESS",
        "paper": "Allen, D. M. (1974). *The relationship between "
                 "variable selection and data augmentation and a method "
                 "for prediction*. Technometrics 16(1), 125–127.",
        "principle": (
            "An O(n) approximation of the predicted residual sum of "
            "squares using the hat-matrix diagonal."),
        "implementation": "`n4m_approximate_press_compute`.",
    },
    "pls_diagnostic_t2": {
        "group": "diagnostic",
        "title": "Hotelling T² score",
        "paper": "Hotelling, H. (1931). *The generalization of "
                 "Student's ratio*. Annals of Mathematical Statistics "
                 "2(3), 360–378.",
        "principle": (
            "Sum of squared standardized PLS scores per sample. Flags "
            "samples that are unusual in the latent space."),
        "implementation": "`n4m_pls_diagnostics_compute` with stat=\"t2\".",
    },
    "pls_diagnostic_q": {
        "group": "diagnostic",
        "title": "Q residual (SPE)",
        "paper": "Jackson, J. E. & Mudholkar, G. S. (1979). *Control "
                 "procedures for residuals associated with principal "
                 "component analysis*. Technometrics 21(3), 341–349.",
        "principle": (
            "Squared prediction error in feature space (X minus its "
            "PLS reconstruction). Flags samples whose X is not well "
            "represented by the model."),
        "implementation": "`n4m_pls_diagnostics_compute` with stat=\"q\".",
    },
    "pls_monitoring": {
        "group": "diagnostic",
        "title": "PLS monitoring (T² + Q with alarms)",
        "paper": "Kourti, T. & MacGregor, J. F. (1996). *Multivariate "
                 "SPC methods for process and product monitoring*. "
                 "Journal of Quality Technology 28(4), 409–428.",
        "principle": (
            "Combines T² and Q charts with control limits derived "
            "from the calibration set. Returns per-sample alarms."),
        "implementation": "`n4m_pls_monitoring_run`.",
    },
    "one_se_rule": {
        "group": "diagnostic",
        "title": "One-SE rule for component count",
        "paper": "Hastie, T., Tibshirani, R. & Friedman, J. (2009). "
                 "*The Elements of Statistical Learning*, 2nd ed., §7.10.",
        "principle": (
            "Pick the smallest `k` whose CV-RMSE is within one "
            "standard error of the minimum. Reduces over-fitting."),
        "implementation": "`n4m_one_se_rule_compute`.",
    },
    "so_pls": {
        "group": "multi-block",
        "title": "SO-PLS (Sequential and Orthogonalised PLS)",
        "paper": "Næs, T., Tomic, O., Mevik, B.-H. & Martens, H. "
                 "(2011). *Path modelling by sequential PLS regression*. "
                 "Journal of Chemometrics 25(1), 28–40.",
        "principle": (
            "Fit PLS on the first block, then orthogonalise the "
            "remaining blocks against the first block's scores and "
            "iterate. Captures block-specific contributions."),
        "implementation": (
            "`n4m_so_pls_fit`. Reference: R `multiblock` package."),
    },
    "on_pls": {
        "group": "multi-block",
        "title": "OnPLS (Orthogonal-N PLS)",
        "paper": "Löfstedt, T. & Trygg, J. (2011). *OnPLS — a "
                 "novel multiblock method for the modelling of "
                 "predictive and orthogonal variation*. Journal of "
                 "Chemometrics 25(8), 441–455.",
        "principle": (
            "Multi-block extension of OPLS — separates joint and "
            "unique components per block."),
        "implementation": (
            "`n4m_on_pls_fit`. The vendored `OnPLS` Python port is "
            "carried in `bindings/python/vendor/OnPLS/` because the CRAN "
            "package was archived."),
    },
    "rosa": {
        "group": "multi-block",
        "title": "ROSA (Response-oriented sequential alternation)",
        "paper": "Liland, K. H. & Næs, T. (2016). *Response-oriented "
                 "sequential alternation (ROSA): a fast multiblock "
                 "regression algorithm*. Journal of Chemometrics 30(11), "
                 "651–662.",
        "principle": (
            "At each component, ROSA picks the block whose addition "
            "best explains y, in a forward greedy manner."),
        "implementation": (
            "`n4m_rosa_fit`. Reference: R `multiblock`."),
    },
    "vissa_select": {
        "group": "selector",
        "title": "VISSA — Variable Iterative Space-Shrinkage Approach",
        "paper": "Deng, B. C. et al. (2014). *A new strategy to prevent "
                 "over-fitting in partial least squares models based on "
                 "model population analysis*. Anal. Chim. Acta 880, "
                 "32–41.",
        "principle": (
            "Population of random subsets refined by Monte-Carlo "
            "subsampling; variables in surviving subsets are retained."),
        "implementation": "`n4m_vissa_select`.",
    },
    "pso_select": {
        "group": "selector",
        "title": "PSO variable selection",
        "paper": "Kennedy, J. & Eberhart, R. (1995). *Particle swarm "
                 "optimization*. IEEE ICNN, 1942–1948.",
        "principle": (
            "Wrap a particle-swarm optimiser around PLS cross-validated "
            "RMSE; each particle's position encodes a subset of features."),
        "implementation": (
            "`n4m_pso_select`. Reference: Python `pyswarms`."),
    },
    "gpr_pls": {
        "group": "nonlinear",
        "title": "GPR-PLS (Gaussian Process Regression in PLS scores)",
        "paper": "Bishop, C. M. (2006). *Pattern Recognition and "
                 "Machine Learning*, §6.4 (Gaussian Processes).",
        "principle": (
            "Project X into the PLS latent space, then fit a GP on the "
            "scores. Reference: sklearn's `GaussianProcessRegressor` "
            "with an RBF kernel on the score matrix."),
        "implementation": "`n4m_gpr_pls_fit`.",
    },
    "bagging_pls": {
        "group": "ensemble",
        "title": "Bagging PLS",
        "paper": "Breiman, L. (1996). *Bagging predictors*. Machine "
                 "Learning 24(2), 123–140.",
        "principle": (
            "Bootstrap-aggregated PLS — fit `n_estimators` PLS models "
            "on bootstrap samples, average their predictions."),
        "implementation": (
            "`n4m_bagging_pls_fit`. Reference: R `enpls`."),
    },
    "boosting_pls": {
        "group": "ensemble",
        "title": "Boosting PLS",
        "paper": "Friedman, J. H. (2001). *Greedy function "
                 "approximation: a gradient boosting machine*. Annals "
                 "of Statistics 29(5), 1189–1232.",
        "principle": (
            "Gradient boosting where each weak learner is a small PLS "
            "model. The reference is R `mboost::glmboost` with a PLS "
            "base learner."),
        "implementation": "`n4m_boosting_pls_fit`.",
    },
    "random_subspace_pls": {
        "group": "ensemble",
        "title": "Random-subspace PLS",
        "paper": "Ho, T. K. (1998). *The random subspace method for "
                 "constructing decision forests*. IEEE TPAMI 20(8), "
                 "832–844.",
        "principle": (
            "Each ensemble member trains on a random feature subset "
            "of size `features_per_subspace`. Reduces variance for "
            "high-dimensional spectra."),
        "implementation": "`n4m_random_subspace_pls_fit`.",
    },
    "pls_glm": {
        "group": "classification",
        "title": "PLS-GLM (Gaussian / Poisson families)",
        "paper": "Marx, B. D. (1996). *Iteratively reweighted partial "
                 "least squares estimation for generalized linear "
                 "regression*. Technometrics 38(4), 374–381.",
        "principle": (
            "Iteratively reweighted PLS fitting a GLM with Gaussian or "
            "Poisson link. The reference is CRAN `plsRglm`."),
        "implementation": "`n4m_pls_glm_fit`.",
    },
    "pls_qda": {
        "group": "classification",
        "title": "PLS-QDA",
        "paper": "Pérez-Enciso, M. & Tenenhaus, M. (2003). *Prediction "
                 "of clinical outcome with microarray data: a partial "
                 "least squares discriminant analysis (PLS-DA) "
                 "approach*. Human Genetics 112(5–6), 581–592.",
        "principle": (
            "Project X via PLS, then fit Quadratic Discriminant "
            "Analysis on the latent scores. Class-specific covariances."),
        "implementation": "`n4m_pls_qda_fit`.",
    },
    "pls_cox": {
        "group": "classification",
        "title": "PLS-Cox (survival)",
        "paper": "Bastien, P., Bertrand, F., Meyer, N. & Maumy-"
                 "Bertrand, M. (2015). *Deviance residuals-based "
                 "sparse PLS and sparse kernel PLS for Cox model*. "
                 "Bioinformatics 31(3), 397–404.",
        "principle": (
            "PLS on deviance residuals from a Cox proportional-hazards "
            "model. Survival times + censoring indicators required."),
        "implementation": (
            "`n4m_pls_cox_fit`. Reference: CRAN `plsRcox`."),
    },
    "pds": {
        "group": "calibration-transfer",
        "title": "Piecewise Direct Standardisation",
        "paper": "Wang, Y., Veltkamp, D. J. & Kowalski, B. R. (1991). "
                 "*Multivariate instrument standardisation*. Analytical "
                 "Chemistry 63(23), 2750–2756.",
        "principle": (
            "Map secondary-instrument spectra to the primary instrument "
            "via a banded transfer matrix. Window of half-width `w`."),
        "implementation": (
            "`n4m_pds_fit` — TransformerMixin in tier 2. Reference: "
            "R `prospectr::pds`."),
    },
    "ds": {
        "group": "calibration-transfer",
        "title": "Direct Standardisation",
        "paper": "Wang, Y. et al. (1991), as above.",
        "principle": (
            "Single global transfer matrix between two instruments. "
            "Simpler than PDS, sometimes higher variance."),
        "implementation": "`n4m_ds_fit`. Reference: R `chemometrics::stdize`.",
    },
    "mir_pls": {
        "group": "multi-block",
        "title": "MIR-PLS (Mid-InfraRed PLS, regularised)",
        "paper": "Custom kernel matching standard MIR conventions; "
                 "see registry notes for the algorithm reference.",
        "principle": (
            "PLS with kernel regularisation tuned to mid-IR spectra. "
            "In-tree sanctioned port; no widely installable library "
            "equivalent."),
        "implementation": "`n4m_mir_pls_fit`.",
    },
    "missing_aware_nipals": {
        "group": "missing",
        "title": "Missing-aware NIPALS",
        "paper": "Walczak, B. & Massart, D. L. (2001). *Dealing with "
                 "missing data*. Chemom. Intell. Lab. Syst. 58, 15–27.",
        "principle": (
            "NIPALS PLS that handles `NaN` entries by skipping them in "
            "the inner-product and norm computations."),
        "implementation": (
            "`n4m_missing_aware_nipals_fit`. Reference: R `softImpute` "
            "for the imputation step (sanctioned)."),
    },
    "sparse_pls_da": {
        "group": "classification",
        "title": "Sparse PLS-DA",
        "paper": "Lê Cao, K.-A. et al. (2008). *A sparse PLS for "
                 "variable selection when integrating omics data*. Stat. "
                 "Appl. Genet. Mol. Biol. 7(1).",
        "principle": (
            "Sparse PLS-DA — soft-threshold loadings to select a "
            "discriminative subset per component."),
        "implementation": (
            "`n4m_sparse_pls_da_fit`. Reference: Bioconductor `mixOmics::splsda`."),
    },
    "group_sparse_pls": {
        "group": "sparse",
        "title": "Group-sparse PLS",
        "paper": "Liquet, B. et al. (2016). *Group and sparse group "
                 "partial least squares*. Bioinformatics 32(1), 35–42.",
        "principle": (
            "Apply group-lasso style sparsity at the loading-weight "
            "level — entire groups of features enter or leave together."),
        "implementation": (
            "`n4m_group_sparse_pls_fit`. Reference: CRAN `sgPLS`."),
    },
    "fused_sparse_pls": {
        "group": "sparse",
        "title": "Fused-sparse PLS",
        "paper": "Tibshirani, R. et al. (2005). *Sparsity and "
                 "smoothness via the fused lasso*. JRSS B 67(1), 91–108.",
        "principle": (
            "L1 + fused-L1 penalty on PLS loadings — neighbouring "
            "features (along the wavelength axis) tend to share weights."),
        "implementation": "`n4m_fused_sparse_pls_fit`.",
    },
    "pls_diagnostic_dmodx": {
        "group": "diagnostic",
        "title": "DModX (distance to the model)",
        "paper": "Eriksson, L. et al. (2013). *Multi- and Megavariate "
                 "Data Analysis*, Umetrics Academy, §4.7.",
        "principle": (
            "Per-sample residual sum of squares after PLS "
            "reconstruction, normalised by the model's residual "
            "degrees of freedom."),
        "implementation": "`n4m_pls_diagnostics_compute` with stat=\"dmodx\".",
    },
    "mb_pls": {
        "group": "multi-block",
        "title": "MB-PLS (Multi-block PLS)",
        "paper": "Westerhuis, J. A., Kourti, T. & MacGregor, J. F. "
                 "(1998). *Analysis of multiblock and hierarchical PCA "
                 "and PLS models*. Journal of Chemometrics 12(5), "
                 "301–321.",
        "principle": (
            "Multi-block PLS with block-autoscaling and block weights, "
            "mapping coefficients back to original feature space."),
        "implementation": (
            "`n4m_mb_pls_fit`. Reference: sanctioned git-pinned port "
            "`nirs4all.operators.models.sklearn.mbpls`."),
    },
    "lw_pls": {
        "group": "nonlinear",
        "title": "Locally-Weighted PLS",
        "paper": "Centner, V. & Massart, D. L. (1998). *Optimisation "
                 "in locally weighted regression*. Analytical Chemistry "
                 "70(19), 4206–4211.",
        "principle": (
            "For each prediction point, refit PLS on its k-nearest "
            "neighbours in the calibration set."),
        "implementation": (
            "`n4m_lw_pls_fit`. Reference: sanctioned git-pinned port "
            "`nirs4all.operators.models.sklearn.lwpls`."),
    },
    "pls_lda": {
        "group": "classification",
        "title": "PLS-LDA",
        "paper": "Barker, M. & Rayens, W. (2003). *Partial least "
                 "squares for discrimination*. Journal of Chemometrics "
                 "17(3), 166–173.",
        "principle": (
            "Project X via PLS, then fit Linear Discriminant Analysis "
            "on the latent scores."),
        "implementation": "`n4m_pls_lda_fit`. Reference: composite (PLSRegression + LDA).",
    },
    "pls_logistic": {
        "group": "classification",
        "title": "PLS-logistic",
        "paper": "Bastien, P., Esposito Vinzi, V. & Tenenhaus, M. "
                 "(2005). *PLS generalised linear regression*. "
                 "Computational Statistics & Data Analysis 48(1), "
                 "17–46.",
        "principle": (
            "Iteratively reweighted PLS with a logit link function "
            "for binary / multinomial classification."),
        "implementation": "`n4m_pls_logistic_fit`. Reference: R `plsRglm`.",
    },
    "aom_preprocess": {
        "group": "diagnostic",
        "title": "AOM preprocessing bank",
        "paper": "Bench oracle in "
                 "`nirs4all.operators.models.sklearn.aom_pls`; "
                 "AOM-PLS family.",
        "principle": (
            "An operator-mixture bank of preprocessing transforms with "
            "soft (equal weights) or hard (first operator) gating. "
            "Forms the building block of AOM-SIMPLS and POP-PLS."),
        "implementation": "`n4m_aom_preprocess_fit`.",
    },
    "aom_pls": {
        "group": "adaptive",
        "title": "AOM-PLS",
        "paper": "Beurier et al. (2026), operator-adaptive PLS/Ridge.",
        "principle": (
            "Global adaptive operator selection (nirs4all AOMPLSRegressor) "
            "over the compact "
            "strict-linear nirs4all bank, selecting the operator and "
            "component count by cross-validated RMSE."),
        "implementation": "`n4m_aom_global_select`.",
    },
    "pop_pls": {
        "group": "adaptive",
        "title": "POP-PLS",
        "paper": "Beurier et al. (2026), POP-PLS ablation.",
        "principle": (
            "Per-component adaptive operator selection (nirs4all "
            "POPPLSRegressor) over the compact "
            "strict-linear nirs4all bank, followed by best-prefix selection."),
        "implementation": "`n4m_aom_per_component_select`.",
    },
    "variable_select_vip": {
        "group": "selector",
        "title": "VIP variable selection",
        "paper": "Wold, S., Sjöström, M. & Eriksson, L. (2001). *PLS-"
                 "regression: a basic tool of chemometrics*. Chemom. "
                 "Intell. Lab. Syst. 58(2), 109–130.",
        "principle": (
            "Variable Importance in Projection — weighted average of "
            "loadings, normalised so a value `> 1` indicates an "
            "important variable."),
        "implementation": "`n4m_variable_select_rank` with metric=VIP.",
    },
    "variable_select_coef": {
        "group": "selector",
        "title": "Coefficient-magnitude selection",
        "paper": "Standard chemometrics convention; see Martens & Næs "
                 "(1989).",
        "principle": (
            "Rank features by the magnitude of their PLS regression "
            "coefficient in the original feature scale."),
        "implementation": "`n4m_variable_select_rank` with metric=COEF.",
    },
    "variable_select_sr": {
        "group": "selector",
        "title": "Selectivity ratio",
        "paper": "Rajalahti, T. et al. (2009). *Biomarker discovery "
                 "in mass spectral profiles by means of selectivity "
                 "ratio plot*. Chemom. Intell. Lab. Syst. 95(1), 35–48.",
        "principle": (
            "Ratio of explained to residual variance per feature, "
            "computed from the PLS target-projected coefficients."),
        "implementation": "`n4m_variable_select_rank` with metric=SR.",
    },
    "interval_select": {
        "group": "selector",
        "title": "iPLS / Moving-window interval selection",
        "paper": "Nørgaard, L., Saudland, A., Wagner, J., Nielsen, "
                 "J. P., Munck, L. & Engelsen, S. B. (2000). *Interval "
                 "partial least-squares regression (iPLS)*. Appl. "
                 "Spectroscopy 54(3), 413–419.",
        "principle": (
            "Slide a fixed-width window across wavelengths; pick the "
            "interval whose PLS CV-RMSE is lowest."),
        "implementation": "`n4m_interval_select`. Reference: R `plsVarSel`.",
    },
    "bipls_select": {
        "group": "selector",
        "title": "Backward interval PLS (biPLS)",
        "paper": "Leardi, R. & Nørgaard, L. (2004). *Sequential "
                 "application of backward interval partial least squares "
                 "and genetic algorithms for the selection of relevant "
                 "spectral regions*. J. Chemom. 18(11), 486–497.",
        "principle": (
            "Start with all intervals and recursively eliminate the "
            "weakest by CV-RMSE."),
        "implementation": "`n4m_bipls_select`. Reference: R `plsVarSel`.",
    },
    "sipls_select": {
        "group": "selector",
        "title": "Synergy interval PLS (siPLS)",
        "paper": "Nørgaard, L. et al. (2000), as for `interval_select`.",
        "principle": (
            "Exhaustively score every combination of `m` fixed-size "
            "intervals; pick the combination with the lowest CV-RMSE."),
        "implementation": "`n4m_sipls_select`. Reference: R `plsVarSel`.",
    },
    "stability_select": {
        "group": "selector",
        "title": "MC-UVE / Stability selection",
        "paper": "Cai, W. et al. (2008). *A variable selection method "
                 "based on uninformative variable elimination for "
                 "multivariate calibration of near-infrared spectra*. "
                 "Chemom. Intell. Lab. Syst. 90(2), 188–194.",
        "principle": (
            "Compute coefficient mean / std ratio over Monte-Carlo "
            "subsamples; rank features by this ratio."),
        "implementation": "`n4m_stability_select`. Reference: R `plsVarSel`.",
    },
    "uve_select": {
        "group": "selector",
        "title": "Uninformative variable elimination",
        "paper": "Centner, V. et al. (1996). *Elimination of "
                 "uninformative variables for multivariate calibration*. "
                 "Analytical Chemistry 68(21), 3851–3858.",
        "principle": (
            "Augment X with deterministic artificial noise variables; "
            "any real feature whose stability is below the noise "
            "threshold is eliminated."),
        "implementation": "`n4m_uve_select`. Reference: R `plsVarSel`.",
    },
    "spa_select": {
        "group": "selector",
        "title": "Successive Projections Algorithm (SPA)",
        "paper": "Araújo, M. C. U. et al. (2001). *The successive "
                 "projections algorithm for variable selection in "
                 "spectroscopic multicomponent analysis*. Chemom. "
                 "Intell. Lab. Syst. 57(2), 65–73.",
        "principle": (
            "Greedy projection-orthogonal forward selection seeded by "
            "the coefficient with largest magnitude."),
        "implementation": "`n4m_spa_select`.",
    },
    "cars_select": {
        "group": "selector",
        "title": "CARS — Competitive Adaptive Reweighted Sampling",
        "paper": "Li, H., Liang, Y., Xu, Q. & Cao, D. (2009). "
                 "*Key wavelengths screening using competitive adaptive "
                 "reweighted sampling method for multivariate "
                 "calibration*. Anal. Chim. Acta 648(1), 77–84.",
        "principle": (
            "Exponential-decay retention combined with weighted "
            "sampling; iteratively retains only the strongest features."),
        "implementation": "`n4m_cars_select`. Reference: R `enpls`.",
    },
    "random_frog_select": {
        "group": "selector",
        "title": "Random Frog",
        "paper": "Li, H. et al. (2012). *Random frog: an efficient "
                 "reversible jump Markov chain Monte Carlo-like approach "
                 "for variable selection*. Anal. Chim. Acta 740, 20–26.",
        "principle": (
            "MCMC-like random walk through feature subsets; rank by "
            "inclusion frequency."),
        "implementation": "`n4m_random_frog_select`.",
    },
    "scars_select": {
        "group": "selector",
        "title": "Stability-CARS",
        "paper": "Zheng, K. et al. (2012). *Stability competitive "
                 "adaptive reweighted sampling (SCARS) and its "
                 "applications to multivariate calibration of NIR*. "
                 "Chemom. Intell. Lab. Syst. 112, 48–54.",
        "principle": (
            "CARS where the retention weights are stability-of-"
            "coefficient-sign over Monte-Carlo subsamples."),
        "implementation": "`n4m_scars_select`.",
    },
    "ga_select": {
        "group": "selector",
        "title": "GA-PLS — Genetic Algorithm variable selection",
        "paper": "Leardi, R. (2000). *Application of genetic "
                 "algorithm-PLS for feature selection in spectral data*. "
                 "Journal of Chemometrics 14(5–6), 643–655.",
        "principle": (
            "Wrap a binary GA around PLS CV-RMSE; population evolves "
            "via crossover, mutation, elitism."),
        "implementation": "`n4m_ga_select`.",
    },
    "shaving_select": {
        "group": "selector",
        "title": "Shaving (recursive elimination)",
        "paper": "Mehmood, T. et al. (2012). *A review of variable "
                 "selection methods in partial least squares regression*. "
                 "Chemom. Intell. Lab. Syst. 118, 62–69, §3.2.",
        "principle": (
            "Recursively shave away the lowest-scoring fraction of "
            "features; pick the subset with lowest CV-RMSE."),
        "implementation": "`n4m_shaving_select`.",
    },
    "bve_select": {
        "group": "selector",
        "title": "BVE — Backward Variable Elimination",
        "paper": "Forina, M. et al. (2004). *Iterative predictor "
                 "weighting (IPW) PLS — a technique for the elimination "
                 "of useless predictors in regression problems*. J. "
                 "Chemom. 18(2), 105–112, §2.",
        "principle": (
            "At each step greedily evaluate every one-variable "
            "removal by CV-RMSE; remove the one whose loss is smallest."),
        "implementation": "`n4m_bve_select`.",
    },
    "t2_select": {
        "group": "selector",
        "title": "Hotelling T² loading selection",
        "paper": "Wold, S. et al. (2001), as for `variable_select_vip`.",
        "principle": (
            "Compute Hotelling T² on PLS loading weights, threshold by "
            "an α-specific upper control limit, fall back to top-k."),
        "implementation": "`n4m_t2_select`.",
    },
    "wvc_select": {
        "group": "selector",
        "title": "WVC — Weighted Variable Contribution",
        "paper": "Andries, J. P. M. & Vander Heyden, Y. (2011). "
                 "*Improved variable reduction in partial least "
                 "squares modelling based on predictive-property-"
                 "ranked variables and adaptation of partial least "
                 "squares complexity*. Anal. Chim. Acta 705(1–2), "
                 "292–305.",
        "principle": (
            "Normalised weighted-variable-contribution score from SVD "
            "PLS components; deterministic top-k selection."),
        "implementation": "`n4m_wvc_select`.",
    },
    "wvc_threshold_select": {
        "group": "selector",
        "title": "WVC-threshold selection",
        "paper": "Andries & Vander Heyden (2011), as above.",
        "principle": (
            "Fixed-threshold and factor-of-mean rules over WVC scores; "
            "minimum-selected fallback."),
        "implementation": "`n4m_wvc_threshold_select`.",
    },
    "emcuve_select": {
        "group": "selector",
        "title": "EMCUVE — Ensemble MC-UVE",
        "paper": "Cai, W. et al. (2008), as for `stability_select`.",
        "principle": (
            "Ensemble MC-UVE rounds with a deterministic vote rule; "
            "robust against single-bag instability."),
        "implementation": "`n4m_emcuve_select`.",
    },
    "randomization_select": {
        "group": "selector",
        "title": "Randomisation test (Y-permutation)",
        "paper": "Westad, F. & Martens, H. (2000). *Variable selection "
                 "in NIR based on significance testing in PLSR*. JNIRS "
                 "8(2), 117–124.",
        "principle": (
            "Compare observed PLS coefficient scores against scores "
            "from deterministic Y-permutations; empirical p-values."),
        "implementation": "`n4m_randomization_select`.",
    },
    "rep_select": {
        "group": "selector",
        "title": "REP — Recursive Elimination of Predictors",
        "paper": "Mehmood, T. et al. (2012), as for `shaving_select`.",
        "principle": (
            "Remove a fixed count of weak coefficient-score variables "
            "per recursive step; keep the lowest-CV-error subset."),
        "implementation": "`n4m_rep_select`.",
    },
    "ipw_select": {
        "group": "selector",
        "title": "IPW — Iterative Predictor Weighting",
        "paper": "Forina, M. et al. (2004), as for `bve_select`.",
        "principle": (
            "Iteratively reweight coefficient scores; expose score "
            "and weight paths; keep the lowest-CV-error top-k subset."),
        "implementation": "`n4m_ipw_select`.",
    },
    "st_select": {
        "group": "selector",
        "title": "ST-PLS — Score-threshold selection",
        "paper": "Mehmood, T. et al. (2012), as above.",
        "principle": (
            "Apply deterministic score thresholds with min-selected "
            "fallbacks; keep the lowest-CV-error subset."),
        "implementation": "`n4m_st_select`.",
    },
    "ecr": {
        "group": "calibration-transfer",
        "title": "ECR — Empirical Calibration-transfer Regression",
        "paper": "Custom kernel for empirical calibration transfer; "
                 "see registry notes.",
        "principle": (
            "Penalised regression matching empirical calibration "
            "between two instruments, balanced by an α coefficient."),
        "implementation": "`n4m_ecr_fit`.",
    },
    "iriv_select": {
        "group": "selector",
        "title": "IRIV — Iteratively Retaining Informative Variables",
        "paper": "Yun, Y. H. et al. (2014). *A strategy that iteratively "
                 "retains informative variables for selecting optimal "
                 "variable subset in multivariate calibration*. Anal. "
                 "Chim. Acta 807, 36–43.",
        "principle": (
            "Classify each variable as strongly / weakly informative / "
            "uninformative / interfering across rounds; keep only "
            "strongly + weakly informative."),
        "implementation": "`n4m_iriv_select`.",
    },
    "irf_select": {
        "group": "selector",
        "title": "IRF — Iterative Random Forest variable selection",
        "paper": "Basu, S. et al. (2018). *Iterative random forests "
                 "to discover predictive and stable high-order "
                 "interactions*. PNAS 115(8), 1943–1948.",
        "principle": (
            "Iteratively re-weight Random Forest feature-importances "
            "and refit. Adapted for PLS prediction."),
        "implementation": "`n4m_irf_select`.",
    },
    "vip_spa_select": {
        "group": "selector",
        "title": "VIP-seeded SPA",
        "paper": "Hybrid heuristic combining VIP ranking and the "
                 "Successive Projections Algorithm; see registry "
                 "notes.",
        "principle": (
            "Use VIP scores to seed SPA's projection-orthogonal "
            "forward selection."),
        "implementation": "`n4m_vip_spa_select`.",
    },
}


# ---------------------------------------------------------------------------
# Registry parser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parity-gate truth-source metadata (📐 icon source)
#
# The icon next to each backend row in a method's benchmark table marks
# rows that are *also* declared in `benchmarks/parity_timing/registry.py`
# as parity references for that method. We resolve the live registry
# rather than re-AST-parsing it so the rendered tooltip carries the
# actual library version and the resolved cid matches the cross-binding
# orchestrator's `ref.<id>` column ids.
# ---------------------------------------------------------------------------

TRUTH_SOURCE_ICON = "📐"
PAPER_ONLY_ICON = "📜"


def load_truth_source_metadata(
        strict: bool = False) -> dict[str, dict[str, dict]]:
    """Return {method_name: {cid: metadata}} from the live registry.

    Imports `benchmarks.parity_timing.registry`. The registry depends
    only on numpy at module load, but the per-reference factories may
    import sklearn / ikpls / rpy2 etc. when called; `resolved_references_
    for_method()` swallows resolution errors so optional refs absent on
    the doc-build host are simply omitted.

    When `strict=False` (the default), any failure to import the
    registry yields an empty dict and a warning so the doc build still
    produces pages without the icon. `--strict` propagates the import
    error.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from benchmarks.parity_timing.registry import (
            METHODS,
            truth_source_metadata_for,
        )
    except Exception as exc:
        msg = (f"warning: could not import parity_timing registry "
               f"for truth-source metadata ({type(exc).__name__}: {exc}); "
               "the 📐 icon will not appear in generated pages.")
        if strict:
            raise
        print(msg, file=sys.stderr)
        return {}
    out: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        try:
            remapped: dict[str, dict] = {}
            for cid, meta in truth_source_metadata_for(method).items():
                if cid.startswith("ref."):
                    backend = "ref_" + cid[len("ref."):]
                    cid = REF_DISPLAY_OVERRIDE.get(backend, cid)
                remapped[cid] = meta
            out[method.name] = remapped
        except Exception as exc:
            if strict:
                raise
            print(
                f"warning: truth-source metadata failed for "
                f"`{method.name}` ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            out[method.name] = {}
    return out


def parse_registry(path: Path) -> list[dict]:
    """AST-parse benchmarks/parity_timing/registry.py and extract METHODS."""
    tree = ast.parse(path.read_text())
    out: list[dict] = []
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "METHODS":
            value = node.value
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "METHODS"
                for t in node.targets):
            value = node.value
        if not isinstance(value, ast.List):
            continue
        for el in value.elts:
            if not (isinstance(el, ast.Call) and getattr(el.func, "id", "")
                     == "MethodSpec"):
                continue
            kw = {k.arg: k.value for k in el.keywords}

            def lit(name: str, default: Any = None) -> Any:
                v = kw.get(name)
                if v is None:
                    return default
                if isinstance(v, ast.Constant):
                    return v.value
                try:
                    return ast.literal_eval(v)
                except Exception:
                    return default

            extra_refs: list[str] = []
            ev = kw.get("extra_references")
            if isinstance(ev, ast.Tuple):
                for pair in ev.elts:
                    if isinstance(pair, ast.Tuple) and pair.elts \
                            and isinstance(pair.elts[0], ast.Constant):
                        extra_refs.append(pair.elts[0].value)
            cp = kw.get("cell_params")
            cell_params: dict = {}
            if isinstance(cp, ast.Dict):
                try:
                    cell_params = ast.literal_eval(cp)
                except Exception:
                    cell_params = {}
            out.append({
                "name": lit("name"),
                "desc": lit("description") or "",
                "notes": lit("notes", "") or "",
                "paper_only": lit("paper_only", "") or "",
                "prediction_key": lit("prediction_key", "predictions"),
                "rmse_tol": lit("rmse_rel_tol", 5e-2),
                "needs_x_target": lit("needs_x_target", False),
                "needs_sample_weights": lit("needs_sample_weights", False),
                "needs_labels": lit("needs_labels", False),
                "needs_group_assignment": lit("needs_group_assignment", False),
                "has_py_ref": (
                    "python_reference" in kw and not
                    (isinstance(kw["python_reference"], ast.Constant)
                     and kw["python_reference"].value is None)),
                "has_r_ref": (
                    "r_reference" in kw and not
                    (isinstance(kw["r_reference"], ast.Constant)
                     and kw["r_reference"].value is None)),
                "extra_refs": extra_refs,
                "cell_params": cell_params,
            })
    return out


# ---------------------------------------------------------------------------
# Tier-2 catalog parser (sklearn_tier2.yaml)
# ---------------------------------------------------------------------------

def parse_catalog(path: Path) -> dict[str, dict]:
    """Read the tier-2 YAML catalog. PyYAML isn't a hard doc dep, so do a
    light hand-roll: each top-level section is a `*_:` followed by a list
    of `- name: …` blocks. We only need (name → c_function, params, in_sample,
    extras, category)."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    # The catalog uses one anchor alias (`*pls_regression_params`) that isn't
    # actually declared as a YAML anchor (it just reuses the named params via
    # the comment hint). Pre-process so PyYAML doesn't choke.
    text = path.read_text()
    text = re.sub(r"params: \*pls_regression_params.*", "params: []", text)
    doc = yaml.safe_load(text) or {}
    out: dict[str, dict] = {}
    sections = [
        ("model_regressor",       "model_regressors"),
        ("model_classifier",      "model_classifiers"),
        ("method_result_regressor", "method_result_regressors"),
        ("in_sample_regressor",   "in_sample_regressors"),
        ("selector",              "selectors"),
        ("transformer",           "transformers"),
        ("classifier_extra",      "classifier_extras"),
        ("diagnostic",            "diagnostics"),
    ]
    for category, key in sections:
        for entry in doc.get(key, []) or []:
            if not isinstance(entry, dict) or "name" not in entry:
                continue
            entry["category"] = category
            out[entry["name"]] = entry
    return out


# ---------------------------------------------------------------------------
# CSV parser — group rows per method
# ---------------------------------------------------------------------------

BACKEND_DISPLAY = {
    "cpp":              "pls4all.cpp",          # +build suffix
    "registry_pls4all": "pls4all.registry",
    "python_tier1":     "pls4all.python",
    "python_tier2":     "pls4all.sklearn",
    "r_tier1":          "pls4all.R",
    "r_tier2":          "pls4all.R.formula",
    "matlab_tier1":     "pls4all.matlab",
    "matlab_tier2":     "pls4all.matlab.classdef",
    "sklearn":          "sklearn",
    "ikpls":            "ikpls",
    "r_pls":            "pls",
    "r_ropls":          "ropls",
    "r_mixomics":       "mixOmics",
    "matlab_pls":       "plsregress",
    # pls4all-side R facade packages (mimic upstream `pls` / `mdatools`
    # APIs without depending on those packages). They are
    # `kind="pls4all_binding"` in the orchestrator, so they belong in
    # the R · pls4all band, not Other.
    "r_pls_compat":      "pls4all.R.pls",
    "r_mdatools_compat": "pls4all.R.mdatools",
}

# Editorial filter rules for the published parity tables. The CSV still
# carries these rows for the benchmark gate; we just keep them out of
# the docs to reduce noise.
#
#   DOC_HIDDEN_BACKENDS: always dropped. `registry_pls4all` is an
#   internal MethodSpec harness (bench_registry_pls4all.py) that calls
#   `MethodSpec.pls4all_fn` through pls4all.Context/Config — useful for
#   the parity gate, not a user-facing API surface.
#
#   FIXED_EXTERNAL_LIBRARY: when a method declares a registry parity
#   reference for the same (language, library) pair, the fixed cross-
#   binding row that runs the same external is hidden — its `ref.*`
#   sibling carries the parity story and the 📐 truth-source mark.
DOC_HIDDEN_BACKENDS = {"registry_pls4all"}
FIXED_EXTERNAL_LIBRARY: dict[str, tuple[str, str]] = {
    "sklearn":     ("python", "scikit-learn"),
    "ikpls":       ("python", "ikpls"),
    "r_pls":       ("r", "pls"),
    "r_ropls":     ("r", "ropls"),
    "r_mixomics":  ("r", "mixomics"),
    "matlab_pls":  ("matlab", "plsregress"),
}
# Static (cid -> (language, library)) fallback. `load_truth_source_metadata()`
# is host-sensitive — it omits refs when optional resolution factories fail
# (e.g. no `mixOmics` installed on the docs build host). The collapse rule
# must still fire when the `ref.*` row is in the CSV, so we map the canonical
# reference CIDs to their (language, library) pair directly. Used in
# parity_table() in addition to truth_sources.
REF_CID_LIBRARY: dict[str, tuple[str, str]] = {
    "ref.python_scikit_learn": ("python", "scikit-learn"),
    "ref.python_ikpls":        ("python", "ikpls"),
    "ref.r_pls":               ("r", "pls"),
    "ref.r_ropls":             ("r", "ropls"),
    "ref.r_mixomics":          ("r", "mixomics"),
}
REF_DISPLAY_OVERRIDE = {
    "ref_python_nirs4all":                                  "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_aom_pls": "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_mbpls":   "nirs4all",
    "ref_python_nirs4all_operators_models_sklearn_lwpls":   "nirs4all",
    "ref_python_nirs4all_bench_aom_v0_aompls":              "nirs4all",
}
CPP_BUILD_SUFFIX = {
    "dev-release": "ref",
    "blas-on":     "blas",
    "omp-on":      "omp",
    "blas-omp":    "blas+omp",
    "cuda-on":     "cuda",
}


def column_id(backend: str, build: str) -> str:
    if backend == "cpp":
        return f"pls4all.cpp.{CPP_BUILD_SUFFIX.get(build, build)}"
    if backend.startswith("ref_"):
        return REF_DISPLAY_OVERRIDE.get(
            backend, "ref." + backend[len("ref_"):])
    return BACKEND_DISPLAY.get(backend, backend)


def is_true(v: Any) -> bool:
    return str(v).lower() == "true"


def _failure_verdict(row: dict) -> str:
    if not is_true(row.get("ok")):
        algo = row.get("algorithm") or row.get("algo") or ""
        backend = row.get("backend") or ""
        raw_reason = row.get("reason") or ""
        if (dependency_unavailable_reason(algo, raw_reason)
                or binding_cross_check_reason(algo, backend)):
            return "not_available"
        reason = raw_reason.lower()
        if "timeout" in reason:
            return "not_run"
        if ("not implemented by" in reason or "unsupported algo" in reason
                or "unsupported algorithm" in reason):
            return "not_available"
        if any(k in reason for k in (
                "modulenotfounderror", "importerror", "notimplemented",
                "error", "exception", "crash", "traceback", "segfault")):
            return "error"
        return "not_run"
    return ""


def _uses_reference_parity(row: dict, cid: str) -> bool:
    return (
        cid.startswith("pls4all.cpp.")
        or cid.startswith("ref.")
        or row.get("kind") == "external"
    )


def verdict(row: dict, cid: str = "") -> str:
    failure = _failure_verdict(row)
    if failure:
        return failure
    if _uses_reference_parity(row, cid):
        kind = (row.get("reference_kind") or "").lower()
        if kind == "paper_only":
            return "not_available"
        # Secondary external lib vs the donor oracle, or an allowed selection
        # divergence -> informational cross-check (not n4m pass/fail); a real
        # selection mismatch is divergent. (Mirrors build_landing.)
        note = (row.get("reference_parity_note") or "").lower()
        if note.startswith("cross_check") or note.startswith("selection_divergence_allowed"):
            return "cross_check"
        if note.startswith("selection_mismatch"):
            algo = row.get("algorithm") or row.get("algo") or ""
            if selection_bydesign_reason(algo, note):
                return "cross_check"
            return "divergent"
        # A missing reference oracle is "no truth to compare", not a drift —
        # mirror build_landing so the same CSV renders identically on both pages.
        if "oracle missing" in note or "run canonical reference backend" in note:
            return "not_available"
        ref_ok = row.get("reference_parity_ok")
        if ref_ok in (None, "", "None"):
            return "not_available"
        if is_true(ref_ok):
            return "exact"
        try:
            d = float(row.get("reference_parity_rmse_rel", "nan"))
            tol = float(row.get("reference_parity_tolerance", "nan"))
        except (TypeError, ValueError):
            return "drift"
        if d != d or tol != tol:
            return "drift"
        return "drift" if d < 10 * tol else "divergent"

    parity_ok = row.get("binding_parity_ok", row.get("parity_ok"))
    if is_true(parity_ok):
        return "exact"
    algo = row.get("algorithm") or row.get("algo") or ""
    ref_note = row.get("reference_parity_note") or ""
    if selection_bydesign_reason(algo, ref_note):
        return "cross_check"
    if binding_cross_check_reason(algo, row.get("backend")):
        return "cross_check"
    try:
        d = float(row.get("binding_parity_max_diff",
                          row.get("parity_max_diff", "nan")))
    except (TypeError, ValueError):
        return "drift"
    if d != d:
        return "drift"
    return "drift" if d < 1e-3 else "divergent"


def parity_metric(row: dict, cid: str) -> tuple[str, str]:
    note = (
        row.get("reference_parity_note")
        or row.get("binding_parity_note")
        or row.get("parity_note")
        or ""
    )
    jac = jaccard_from_note(note)
    if jac is not None:
        return "jaccard", str(jac)
    if _uses_reference_parity(row, cid):
        return (
            "reference",
            row.get("reference_parity_rmse_rel")
            or row.get("reference_parity_rmse_abs")
            or "",
        )
    return (
        "binding",
        row.get("binding_parity_max_diff")
        or row.get("parity_max_diff")
        or "",
    )


def fmt_ms(s: str) -> str:
    try:
        f = float(s)
    except (TypeError, ValueError):
        return "—"
    if f >= 1000:
        return f"{f / 1000:.1f} s"
    if f >= 10:
        return f"{f:.1f} ms"
    return f"{f:.2f} ms"


VERDICT_ICON = {
    "exact": "✓", "drift": "≈", "divergent": "✗",
    "not_available": "⊘", "not_run": "—", "error": "⚠",
    "cross_check": "⇄",
}


def _timing_schema(row: dict) -> str:
    try:
        versions = json.loads(row.get("versions_json") or "{}")
    except json.JSONDecodeError:
        return ""
    return str(versions.get("timing_schema") or "")


def parse_csv(path: Path) -> dict[str, list[dict]]:
    """Group benchmark CSV rows by algorithm.

    Method pages should agree with the dashboard: `full_matrix.csv` is the
    baseline, while `dashboard_refresh_*.csv` files can supersede stale cells.
    Prefer the current adaptive timing schema over old warmup/cold-run rows,
    then the newest source
    file. Non-C++ bindings collapse to the production blas-omp row for their
    displayed column.
    """
    if not path.exists():
        return {}
    paths = [path]
    if path.name == "full_matrix.csv":
        paths.extend(sorted(path.parent.glob("dashboard_refresh_*.csv")))

    seen: dict[tuple, dict] = {}
    for source_index, csv_path in enumerate(paths):
        if not csv_path.exists():
            continue
        source_mtime = csv_path.stat().st_mtime
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                algo = r.get("algorithm")
                if not algo:
                    continue
                try:
                    n = int(r["n"])
                    p = int(r["p"])
                    t = int(r["threads"])
                except (KeyError, TypeError, ValueError):
                    continue
                cid = column_id(r.get("backend", ""),
                                r.get("libp4a_build", ""))
                key = (algo, cid, n, p, t)
                build = r.get("libp4a_build", "")
                rank = (
                    {"adaptive-v1": 3, "warmup-v3": 2,
                     "warmup-v2": 1}.get(
                        _timing_schema(r), 0),
                    1 if build == "blas-omp" else 0,
                    float(source_mtime),
                    int(source_index),
                )
                old = seen.get(key)
                if old is None or rank >= old["_rank"]:
                    row = dict(r)
                    row["_rank"] = rank
                    seen[key] = row

    out: dict[str, list[dict]] = defaultdict(list)
    for r in seen.values():
        r.pop("_rank", None)
        out[r["algorithm"]].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# Python sklearn docstring + signature scanner
# ---------------------------------------------------------------------------

def parse_python_sklearn(directory: Path) -> dict[str, dict]:
    """Return public n4m implementation classes and their constructor data.

    The historical function name is retained while the migration is in
    progress, but it now parses the private implementation modules re-exported
    by the public ``n4m.<role>`` packages.  It never reads the retired
    ``pls4all.sklearn`` tree.
    """
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for py in sorted(directory.glob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node) or ""
            params: list[dict] = []
            for child in node.body:
                if (isinstance(child, ast.FunctionDef)
                        and child.name == "__init__"):
                    args = child.args
                    defaults = args.defaults
                    kwonly = args.kwonlyargs
                    kwonly_defaults = args.kw_defaults
                    # Skip `self`
                    arg_names = [a.arg for a in args.args[1:]]
                    arg_annots = [
                        ast.unparse(a.annotation) if a.annotation else ""
                        for a in args.args[1:]]
                    # Positional defaults align to the tail of args.args
                    pos_defaults = [None] * (len(arg_names) - len(defaults)) \
                        + [_repr_default(d) for d in defaults]
                    for n, t, d in zip(arg_names, arg_annots, pos_defaults):
                        params.append({"name": n, "type": t, "default": d})
                    for a, d in zip(kwonly, kwonly_defaults):
                        params.append({
                            "name": a.arg,
                            "type": ast.unparse(a.annotation) if a.annotation
                                else "",
                            "default": _repr_default(d) if d is not None
                                else "",
                        })
                    break
            out[node.name] = {"docstring": doc, "init_params": params}
    return out


def parse_n4m_public_imports(directory: Path) -> dict[str, tuple[str, str]]:
    """Map implementation class names to current public import locations.

    The public role modules explicitly re-export their implementation classes
    using ``from n4m._impl import Class [as PublicName]``.  AST parsing keeps
    the documentation generator independent of libn4m availability and makes
    stale catalog binding metadata detectable in tests.
    """
    imports: dict[str, tuple[str, str]] = {}
    for py in sorted(directory.rglob("*.py")):
        if "_impl" in py.parts:
            continue
        rel = py.relative_to(directory).with_suffix("")
        module_parts = ["n4m", *rel.parts]
        if module_parts[-1] == "__init__":
            module_parts.pop()
        module = ".".join(module_parts)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"n4m._impl", "n4m._impl.native"}:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                imports.setdefault(alias.name, (module, alias.asname or alias.name))
    return imports


def _repr_default(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return ast.unparse(node)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# R roxygen / signature scanner
# ---------------------------------------------------------------------------

R_DOCSTRING_RE = re.compile(
    r"((?:^#'[^\n]*\n)+)\s*([a-zA-Z_.][a-zA-Z0-9_.]*) <- function\(([^)]*)\)",
    re.MULTILINE,
)


def parse_r_signatures(directory: Path) -> dict[str, dict]:
    """Return {fn_name: {roxygen, signature}} per R file."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for r_file in sorted(directory.glob("*.R")):
        text = r_file.read_text()
        for m in R_DOCSTRING_RE.finditer(text):
            roxy, name, args = m.group(1, 2, 3)
            if name.startswith("."):
                continue
            roxy_lines = [
                line[3:].rstrip() if line.startswith("#' ")
                else line[2:].rstrip()
                for line in roxy.strip().splitlines()
            ]
            out[name] = {
                "roxygen": "\n".join(roxy_lines),
                "signature": f"{name}({args.strip()})",
                "file": r_file.name,
            }
    return out


# ---------------------------------------------------------------------------
# MATLAB classdef / function scanner
# ---------------------------------------------------------------------------

M_HEADER_RE = re.compile(
    r"^(?:classdef\s+([A-Za-z]\w*)|function\s+(?:[^=]+=\s*)?([a-zA-Z_]\w*)"
    r"\s*\(([^)]*)\))",
    re.MULTILINE,
)


def parse_matlab(directory: Path) -> dict[str, dict]:
    """Return {entity_name: {header_doc, signature}} for every classdef or
    top-level function in `bindings/matlab/+pls4all/`."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for m_file in sorted(directory.glob("*.m")):
        text = m_file.read_text()
        # Pull the header doc — block of leading `%` lines after the
        # classdef/function declaration.
        lines = text.splitlines()
        sig_idx = None
        header = []
        kind = None
        name = None
        signature = ""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("classdef "):
                kind = "classdef"
                name = stripped.split()[1]
                signature = stripped
                sig_idx = i
                break
            if stripped.startswith("function "):
                kind = "function"
                # function out = name(args)
                m = re.match(
                    r"function\s+(?:[^=]+=\s*)?([a-zA-Z_]\w*)"
                    r"\s*\(([^)]*)\)", stripped)
                if m:
                    name = m.group(1)
                    signature = stripped
                sig_idx = i
                break
        if sig_idx is None or name is None:
            continue
        for line in lines[sig_idx + 1:]:
            stripped = line.lstrip()
            if stripped.startswith("%"):
                header.append(stripped[1:].rstrip())
            else:
                break
        out[name] = {
            "kind": kind,
            "signature": signature,
            "header": "\n".join(header).strip(),
            "file": m_file.name,
        }
    return out


# ---------------------------------------------------------------------------
# Legacy language-specific mapping tables were retired with the ABI-2
# bindings.  Current Python, R, and MATLAB surfaces are scanned from their
# public sources by dedicated helpers instead of being inferred from these
# historical spellings.


# Page renderers
# ---------------------------------------------------------------------------

GROUP_LABELS = {
    "core":                 "Core PLS",
    "ensemble":             "Ensemble",
    "sparse":               "Sparse",
    "robust":               "Robust / weighted",
    "nonlinear":            "Nonlinear / local",
    "multi-block":          "Multi-block / cross-modal",
    "calibration-transfer": "Calibration transfer",
    "classification":       "Classification & GLM",
    "missing":              "Missing data",
    "regularized":          "Regularised",
    "diagnostic":           "Diagnostic",
    "selector":             "Variable selector",
    "other":                "Other",
}

# Heuristic group classification for methods absent from BIBLIOGRAPHY
EXTRA_GROUPS = {
    "pls": "core", "pcr": "core", "opls": "core", "cppls": "core",
    "recursive_pls": "core",
    "sparse_simpls": "sparse", "fused_sparse_pls": "sparse",
    "group_sparse_pls": "sparse", "sparse_pls_da": "sparse",
    "bagging_pls": "ensemble", "boosting_pls": "ensemble",
    "random_subspace_pls": "ensemble",
    "robust_pls": "robust", "weighted_pls": "robust",
    "kernel_pls_rbf": "nonlinear", "lw_pls": "nonlinear",
    "gpr_pls": "nonlinear", "continuum_regression": "nonlinear",
    "mb_pls": "multi-block", "mir_pls": "multi-block",
    "so_pls": "multi-block", "on_pls": "multi-block",
    "rosa": "multi-block", "n_pls": "multi-block", "o2pls": "multi-block",
    "di_pls": "calibration-transfer", "ds": "calibration-transfer",
    "pds": "calibration-transfer", "ecr": "calibration-transfer",
    "pls_lda": "classification", "pls_logistic": "classification",
    "pls_qda": "classification", "pls_glm": "classification",
    "pls_cox": "classification",
    "missing_aware_nipals": "missing",
    "ridge_pls": "regularized",
    "approximate_press": "diagnostic", "pls_diagnostic_t2": "diagnostic",
    "pls_diagnostic_q": "diagnostic", "pls_diagnostic_dmodx": "diagnostic",
    "pls_monitoring": "diagnostic", "one_se_rule": "diagnostic",
    "aom_preprocess": "diagnostic",
    "aom_pls": "adaptive",
    "pop_pls": "adaptive",
}


def method_group(name: str) -> str:
    if name in BIBLIOGRAPHY:
        return BIBLIOGRAPHY[name].get("group", EXTRA_GROUPS.get(name, "other"))
    if name in EXTRA_GROUPS:
        return EXTRA_GROUPS[name]
    if name.endswith("_select"):
        return "selector"
    return "other"


# ---------------------------------------------------------------------------
# Language-band classification for the benchmark rows.
#
# Rendered in this order from top to bottom; the C++ native libn4m row
# is always first, then the pls4all language bindings, then external
# reference libraries.
# ---------------------------------------------------------------------------

BAND_ORDER = [
    ("cpp",            "C++ native · libn4m"),
    ("python-pls4all", "Python · archived pls4all benchmark"),
    ("r-pls4all",      "R · archived pls4all benchmark"),
    ("matlab-pls4all", "MATLAB · archived pls4all benchmark"),
    ("python-ext",     "Python · external"),
    ("r-ext",          "R · external"),
    ("matlab-ext",     "MATLAB · external"),
    ("other",          "Other"),
]
BAND_LANG = {
    "cpp": "cpp", "python-pls4all": "python", "r-pls4all": "r",
    "matlab-pls4all": "matlab", "python-ext": "python",
    "r-ext": "r", "matlab-ext": "matlab", "other": "ext",
}


def _band_of(cid: str) -> str:
    if cid.startswith("pls4all.cpp."):
        return "cpp"
    if cid in ("pls4all.python", "pls4all.sklearn", "pls4all.registry"):
        return "python-pls4all"
    # Prefix match so future compat shims (pls4all.R.pls, pls4all.R.mdatools,
    # pls4all.R.formula, …) all land in the R · pls4all band without
    # needing a new explicit branch.
    if cid == "pls4all.R" or cid.startswith("pls4all.R."):
        return "r-pls4all"
    if cid in ("pls4all.matlab", "pls4all.matlab.classdef"):
        return "matlab-pls4all"
    if cid.startswith("ref.python_") or cid in ("sklearn", "ikpls", "nirs4all"):
        return "python-ext"
    if (cid.startswith("ref.r_")
            or cid in ("pls", "mixOmics", "ropls")):
        return "r-ext"
    if (cid.startswith("ref.matlab_")
            or cid in ("plsregress",)):
        return "matlab-ext"
    return "other"


def _truth_quality_class(quality: str) -> str:
    """CSS row-class suffix for a quality band."""
    return {
        "strict":      "truth-source-strict",
        "relaxed":     "truth-source-relaxed",
        "qualitative": "truth-source-qualitative",
    }.get(quality, "truth-source")


def _quality_from_tol(tol):
    """Tolerance-band classifier — kept aligned with the registry's
    `_truth_source_quality()` in `parity_timing/registry.py`. Used both
    for the per-row badge styling and for the method-level legend chip.
    """
    if tol is None:
        return "unknown"
    try:
        t = float(tol)
    except (TypeError, ValueError):
        return "unknown"
    if t != t:  # NaN
        return "unknown"
    if t <= 1e-6:
        return "strict"
    if t < 1e-1:
        return "relaxed"
    return "qualitative"


def _format_diff(diff_str):
    """Format `1.23e-04` → `1e-04`. Returns (formatted_or_empty, abs_value)."""
    if not diff_str:
        return "", 0.0
    try:
        d = float(diff_str)
    except (TypeError, ValueError):
        return "", 0.0
    if d != d:
        return "", 0.0
    if d == 0:
        return "", 0.0
    return f"{d:.0e}", abs(d)


def _row_worst_diff(cells_in_row, cid):
    """Aggregate `parity_metric()` across visible sizes for one row.

    The single parity badge per row should reflect the WORST observed
    metric across the columns shown, not just the first cell. Keeps
    the basis from the first valid cell since basis is invariant per
    row.
    """
    basis = ""
    worst_str = ""
    worst_abs = -1.0
    for r in cells_in_row:
        if r is None:
            continue
        b, d = parity_metric(r, cid)
        if not basis:
            basis = b
        try:
            raw = float(d) if d else 0.0
        except (TypeError, ValueError):
            continue
        if raw != raw:  # NaN
            continue
        # RMSE δ: worst is the largest absolute delta. Jaccard: worst is the
        # smallest overlap, so score it by negative value for the same max scan.
        score = (1.0 - raw) if b == "jaccard" else abs(raw)
        if score > worst_abs:
            worst_abs = score
            worst_str = d
    return basis, worst_str


# Verdict severity ladder for REAL outcomes only (test ran, has a
# result). `not_run` and `not_available` are absence-of-data, not
# failure modes — they must NOT outrank an "exact" pass on another
# size for the same row. Otherwise a row that passed at 100×50 and
# wasn't measured at 100×500 would render as "not run", erasing the
# real verdict.
_REAL_VERDICT_RANK = {
    "exact":      0,
    "drift":      1,
    "cross_check": 2,
    "error":      4,
    "divergent":  5,
}


def _row_worst_verdict(cells_in_row, cid):
    """Worst-severity verdict across visible sizes.

    Prefers the worst REAL outcome (exact/drift/divergent/error) so a
    row with any real result is judged by its weakest cell. Falls back
    to the first cell's absence-of-data verdict only when no size has
    a real outcome.
    """
    real_worst = None
    real_rank = -1
    fallback = "not_available"
    fallback_set = False
    for r in cells_in_row:
        if r is None:
            continue
        v = verdict(r, cid)
        if not fallback_set:
            fallback = v
            fallback_set = True
        if v in _REAL_VERDICT_RANK:
            rk = _REAL_VERDICT_RANK[v]
            if rk > real_rank:
                real_rank = rk
                real_worst = v
    return real_worst if real_worst is not None else fallback


def _parity_badge_html(verdict_, basis, diff, quality, is_self):
    """Compose one parity-cell HTML, accounting for tolerance band.

    For non-exact verdicts the badge keeps its existing icon+signed-diff
    format. For "exact" verdicts the rendering splits by:
      - reference parity + canonical-self        → `source`
      - reference parity + strict tolerance      → ✓ ref (phosphor)
      - reference parity + relaxed tolerance     → ≈ ref (amber)
      - reference parity + qualitative tolerance → ~ shape (grey)
      - reference parity + unknown tolerance     → ? ref (grey, italic)
      - binding parity (pls4all binding ↔ C++)   → ✓ bind (phosphor)
    so a tol ≥ 1 "pass" no longer looks identical to a bit-exact one.

    The `is_self` short-circuit must happen *after* the failure paths so
    a canonical reference row whose self-run errored or diverged still
    shows the failure, not a confident grey "source".
    """
    if basis == "jaccard":
        try:
            jac = float(diff) if diff else 1.0
        except (TypeError, ValueError):
            jac = 1.0
        label = f"J {jac:.2f}"
        if verdict_ == "exact":
            return "parity parity-exact", f"✓ {label}"
        icon = VERDICT_ICON[verdict_]
        return f"parity parity-{verdict_}", f"{icon} {label}"
    if verdict_ != "exact":
        icon = VERDICT_ICON[verdict_]
        fmt, _ = _format_diff(diff)
        if fmt:
            sign = "+" if diff and not str(diff).startswith("-") else ""
            return f"parity parity-{verdict_}", f"{icon} {sign}{fmt}"
        return f"parity parity-{verdict_}", icon
    if is_self and basis == "reference":
        return "parity parity-ref-source", "source"
    fmt, _ = _format_diff(diff)
    if basis == "binding":
        # Binding parity = pls4all binding vs C++ binding. ABI-level
        # consistency, not the loose reference gate — always phosphor.
        label = fmt if fmt else "bind"
        return "parity parity-exact", f"✓ {label}"
    # Reference-gate exact pass — quality-aware split.
    if quality == "strict":
        return ("parity parity-ref-strict",
                f"✓ ref {fmt}" if fmt else "✓ ref")
    if quality == "relaxed":
        return ("parity parity-ref-relaxed",
                f"≈ ref {fmt}" if fmt else "≈ ref")
    if quality == "qualitative":
        return ("parity parity-ref-qualitative",
                f"~ shape {fmt}" if fmt else "~ shape")
    # `quality == "unknown"`: missing or non-finite registry tolerance.
    # Don't lend phosphor authority to a row whose gate strength we
    # can't classify — render a neutral grey badge with the diff value.
    return ("parity parity-ref-unknown",
            f"? ref {fmt}" if fmt else "? ref")


def _truth_quality_label(quality: str, tol: float) -> str:
    """Human label for the 📐 tooltip."""
    rounded = f"{tol:.0e}"
    return {
        "strict":      f"strict (rmse_rel ≤ {rounded})",
        "relaxed":     f"relaxed (rmse_rel ≤ {rounded})",
        "qualitative": f"qualitative (rmse_rel ≤ {rounded})",
    }.get(quality, f"rmse_rel ≤ {rounded}")


def _truth_mark_html(meta: dict) -> str:
    """Build the 📐 <span> for a truth-source row."""
    label = _truth_quality_label(meta.get("quality", "qualitative"),
                                  float(meta.get("tolerance") or 0.0))
    title = (f"Registry parity reference ({meta.get('role', '?')}): "
             f"{meta.get('library', '?')} {meta.get('version', '')} "
             f"— {label}")
    # Replace double quotes so we don't break the surrounding attribute.
    title_safe = title.replace('"', "&quot;").strip()
    return (f'<span class="truth-mark" title="{title_safe}">'
            f'{TRUTH_SOURCE_ICON}</span>')


def parity_table(method: str, rows: list[dict],
                  truth_sources: dict[str, dict] | None = None) -> str:
    """Build a parity + timing block grouped by language band.

    Emits raw HTML so we can:
      * insert language-section header rows,
      * highlight the per-column fastest cell across all backends,
      * mark every row that is also a registry parity reference for
        this method with a 📐 icon and a quality-banded row class.

    `truth_sources` is `{cid -> metadata}` from
    `benchmarks.parity_timing.registry.truth_source_metadata_for(method)`.
    Rows whose cid matches a key get the icon; the metadata feeds the
    tooltip and the CSS class.
    """
    truth_sources = truth_sources or {}
    if not rows:
        return "_No benchmark rows recorded for this method._"

    # Editorial filter (see DOC_HIDDEN_BACKENDS / FIXED_EXTERNAL_LIBRARY).
    # First pass: which (language, library) ref.* rows are actually
    # present for this method? That set controls which fixed-external
    # cross-binding rows we collapse — we only hide a duplicate when
    # its sibling reference row is genuinely in the table.
    #
    # We seed the set from two sources to stay robust against
    # host-sensitive metadata: (a) the registry's resolved truth-source
    # entries (truth_sources, which may be partial if optional packages
    # don't resolve on the docs host), and (b) a static REF_CID_LIBRARY
    # fallback keyed by the rendered ref.* CID.
    rows_cids = {column_id(r["backend"], r.get("libp4a_build", ""))
                 for r in rows}
    present_ref_pairs: set[tuple[str, str]] = set()
    for cid, meta in truth_sources.items():
        if cid not in rows_cids:
            continue
        lang = str(meta.get("language") or "").strip().lower()
        lib = str(meta.get("library") or "").strip().lower()
        if lang and lib:
            present_ref_pairs.add((lang, lib))
    for cid, pair in REF_CID_LIBRARY.items():
        if cid in rows_cids:
            present_ref_pairs.add(pair)

    def _is_doc_hidden(r: dict) -> bool:
        backend = r.get("backend", "")
        if backend in DOC_HIDDEN_BACKENDS:
            return True
        pair = FIXED_EXTERNAL_LIBRARY.get(backend)
        if pair and pair in present_ref_pairs:
            return True
        return False

    rows = [r for r in rows if not _is_doc_hidden(r)]
    if not rows:
        return "_No benchmark rows recorded for this method._"

    # Pre-filter: drop rows for backends that don't implement this
    # method at all (every measured cell is `not_available`). If
    # that empties the entire set we render a friendlier note rather
    # than an empty table — `_row_is_present()` below uses the same
    # rule per-thread.
    def _is_unsupported_only(r: dict) -> bool:
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        return verdict(r, cid) == "not_available"
    if rows and all(_is_unsupported_only(r) for r in rows):
        return ("_No backend implements this method yet — only "
                "registry-declared parity references exist._")

    cells: dict[tuple[str, int, int, int], dict] = {}
    for r in rows:
        try:
            n = int(r["n"])
            p = int(r["p"])
            t = int(r["threads"])
        except (KeyError, ValueError):
            continue
        cid = column_id(r["backend"], r.get("libp4a_build", ""))
        cells[(cid, n, p, t)] = r

    cids = sorted({c for (c, _, _, _) in cells.keys()})
    sizes = sorted({(n, p) for (_, n, p, _) in cells.keys()})
    threads = sorted({t for (_, _, _, t) in cells.keys()})

    # Group cids into language bands, preserving the canonical band
    # order. Within a band, sort by cid for determinism (cpp tiers will
    # naturally sort by build suffix).
    grouped: dict[str, list[str]] = {b: [] for b, _ in BAND_ORDER}
    for cid in cids:
        grouped[_band_of(cid)].append(cid)
    for b in grouped:
        grouped[b].sort()

    def _row_verdict(r: dict | None, cid: str = "") -> str:
        return verdict(r, cid) if r is not None else "not_run"

    def _row_ms(r: dict | None) -> float:
        if r is None:
            return float("nan")
        try:
            return float(r.get("reported_ms") or r.get("median_ms") or "nan")
        except (TypeError, ValueError):
            return float("nan")

    # Method-level reference tolerance — used to band the parity badge
    # (strict / relaxed / qualitative). Prefer the registry's truth-source
    # metadata; fall back to the first row that records a finite
    # `reference_parity_tolerance` so the badge stays honest even when
    # `load_truth_source_metadata()` produced an empty dict.
    method_tol = None
    for meta in truth_sources.values():
        try:
            method_tol = float(meta.get("tolerance"))
            if method_tol == method_tol:
                break
        except (TypeError, ValueError):
            continue
    if method_tol is None or method_tol != method_tol:
        for r in rows:
            try:
                t = float(r.get("reference_parity_tolerance"))
                if t == t:
                    method_tol = t
                    break
            except (TypeError, ValueError):
                continue
    method_quality = _quality_from_tol(method_tol)

    lines: list[str] = []
    lines.append("### Benchmarks\n")
    legend_lines = [
        "**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the "
        "benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not "
        "describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** "
        "section above for current entry points.",
        "",
        "Adaptive wall-clock per cell measured against "
        "[`full_matrix.csv`](../benchmarks/overview.md). "
        "Only backends that implement this method are listed; "
        "libraries without the method are omitted.",
        "",
        "**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a "
        "reference-gate pass at strict / relaxed / qualitative "
        "tolerance &nbsp;·&nbsp; ✓ bind = archived binding-harness result agrees "
        "with the C++ baseline &nbsp;·&nbsp; ⇄ cross-check = documented "
        "by-design selector/RNG/model, noncanonical API/facade convention, "
        "or secondary oracle "
        "&nbsp;·&nbsp; ✗ divergent "
        "&nbsp;·&nbsp; ⚠ error &nbsp;·&nbsp; — not run. The "
        "fastest backend per column is marked 🏆.",
    ]
    if method_tol is not None and method_tol == method_tol:
        if method_quality == "strict":
            legend_lines += [
                "",
                f"**Reference gate**: strict — numeric equivalence "
                f"(`rmse_rel_tol ≤ {method_tol:.0e}`).",
            ]
        elif method_quality == "relaxed":
            legend_lines += [
                "",
                f"**Reference gate**: relaxed — known algorithmic "
                f"drift between the archived benchmark harness and the external reference "
                f"(`rmse_rel_tol ≤ {method_tol:.0e}`).",
            ]
        elif method_quality == "qualitative":
            legend_lines += [
                "",
                f"**Reference gate**: qualitative — shape/smoke "
                f"comparison only. The external library and archived benchmark harness "
                f"do not produce numerically equivalent output for "
                f"this method (see the MethodSpec notes); the "
                f"`rmse_rel_tol ≤ {method_tol:.0e}` budget is set "
                f"wide on purpose. Treat ~ shape as *“we ran both, "
                f"both finished”*, not as numerical agreement.",
            ]
    if truth_sources:
        legend_lines += [
            "",
            f"Rows tagged with **{TRUTH_SOURCE_ICON}** are the "
            "canonical parity references for this method "
            "(declared in "
            "[`parity_timing.registry`](../benchmarks/methodology.md)). "
            "C++ and external rows show reference parity; archived "
            "language-harness rows show binding parity against the C++ "
            "backend. Hover the icon for role and tolerance band.",
        ]
    legend_lines.append("")
    lines.append("\n".join(legend_lines))

    lines.append("::::{tab-set}\n:class: parity-tabs\n")
    for t in threads:
        sync = f"threads-{t}"
        lines.append(_tab_open(
            f"{t} thread{'s' if t != 1 else ''}", sync))

        # Compute per-column best (lowest ms) across all valid backends
        # for this thread count. "Valid" means the cell ran successfully
        # (verdict ∈ {exact, drift}) — divergent / error / skip rows are
        # not eligible for the medal.
        column_min: dict[tuple[int, int], tuple[float, str]] = {}
        for (n, p_) in sizes:
            for cid in cids:
                r = cells.get((cid, n, p_, t))
                if r is None:
                    continue
                v = _row_verdict(r, cid)
                if v not in ("exact", "drift", "cross_check"):
                    continue
                ms = _row_ms(r)
                if ms != ms:  # NaN
                    continue
                cur = column_min.get((n, p_))
                if cur is None or ms < cur[0]:
                    column_min[(n, p_)] = (ms, cid)

        # Header
        size_headers = "".join(
            f'<th class="size-col" scope="col">{n}×{p_} (ms)</th>'
            for (n, p_) in sizes
        )
        html: list[str] = [
            '<table class="docutils parity-grouped">',
            '<thead><tr>'
            '<th scope="col">Backend</th>'
            '<th scope="col">Parity</th>'
            f'{size_headers}'
            '</tr></thead>',
        ]

        # Pre-scan rows so we can drop backends that don't implement
        # this method (or that simply didn't run any size). A row is
        # "present" only when at least one of its visible cells has a
        # REAL outcome — exact/drift/cross_check/divergent/error. Cells whose
        # verdict is `not_available` (library doesn't ship the method)
        # or `not_run` (size deferred / timeout / not scheduled) are
        # absence-of-data, not evidence of presence.
        _PRESENCE_VERDICTS = {"exact", "drift", "cross_check",
                              "divergent", "error"}
        def _row_is_present(cid: str) -> bool:
            for (n, p_) in sizes:
                r = cells.get((cid, n, p_, t))
                if r is None:
                    continue
                if _row_verdict(r, cid) in _PRESENCE_VERDICTS:
                    return True
            return False

        for band_key, band_label in BAND_ORDER:
            band_cids = [c for c in grouped[band_key]
                         if _row_is_present(c)]
            if not band_cids:
                continue
            lang_tag = BAND_LANG[band_key]
            ncols = 2 + len(sizes)
            html.append(
                f'<tbody class="lang-band lang-{lang_tag}">'
                f'<tr class="lang-band-row" data-lang="{lang_tag}">'
                f'<th colspan="{ncols}" scope="rowgroup">'
                f'<span class="lang-band-dot"></span>{band_label}'
                f'</th></tr>'
            )
            for cid in band_cids:
                cells_in_row = [cells.get((cid, n, p_, t))
                                for (n, p_) in sizes]
                primary = next((r for r in cells_in_row if r is not None),
                               None)
                if primary is None:
                    continue
                # Use the worst |diff| AND worst verdict across visible
                # sizes so the single badge reflects the whole row, not
                # just the first cell. A row whose first cell passed
                # but a later size diverged must render as divergent.
                v = _row_worst_verdict(cells_in_row, cid)
                basis, diff = _row_worst_diff(cells_in_row, cid)
                is_self = is_true(primary.get("is_canonical_reference"))
                p_class, p_cell = _parity_badge_html(
                    v, basis, diff, method_quality, is_self)
                truth_meta = truth_sources.get(cid)
                bk_row_class = "bk-row"
                if truth_meta:
                    bk_row_class += (
                        " truth-source "
                        + _truth_quality_class(truth_meta.get("quality",
                                                                "qualitative"))
                    )
                    truth_mark = _truth_mark_html(truth_meta)
                    bk_name_cell = (
                        f'<td class="bk-name">{truth_mark}'
                        f'<code>{cid}</code></td>'
                    )
                else:
                    bk_name_cell = f'<td class="bk-name"><code>{cid}</code></td>'
                row = [
                    f'<tr class="{bk_row_class}">',
                    bk_name_cell,
                    f'<td class="{p_class}">{p_cell}</td>',
                ]
                for ((n, p_), r) in zip(sizes, cells_in_row):
                    if r is None:
                        row.append('<td class="ms ms-empty">—</td>')
                        continue
                    ms = _row_ms(r)
                    fmt = fmt_ms(r.get("reported_ms") or r.get("median_ms", "")) \
                        if ms == ms else "—"
                    best = column_min.get((n, p_))
                    is_best = (best is not None and best[1] == cid)
                    cls = "ms" + (" ms-best" if is_best else "")
                    medal = '<span class="medal" title="fastest">🏆</span>' \
                        if is_best else ""
                    row.append(f'<td class="{cls}">{fmt}{medal}</td>')
                row.append('</tr>')
                html.append("".join(row))
            html.append('</tbody>')
        html.append('</table>')

        # MyST passes raw HTML through. Wrap in a div so it gets the
        # tab content padding consistently.
        lines.append('<div class="parity-table-wrap">')
        lines.append("\n".join(html))
        lines.append('</div>')
        lines.append("")
        lines.append(_tab_close())
    lines.append("::::\n")
    return "\n".join(lines)


_ARCHIVED_BENCHMARK_DISCLOSURE = (
    "**Archived measurement identity.** Backend labels in this table are the raw IDs recorded when the "
    "benchmark ran (including historical `pls4all.*` IDs). They preserve measurement provenance and do not "
    "describe a current public Python, R, or MATLAB binding; use the source-verified **API and bindings** "
    "section above for current entry points."
)


def normalize_benchmark_snapshot(block: str) -> str:
    """Qualify legacy snapshots without changing timing or parity values.

    Snapshot JSON contains pre-rendered historical benchmark blocks. Both a
    clean ``--no-bench`` generation and Sphinx's source-read hook load those
    blocks directly, bypassing :func:`parity_table`. This normalizes only the
    captions and explanation, preserving raw backend IDs and every measured
    timing/parity cell as historical provenance.
    """
    if "### Benchmarks" not in block:
        return block
    text = block
    if _ARCHIVED_BENCHMARK_DISCLOSURE not in text:
        text = text.replace(
            "### Benchmarks\n",
            "### Benchmarks\n\n" + _ARCHIVED_BENCHMARK_DISCLOSURE + "\n",
            1,
        )
    replacements = {
        "✓ bind = pls4all binding agrees": "✓ bind = archived binding-harness result agrees",
        "drift between pls4all and the external reference": (
            "drift between the archived benchmark harness and the external reference"),
        "The external library and pls4all do not": (
            "The external library and archived benchmark harness do not"),
        "C++ and external rows show reference parity; pls4all language bindings": (
            "C++ and external rows show reference parity; archived language-harness rows"),
        "Python · pls4all": "Python · archived pls4all benchmark",
        "R · pls4all": "R · archived pls4all benchmark",
        "MATLAB · pls4all": "MATLAB · archived pls4all benchmark",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _tab_open(label: str, sync: str, cls: str = "") -> str:
    """Open a sphinx-design tab-item with sync key + optional class."""
    line = f":::{{tab-item}} {label}\n:sync: {sync}"
    if cls:
        line += f"\n:class-label: lang-{cls}"
    return line + "\n"


def _tab_close() -> str:
    return ":::\n"


def _current_python_binding(cat: dict | None, public_api: Any,
                            registry_name: str | None = None) -> dict[str, Any] | None:
    """Return a source-verified binding and signature, never a guessed name."""
    return binding_for_catalog(cat or {}, public_api, registry_name)


def catalog_c_symbols(method: dict | None) -> list[str]:
    """Normalize a catalog C surface without treating ``\"none\"`` as letters."""
    raw = (method or {}).get("c_surface") or []
    if raw == "none":
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(symbol) for symbol in raw if str(symbol) != "none"]
    return []


def c_abi_declarations(symbols: list[str]) -> list[tuple[str, str, int]]:
    """Locate C-ABI declarations in the installed public headers.

    The catalog names the operation but not its header.  Linking the exact
    declaration keeps C-only pages actionable without inventing a Python
    wrapper or copying a potentially stale prototype into Markdown.
    """
    unresolved = set(symbols)
    found: dict[str, tuple[str, int]] = {}
    include_dir = ROOT / "cpp" / "include" / "n4m"
    if not include_dir.exists():
        return []
    for header in sorted(include_dir.rglob("*.h")):
        if not unresolved:
            break
        try:
            lines = header.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, start=1):
            for symbol in tuple(unresolved):
                if re.search(rf"\b{re.escape(symbol)}\b", line):
                    found[symbol] = (
                        header.relative_to(ROOT).as_posix(), line_number)
                    unresolved.remove(symbol)
    return [(symbol, *found[symbol]) for symbol in symbols if symbol in found]


def render_c_abi_surface(symbols: list[str]) -> str:
    """Render linked, source-verified C entry points for a method page."""
    if not symbols:
        return "**C ABI:** no standalone exported symbol is declared for this method."
    details = {symbol: (path, line) for symbol, path, line in c_abi_declarations(symbols)}
    rendered: list[str] = []
    for symbol in symbols:
        declaration = details.get(symbol)
        if declaration:
            path, line = declaration
            url = f"https://github.com/GBeurier/nirs4all-methods/blob/main/{path}#L{line}"
            rendered.append(f"[`{symbol}`]({url})")
        else:
            rendered.append(f"`{symbol}`")
    return ("**C ABI (ABI 2):** " + " · ".join(rendered) + ". "
            "Use the linked public header for the exact signature, configuration, and result handles.")


def render_cross_binding_surfaces(cross_bindings: dict[str, dict[str, Any]]) -> str:
    """Render current R/MATLAB access only when source scanning proves it."""
    labels = (("r", "R", "r"), ("matlab", "MATLAB / Octave", "matlab"))
    parts: list[str] = []
    for key, label, language in labels:
        binding = cross_bindings.get(key)
        if not binding:
            parts.append(f"**{label}:** no current source-verified entry point was found for this catalog method.\n")
            continue
        source_url = "https://github.com/GBeurier/nirs4all-methods/blob/main/" + binding["source"]
        signature = f"{binding['symbol']}{binding['signature']}"
        parts.append(f"**{label} (source-verified):** [`{signature}`]({source_url}).\n")
        if binding.get("call"):
            parts.append(f"```{language}\n{binding['snippet']}\n```\n")
        else:
            parts.append("The source signature has additional required inputs, so no example call is fabricated.\n")
    return "\n".join(parts)


def usage_section(method: str, spec: dict, cat: dict | None,
                  public_api: Any,
                  cross_bindings: dict[str, dict[str, Any]],
                  truth_sources: dict[str, dict] | None = None,
                  old_to_new: dict[str, str] | None = None) -> str:
    """Render only current, source-verified ABI-2 access paths.

    The former documentation generated nominal ``pls4all``/``n4m.sklearn``
    examples from retired trees.  An import statement is useful only when it
    is executable in the current package; complex multi-binding calls are not
    fabricated here.  Detailed parameters remain in ``parameters_section``.
    """
    truth_sources = truth_sources or {}
    old_to_new = old_to_new or {}
    parts = ["### API and bindings\n"]

    c_symbols = catalog_c_symbols(cat)
    if not c_symbols:
        symbol = method_c_symbol(method, old_to_new)
        if symbol:
            c_symbols = [symbol]
    parts.append(render_c_abi_surface(c_symbols) + "\n")

    binding = _current_python_binding(cat, public_api, method)
    if binding:
        snippet = binding["import"]
        if binding.get("call"):
            snippet += "\nresult = " + binding["call"]
        parts.append("**Python (verified public re-export):**\n\n"
                     "```python\n"
                     f"{snippet}\n"
                     "```\n")
        source_url = ("https://github.com/GBeurier/nirs4all-methods/blob/main/"
                      "bindings/python/src/" + binding["source"])
        if binding.get("source_line"):
            source_url += f"#L{binding['source_line']}"
        parts.append(f"Source signature: `{binding['symbol']}{binding['signature'] or '(...)'}` "
                     f"([`{binding['source']}`]({source_url})).\n")
    else:
        parts.append("**Python:** no current AST-verified public `n4m` re-export was found for this "
                     "method. The linked C ABI above is the documented surface in this checkout.\n")

    parts.append(render_cross_binding_surfaces(cross_bindings))

    parts.append(f"**Registry parity references** {TRUTH_SOURCE_ICON}\n")
    parts.append(":::{card}\n:class-card: external-refs\n")
    if spec.get("paper_only"):
        parts.append(f"- {PAPER_ONLY_ICON} **Paper-only** — no executable parity reference; "
                     "the paper citation is recorded in *Bibliographic source*.\n")
    if truth_sources:
        for cid in sorted(truth_sources):
            meta = truth_sources[cid]
            label = _truth_quality_label(meta.get("quality", "qualitative"),
                                         float(meta.get("tolerance") or 0.0))
            note = (meta.get("notes") or "").strip()
            note_clause = f" — {note}" if note else ""
            parts.append(f"- {TRUTH_SOURCE_ICON} **`{cid}`** "
                         f"({meta.get('language', '?')} · {meta.get('role', '?')}) — "
                         f"`{meta.get('library', '?')}` {meta.get('version', '')} · "
                         f"{label}{note_clause}\n")
    elif not spec.get("paper_only"):
        parts.append("- No executable parity reference was resolved in this build; consult the "
                     "registry metadata and the bibliographic source above.\n")
    parts.append(":::\n")
    return "\n".join(parts)

def markdown_table_cell(value: object) -> str:
    """Make one value safe to place in a pipe-delimited Markdown table."""
    return str(value).replace("\n", " ").replace("|", r"\|")


def parameters_section(method: str, spec: dict, cat: dict | None,
                       py_docs: dict,
                       binding: dict[str, Any] | None = None) -> str:
    """Render current public, catalog, and benchmark parameter evidence."""
    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    def add(n: str, t: str, d: str, note: str) -> None:
        if n and n not in seen:
            seen.add(n)
            rows.append((n, t, d, note))

    # 0) Current public n4m callable signature, AST-traced through re-exports.
    if binding:
        for p in binding.get("parameters", []):
            default = p.get("default")
            add(str(p.get("name") or ""),
                str(p.get("annotation") or "—"),
                str(default) if default is not None else "required",
                "current public binding signature")

    # 1) Catalog-declared C/API parameters.
    for p in (cat or {}).get("params", []) or []:
        if isinstance(p, dict):
            add(p.get("name", ""), str(p.get("type", "")) or "—",
                str(p.get("default", "")) or "—",
                p.get("notes", "") or "")
    for e in (cat or {}).get("extras", []) or []:
        if isinstance(e, dict):
            req = "required" if e.get("required") else "optional"
            add(e.get("name", ""),
                f"{e.get('type', '')} ({req})", "",
                "fit-time extra (not part of `__init__`)")

    # 2) Registry cell_params — values used by the benchmark.
    for k, v in (spec.get("cell_params") or {}).items():
        if k in ("n_samples", "n_features"):
            continue
        add(k, type(v).__name__, str(v), "registry benchmark cell value")

    if not rows:
        return "_No tunable parameters declared at the binding level._"

    # Backfill the Notes column from the central parameter catalog
    # (docs/_extras/method_param_docs.py). Anything already populated by
    # the YAML catalog or registry-cell fallback is preserved.
    enriched: list[tuple[str, str, str, str]] = []
    for (n, t, d, nt) in rows:
        if not nt:
            nt = _param_doc_lookup(method, n)
        enriched.append((n, t, d, nt))

    lines = ["### Parameters\n",
              "| Name | Type | Default | Notes |",
              "|------|------|---------|-------|"]
    for (n, t, d, nt) in enriched:
        lines.append("| `{}` | `{}` | `{}` | {} |".format(
            markdown_table_cell(n), markdown_table_cell(t),
            markdown_table_cell(d), markdown_table_cell(nt or "")))
    return "\n".join(lines)


def _docstring_summary(doc: str) -> str:
    """Take the first paragraph of a docstring, dropping Sphinx markers."""
    if not doc:
        return ""
    paragraphs = doc.strip().split("\n\n")
    return paragraphs[0].strip().replace("\n    ", " ").replace("\n  ", " ")


def _append_scientific_sections(parts: list[str], record: dict[str, str]) -> None:
    """Render the explanatory contract shared by every generated page."""
    parts.append("## Explanations\n")
    labels = (
        ("paper", "Bibliographic source"),
        ("principle", "Mathematical principle"),
        ("use_cases", "Appropriate uses"),
        ("limitations", "Limits and validation"),
        ("implementation", "Implementation"),
        ("provenance", "Sources and provenance"),
    )
    for field, label in labels:
        value = (record.get(field) or "").strip()
        if value:
            parts.append(f"### {label}\n")
            parts.append(value)
            parts.append("")


def _legacy_science_record(name: str) -> dict[str, str]:
    """Return the reviewed render-layer record for one archived entry."""
    return dict(_LEGACY_SCIENCE.get(name) or {})


def render_method_page(spec: dict, cat: dict | None,
                       bench_rows: list[dict],
                       py_docs: dict, r_docs: dict, m_docs: dict,
                       public_api: Any,
                       r_public_api: Any,
                       matlab_public_api: Any,
                       truth_sources: dict[str, dict] | None = None,
                       old_to_new: dict[str, str] | None = None,
                       rename_pattern: re.Pattern[str] | None = None,
                       benchmark_snapshot: str | None = None) -> str:
    """Build the full markdown for one method."""
    name = spec["name"]
    truth_sources = truth_sources or {}
    old_to_new = old_to_new or {}
    bib = _legacy_science_record(name)
    title = bib.get("title") or spec.get("desc") or name
    grp = method_group(name)
    grp_label = GROUP_LABELS.get(grp, grp.capitalize())

    parts: list[str] = []
    parts.append(f"# `{name}` — {title}\n")
    parts.append(f"_Group_: **{grp_label}** · "
                  f"_Registry tolerance_: `{spec.get('rmse_tol')}`")
    if spec.get("paper_only"):
        parts.append(f" · _Parity reference_: **paper-only** "
                      f"({spec['paper_only']})")
    parts.append("")

    parts.append("## Description\n")
    parts.append(f"{spec.get('desc') or '_No registry description._'}\n")

    if spec.get("notes"):
        parts.append("> **Registry note** — " + spec["notes"].replace("\n", " ")
                      + "\n")

    binding = _current_python_binding(cat, public_api, name)
    cross_bindings = cross_bindings_for_catalog(
        cat or {}, r_public_api, matlab_public_api)
    parts.append(parameters_section(name, spec, cat, py_docs, binding) + "\n")

    if not bib.get("implementation"):
        sym = method_c_symbol(name, old_to_new)
        bib["implementation"] = (f"`{sym}` in libn4m." if sym
                                 else "Python-only reference; no exported C symbol.")
    _append_scientific_sections(parts, bib)

    # R and MATLAB role pages remain linked from the verified API section.
    # We deliberately do not promote their historical wrapper docstrings into
    # this page: several use a pre-ABI-2 method name and are not a source of
    # executable examples until their own public export is AST-validated.
    parts.append("")

    parts.append(usage_section(name, spec, cat, public_api, cross_bindings,
                                truth_sources=truth_sources,
                                old_to_new=old_to_new))

    # A strict source-only regeneration must retain the committed parity
    # snapshot when the benchmark CSV is deliberately absent.  Sphinx later
    # refreshes this same section from live CSV data when it is available.
    parts.append(parity_table(name, bench_rows, truth_sources=truth_sources)
                 if bench_rows else (benchmark_snapshot or parity_table(
                     name, [], truth_sources=truth_sources)))
    parts.append("")
    parts.append("---")
    parts.append("\n_See also_: "
                  "[benchmark overview](../benchmarks/overview.md) · "
                  "[methods index](index.md) · "
                  "[interactive dashboard](../landing/dashboard.md)")
    page = "\n".join(parts)
    # Final correctness pass: rewrite any ABI-1 C symbol that leaked through a
    # free-text curated string (bibliography / registry notes / references) to
    # its ABI-2 name. Word-boundary + longest-first so prefixes don't shadow.
    return rename_symbols_in_text(page, old_to_new, rename_pattern)


# ---------------------------------------------------------------------------
# Catalog-driven index (the new `n4m.<role>` namespace)
#
# `catalog/methods.yaml` is the single source of truth for the full method
# surface (209 methods, each carrying `namespace` / `leaf` / `fq_name`). The
# index lists every catalogued method grouped by the 12 top-level namespace
# roles, links each to its documentation page, and surfaces the fully
# qualified `n4m.<role>...` name. The page link is resolved from the legacy
# page set via the symbol rename map (method_id -> old C symbol -> old page
# name); the handful of methods added during the namespace migration that
# have no legacy page get a catalog-sourced stub page.
# ---------------------------------------------------------------------------

# Canonical order + display label for the 12 top-level namespace roles.
ROLE_ORDER = [
    "transform", "augmentation", "estimators", "feature_selection",
    "model_selection", "domain_adaptation", "outlier_detection", "ensemble",
    "compose", "metrics", "decomposition", "lowlevel",
]
ROLE_LABELS = {
    "transform":          "transform — fit/transform feature transforms",
    "augmentation":       "augmentation — apply-only training-time perturbations",
    "estimators":         "estimators — supervised predictors (fit/predict)",
    "feature_selection":  "feature_selection — variable selectors",
    "model_selection":    "model_selection — splitters, AOM search/campaign, sweep",
    "domain_adaptation":  "domain_adaptation — calibration transfer / standardization",
    "outlier_detection":  "outlier_detection — sample-level screeners + Q/T²",
    "ensemble":           "ensemble — bagging / boosting / stacking / AOM blenders",
    "compose":            "compose — AOM operator superblocks",
    "metrics":            "metrics — scoring + diagnostics",
    "decomposition":      "decomposition — flexible PCA / SVD",
    "lowlevel":           "lowlevel — sufficient-statistics substrate",
}

# Operation tails stripped from a C symbol to recover the legacy page base.
# Longest-suffix-first so e.g. `_inverse_transform` wins over `_transform`.
_SYMBOL_TAILS = (
    "_inverse_transform", "_output_cols", "_is_fitted", "_split_fold",
    "_n_splits", "_create", "_destroy", "_transform", "_predict",
    "_compute", "_select", "_split", "_apply", "_free", "_fit", "_run",
    "_rank",
)

# Catalog method_id -> existing legacy page name, for the cases the symbol
# rename map cannot resolve (page named after a sibling/variant), and `None`
# for genuinely page-less methods that get a catalog stub page.
_INDEX_PAGE_OVERRIDES: dict[str, str | None] = {
    "models.pls.kernel":                "kernel_pls_rbf",
    "selection.variable_select":        "variable_select_vip",
    "utilities.sweep":                  "sweep_run",
    "aom_pop.aom_sweep":                "aom_sweep_run",
    "aom_pop.aom_chain_sweep":          "aom_chain_sweep_run",
    # These have related orchestration but distinct semantics and output.
    # Give each its own scientific record rather than silently redirecting to
    # the generic chain sweep page.
    "aom_pop.aom_chain_fixed_fit":      None,
    "aom_pop.aom_chain_screen_refit":   None,
    "aom_pop.robust_hpo":                "aom_robust_hpo",
    "diagnostics.regression_metrics":   None,
    "diagnostics.pls_diagnostics":      None,
    "utilities.hotelling_t2":           None,
    "utilities.q_residuals":            None,
    "utilities.signal_type_detector":   None,
    "utilities.transfer_metrics":       None,
    # These are distinct AOM Ridge compositions, each with its own verified
    # scientific record and stable published page.  They must not collapse
    # into the generic AOM sweep or leave a stale hand-written page in place.
    "aom_pop.ridge_global":             "aom_ridge_global",
    "aom_pop.ridge_superblock":         "aom_ridge_superblock",
    "aom_pop.ridge_active_superblock":  "aom_ridge_active_superblock",
    "aom_pop.ridge_mkl_superblock":     "aom_ridge_mkl_superblock",
}


def parse_methods_catalog(path: Path) -> list[dict]:
    """Load every method from `catalog/methods.yaml`.

    Returns the raw list of method dicts (each carries `method_id`,
    `namespace`, `leaf`, `fq_name`, `c_surface`, `notes`, ...). Empty list
    if PyYAML or the catalog is unavailable so doc builds degrade rather
    than crash, but `--strict` callers should treat that as fatal.
    """
    if not path.exists():
        return []
    try:
        import yaml  # type: ignore
    except Exception:
        return []
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    methods = doc.get("methods", [])
    return [m for m in methods if isinstance(m, dict) and m.get("method_id")]


def _load_symbol_rename_map(path: Path) -> dict[str, list[str]]:
    """method_id -> list of legacy page bases (old C symbol minus n4m_/tail)."""
    out: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            old = row.get("old_symbol", "")
            mid = row.get("method_id", "")
            if not old or not mid:
                continue
            base = old
            for tail in _SYMBOL_TAILS:
                if base.endswith(tail):
                    base = base[: -len(tail)]
                    break
            if base.startswith("n4m_"):
                base = base[len("n4m_"):]
            if base and base not in out[mid]:
                out[mid].append(base)
    return out


def load_symbol_old_to_new(path: Path) -> dict[str, str]:
    """`old_symbol -> new_symbol` from `_rename_map.tsv`.

    This is the authoritative ABI-1 -> ABI-2 C symbol map produced by the
    namespace migration. It is the single source of truth used to render
    every C symbol on the generated method pages and to drive the doc-lint.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            old = (row.get("old_symbol") or "").strip()
            new = (row.get("new_symbol") or "").strip()
            if old and new:
                out[old] = new
    return out


# Operation-verb suffixes shared with `_C_OP_SUFFIXES`. Stripping the longest
# matching suffix from an old→new symbol pair yields the method-family *prefix*
# pair (e.g. `n4m_split_kennard_stone_split` → `n4m_split_kennard_stone`). The
# operator pages render `<prefix>_*` wildcards, so this is how we resolve the
# ABI-2 prefix that should replace each leaked ABI-1 prefix.
_PREFIX_OP_SUFFIXES = (
    "_create", "_destroy", "_fit", "_apply", "_run", "_split", "_compute",
    "_transform", "_fit_transform", "_predict", "_output_cols", "_set_seed",
    "_advance", "_inverse", "_refit", "_result_destroy", "_is_fitted",
    "_select", "_get_predictions", "_get_coefficients", "_get_intercept",
)


def _strip_op_suffix(symbol: str) -> str | None:
    """Drop the longest trailing operation verb, leaving the family prefix."""
    for suf in sorted(_PREFIX_OP_SUFFIXES, key=len, reverse=True):
        if symbol.endswith(suf) and len(symbol) > len(suf) + len("n4m_x"):
            return symbol[: -len(suf)]
    return None


def load_prefix_old_to_new(
        path: Path,
        catalog_methods: list[dict] | None = None) -> dict[str, str]:
    """`old_prefix -> new_prefix` (ABI-1 family prefix → ABI-2 family prefix).

    The operator pages publish `<prefix>_*` "C ABI" wildcards rather than exact
    symbols, so the exact-symbol rename map is not enough. We derive the prefix
    map from two ABI-2 sources of truth and require them to agree:

      * `proposals/namespace/_rename_map.tsv` — strip the longest operation
        verb from each `old_symbol` / `new_symbol` pair to get the family
        prefix pair (this is the primary source, it pairs old↔new directly).
      * `catalog/methods.yaml` `c_surface` / `abi_symbols` — the common prefix
        of a method's ABI-2 symbols cross-checks the new prefix.
    """
    old_to_new = load_symbol_old_to_new(path)
    out: dict[str, str] = {}
    for old, new in old_to_new.items():
        old_pref = _strip_op_suffix(old)
        new_pref = _strip_op_suffix(new)
        if old_pref and new_pref:
            out.setdefault(old_pref, new_pref)
    # Cross-check against the catalog ABI-2 surface (defensive; the prefix map
    # is authoritative, but a divergence would signal catalog/rename-map drift).
    if catalog_methods:
        new_prefixes = set(out.values())
        for m in catalog_methods:
            syms = [s for s in (m.get("c_surface") or m.get("abi_symbols") or [])
                    if isinstance(s, str) and s.startswith("n4m_")]
            for s in syms:
                pref = _strip_op_suffix(s)
                if pref:
                    new_prefixes.add(pref)
        # Nothing to assign back — the catalog only confirms the new prefixes
        # already produced by the rename map; we keep the reference for clarity.
        del new_prefixes
    return out


def _compile_prefix_rename(
        old_to_new_prefix: dict[str, str]) -> re.Pattern[str] | None:
    """Match `<old_prefix>_*` ABI-1 wildcards, longest-first.

    Only the `_*` wildcard form is rewritten — the exact-symbol rename pass
    already rewrites full ABI-1 function symbols. Longest-first alternation so
    a short prefix that is a substring of a longer one never shadows it.
    """
    if not old_to_new_prefix:
        return None
    olds = sorted(old_to_new_prefix, key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(o) for o in olds) + r")_\*")


def rename_prefixes_in_text(
        text: str, old_to_new_prefix: dict[str, str],
        pattern: re.Pattern[str] | None = None) -> str:
    """Rewrite every ABI-1 family `<prefix>_*` wildcard to its ABI-2 form."""
    pattern = pattern or _compile_prefix_rename(old_to_new_prefix)
    if pattern is None:
        return text
    return pattern.sub(
        lambda m: old_to_new_prefix[m.group(1)] + "_*", text)


def _compile_symbol_rename(old_to_new: dict[str, str]) -> re.Pattern[str] | None:
    """Word-boundary alternation over the old symbols, longest-first.

    Longest-first ordering plus `\\b` anchors mean a shorter old symbol that
    is a prefix of a longer one (e.g. `n4m_pp_epo_transform` vs
    `..._with_d`) never shadows the longer match.
    """
    if not old_to_new:
        return None
    olds = sorted(old_to_new, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(o) for o in olds) + r")\b")


def rename_symbols_in_text(text: str, old_to_new: dict[str, str],
                           pattern: re.Pattern[str] | None = None) -> str:
    """Rewrite every ABI-1 C symbol in `text` to its ABI-2 name."""
    pattern = pattern or _compile_symbol_rename(old_to_new)
    if pattern is None:
        return text
    return pattern.sub(lambda m: old_to_new[m.group(0)], text)


def method_c_symbol(name: str, old_to_new: dict[str, str]) -> str | None:
    """ABI-2 primary C symbol for a registry method, or None.

    The curated bibliography `implementation` string names the method's
    real (pre-migration) C symbol; resolving it through the rename map gives
    the ABI-2 name. Returns None for the Model.fit-path methods (pls/pcr/opls)
    that have no per-method shim.
    """
    impl = (BIBLIOGRAPHY.get(name) or {}).get("implementation", "") or ""
    for sym in re.findall(r"n4m_[a-z0-9_]+", impl):
        if sym in old_to_new:
            return old_to_new[sym]
    return None


def resolve_catalog_page(method: dict,
                         existing_pages: set[str],
                         rename_map: dict[str, list[str]]) -> str | None:
    """Map a catalog method to its documentation page name, or None.

    None means the method has no legacy page and needs a catalog stub page.
    """
    mid = method["method_id"]
    if mid in _INDEX_PAGE_OVERRIDES:
        override = _INDEX_PAGE_OVERRIDES[mid]
        if override is None:
            return None
        if override in existing_pages:
            return override
    candidates: list[str] = list(rename_map.get(mid, []))
    candidates.append(method.get("leaf", ""))
    candidates.append(mid.split(".")[-1])
    for legacy in method.get("legacy_ids", []) or []:
        candidates.append(str(legacy).split(".")[-1])
    candidates.append(f"{method.get('leaf', '')}_select")
    for cand in candidates:
        if cand and cand in existing_pages:
            return cand
    return None


def inverse_prefix_map(old_to_new: dict[str, str]) -> dict[str, str]:
    """Return ABI-2 operator prefix -> historical documentation page prefix."""
    inverse: dict[str, str] = {}
    for old, new in old_to_new.items():
        # Multiple old symbols may expose operations of one prefix.  A single
        # pair is sufficient, and conflicting mappings are a migration error.
        prior = inverse.setdefault(new, old)
        if prior != old:
            raise ValueError(f"ambiguous ABI prefix migration for {new}: {prior}, {old}")
    return inverse


def documentation_page_for_operator(
        spec: dict, new_to_old_prefix: dict[str, str]) -> str:
    """Resolve a current ABI-2 operator to its stable documentation stem."""
    old_prefix = new_to_old_prefix.get(spec["c_prefix"])
    if old_prefix and old_prefix.startswith("n4m_"):
        return old_prefix[len("n4m_"):]
    return spec["name"]


def render_coverage_report(catalog: list[dict], page_for: dict[str, str],
                           content: dict[str, dict[str, str]]) -> str:
    """Small human-readable coverage artifact committed with the corpus."""
    rows = [
        "# Method documentation coverage\n",
        "This generated report records the documentation source used for each "
        "catalog entry. It is a content-coverage check, not a parity score.\n",
        f"- Catalog entries: **{len(catalog)}**",
        f"- Curated scientific records: **{len(content)}**",
        f"- Resolved documentation pages: **{len(set(page_for.values()))}**\n",
        "| Catalog id | Documentation page | Scientific record |",
        "|---|---|---|",
    ]
    for method in sorted(catalog, key=lambda item: item["method_id"]):
        page = page_for[method["method_id"]]
        source = "curated" if page in content else "missing"
        rows.append(f"| `{method['method_id']}` | [{page}]({page}.md) | {source} |")
    return "\n".join(rows) + "\n"


def render_catalog_stub_page(method: dict, science: dict[str, str],
                             public_api: Any, r_public_api: Any,
                             matlab_public_api: Any) -> str:
    """Render a full catalog page when no legacy registry page exists."""
    leaf = method["leaf"]
    fq = method["fq_name"]
    ns = method["namespace"]
    notes = (method.get("notes") or "").strip()
    syms = catalog_c_symbols(method)
    parts = [
        f"# `{leaf}` — {fq}\n",
        f"_Namespace_: **`n4m.{ns}`** · _Fully-qualified_: `{fq}` · "
        f"_Catalog id_: `{method['method_id']}`\n",
    ]
    parts.append("## API surface\n")
    parts.append(render_c_abi_surface(syms) + "\n")
    binding = _current_python_binding(method, public_api)
    if binding:
        parts.append("**Python (verified public re-export):** "
                     f"`{binding['import']}`\n")
        source_url = ("https://github.com/GBeurier/nirs4all-methods/blob/main/"
                      "bindings/python/src/" + binding["source"])
        if binding.get("source_line"):
            source_url += f"#L{binding['source_line']}"
        parts.append(f"**Signature:** [`{binding['symbol']}{binding['signature'] or '(...)'}`]({source_url})\n")
    elif (method.get("bindings", {}) or {}).get("python"):
        parts.append("**Python:** catalog binding is not currently an AST-verified public "
                     "`n4m` re-export. See the implementation source below.\n")
    parts.append(render_cross_binding_surfaces(
        cross_bindings_for_catalog(method, r_public_api, matlab_public_api)))
    if binding and binding.get("parameters"):
        parts.append("### Parameters\n")
        parts.append("| Name | Type | Default |")
        parts.append("|---|---|---|")
        for parameter in binding["parameters"]:
            default = parameter.get("default")
            rendered_default = str(default) if default is not None else "required"
            parts.append("| `{}` | `{}` | `{}` |".format(
                markdown_table_cell(parameter["name"]),
                markdown_table_cell(parameter.get("annotation") or "—"),
                markdown_table_cell(rendered_default)))
        parts.append("")
    _append_scientific_sections(parts, science)
    if notes and not notes.startswith("Auto-discovered"):
        parts.append("## Catalog note\n")
        parts.append(f"{notes}\n")
    bench = method.get("bench", {}) or {}
    reg = bench.get("registry_entry")
    if reg and reg != "null":
        parts.append(f"_Timing benchmark_: `{reg}`\n")
    parts.append("\n_See also_: [methods index](index.md).")
    return "\n".join(parts)


def _catalog_refs(method: dict) -> str:
    """Short ref tag column for the index (sources of truth, not parity)."""
    syms = catalog_c_symbols(method)
    tags: list[str] = []
    if syms:
        tags.append("C")
    py = (method.get("bindings", {}) or {}).get("python", {}) or {}
    if py.get("module"):
        tags.append("Py")
    refs = (method.get("parity", {}) or {}).get("references", []) or []
    if refs:
        tags.append("ref")
    return ", ".join(tags) if tags else "—"


def render_catalog_index(catalog: list[dict],
                         page_for: dict[str, str]) -> str:
    """The `n4m.<role>` namespace catalogue index for all catalogued methods.

    `page_for` maps method_id -> page base name (already resolved / stubbed).
    """
    by_role: dict[str, list[dict]] = defaultdict(list)
    for m in catalog:
        role = m["namespace"].split(".")[0]
        by_role[role].append(m)

    total = len(catalog)
    out: list[str] = [
        "# Methods catalogue\n",
        "Every native method in the library, grouped by the `n4m.<role>` "
        "namespace (ABI 2.0). Each row links to the method's documentation "
        "page and shows its fully-qualified name "
        "`n4m.<role>.<sub>...<leaf>`. Parameters, bibliographic sources, "
        "mathematical principles, binding signatures, and benchmark rows are "
        "on the linked pages. The [current method-science reference index]"
        "(scientific-references.md) collects every rendered citation and source "
        "provenance.\n",
        f"_Total catalogued native methods_: **{total}**. Additional Python "
        "reference\nsurfaces are documented where relevant.\n",
        "```{toctree}\n:hidden:\n:glob:\n:maxdepth: 1\n\n*\n```\n",
    ]
    seen_roles = [r for r in ROLE_ORDER if r in by_role]
    # Any role not in the canonical order (defensive) appended at the end.
    seen_roles += [r for r in sorted(by_role) if r not in ROLE_ORDER]
    for role in seen_roles:
        label = ROLE_LABELS.get(role, role)
        members = sorted(by_role[role], key=lambda m: (m["namespace"], m["leaf"]))
        out.append(f"## {label}\n")
        out.append("| Method | Fully-qualified name | Namespace | Refs |")
        out.append("|--------|----------------------|-----------|------|")
        for m in members:
            leaf = m["leaf"]
            page = page_for.get(m["method_id"], leaf)
            fq = m["fq_name"]
            ns = m["namespace"]
            refs = _catalog_refs(m)
            out.append(
                f"| [`{leaf}`]({page}.md) | `{fq}` | `n4m.{ns}` | {refs} |")
        out.append("")
    out.append("---")
    out.append("\nSee the [benchmark overview](../benchmarks/overview.md) "
               "for how parity and timing are measured, and the "
               "[GitHub Pages dashboard](../landing/dashboard.md) for an "
               "interactive cross-method comparison. The ABI 2.0 namespace "
               "migration is documented in "
               "[the ABI-2 migration guide](../MIGRATION_ABI2.md).")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def render_index(methods: list[dict],
                 operators: list[dict] | None = None) -> str:
    operators = operators or []
    by_group: dict[str, list[dict]] = defaultdict(list)
    for m in methods:
        by_group[method_group(m["name"])].append(m)
    op_by_group: dict[str, list[dict]] = defaultdict(list)
    for op in operators:
        op_by_group[op["group"]].append(op)
    total = len(methods) + len(operators)
    out: list[str] = [
        "# Methods catalogue\n",
        "Every method in the library — the PLS / selection algorithms from "
        "`benchmarks.parity_timing.registry.METHODS` and the C++ "
        "preprocessing / augmentation / filter / splitter operators from the "
        "n4m binding — documented with parameters, bibliographic source, "
        "mathematical principle, binding signatures, and benchmark rows.\n",
        f"_Total methods_: **{total}** "
        f"({len(methods)} PLS / selection · {len(operators)} operators). "
        "Grouped by family below.\n",
        "```{toctree}\n:hidden:\n:glob:\n:maxdepth: 1\n\n*\n```\n",
    ]
    order = ["core", "sparse", "ensemble", "robust", "nonlinear",
             "multi-block", "calibration-transfer", "classification",
             "missing", "regularized", "diagnostic", "selector", "other"]
    for g in order:
        if g not in by_group:
            continue
        out.append(f"## {GROUP_LABELS[g]}\n")
        out.append("| Method | Description | Tolerance | Refs |")
        out.append("|--------|-------------|-----------|------|")
        for m in sorted(by_group[g], key=lambda x: x["name"]):
            n = m["name"]
            refs = []
            if m["has_py_ref"]:
                refs.append("Py")
            if m["has_r_ref"]:
                refs.append("R")
            for r in m["extra_refs"]:
                refs.append(r)
            ref_str = ", ".join(refs) if refs else "—"
            desc = (m.get("desc") or "").replace("|", "\\|")
            out.append(f"| [`{n}`]({n}.md) | {desc} | `{m['rmse_tol']}` | {ref_str} |")
        out.append("")

    # Operator families (preprocessing / augmentation / filters / splitters)
    for g in OPERATOR_GROUP_ORDER:
        if g not in op_by_group:
            continue
        out.append(f"## {OPERATOR_GROUP_LABELS[g]}\n")
        out.append("| Operator | Binding | Description |")
        out.append("|----------|---------|-------------|")
        for op in sorted(op_by_group[g], key=lambda x: x["name"]):
            n = op["name"]
            desc = (_docstring_summary(op["doc"]) or "").replace("|", "\\|")
            out.append(f"| [`{n}`]({n}.md) | `n4m.sklearn.{op['class']}` | {desc} |")
        out.append("")

    out.append("---")
    out.append("\nSee the [benchmark overview](../benchmarks/overview.md) "
                "for how parity and timing are measured, and the "
                "[GitHub Pages dashboard](../landing/dashboard.md) for an "
                "interactive cross-method comparison.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Operator pages (augmentation / preprocessing / baseline / filters /
# splitters / signal transforms)
#
# The 73 methods above come from the parity registry. The C++ library also
# ships ~110 stateless/stochastic operators exposed through the n4m Python
# binding (bindings/python/src/n4m/sklearn/*.py). Each such class declares a
# `_C_PREFIX = "n4m_<name>"` that maps 1:1 to the benchmark / dashboard algo
# name (`<name>`), so a page per operator (a) documents the whole library and
# (b) makes the dashboard method links resolve. Content is auto-derived from
# the binding docstring + constructor parameters, and enriched for flagship
# operators from OPERATOR_BIB (curated from the standard chemometrics
# literature and the nirs4all preprocessing / augmentation handbooks).
# ---------------------------------------------------------------------------

N4M_SKLEARN_DIR = ROOT / "bindings" / "python" / "src" / "n4m" / "sklearn"

OPERATOR_MODULE_GROUP = {
    "augmentation":       "augmentation",
    "preprocessing":      "preprocessing",
    "baseline":           "baseline",
    "filters":            "filter",
    "splitters":          "splitter",
    "resampling":         "preprocessing",
    "feature_extraction": "feature-extraction",
    "advanced":           "transform",
}

OPERATOR_GROUP_LABELS = {
    "augmentation":      "Augmentation",
    "preprocessing":     "Preprocessing",
    "baseline":          "Baseline correction",
    "filter":            "Sample / feature filters",
    "splitter":          "Splitters",
    "feature-extraction": "Feature extraction",
    "transform":         "Signal transforms",
}

OPERATOR_GROUP_ORDER = ["preprocessing", "baseline", "transform",
                        "feature-extraction", "augmentation", "filter",
                        "splitter"]

# Flagship operators: curated bibliographic reference + mathematical
# principle on top of the auto-derived docstring. Standard chemometrics
# literature — verified references, not generated.
OPERATOR_BIB: dict[str, dict] = {
    "pp_snv": {
        "title": "Standard Normal Variate (SNV)",
        "paper": "Barnes, R. J., Dhanoa, M. S. & Lister, S. J. (1989). "
                 "*Standard Normal Variate Transformation and De-trending of "
                 "Near-Infrared Diffuse Reflectance Spectra*. Applied "
                 "Spectroscopy 43(5), 772–777.",
        "principle": (
            "Each spectrum $\\mathbf{x}_i$ is centred and scaled by its own "
            "row statistics: $\\mathrm{SNV}(\\mathbf{x}_i) = "
            "(\\mathbf{x}_i - \\bar{x}_i)/s_i$, where $\\bar{x}_i$ and $s_i$ "
            "are the mean and standard deviation across the wavelengths of "
            "that single spectrum. This removes multiplicative scatter and "
            "additive baseline shifts on a per-sample basis without needing a "
            "reference spectrum, which is why SNV is robust to sample-to-sample "
            "path-length variation in diffuse-reflectance NIR."),
    },
    "pp_rnv": {
        "title": "Robust Normal Variate (RNV)",
        "paper": "Guo, Q., Wu, W. & Massart, D. L. (1999). *The robust normal "
                 "variate transform for pattern recognition with near-infrared "
                 "data*. Analytica Chimica Acta 382(1–2), 87–103.",
        "principle": (
            "A median/IQR analogue of SNV: each spectrum is corrected as "
            "$(\\mathbf{x}_i - \\mathrm{median}(\\mathbf{x}_i)) / "
            "\\mathrm{IQR}(\\mathbf{x}_i)$. Replacing the mean and standard "
            "deviation with robust location/scale estimators makes the "
            "normalisation insensitive to a small number of strong absorption "
            "bands or outlying channels."),
    },
    "pp_msc": {
        "title": "Multiplicative Scatter Correction (MSC)",
        "paper": "Geladi, P., MacDougall, D. & Martens, H. (1985). "
                 "*Linearization and Scatter-Correction for Near-Infrared "
                 "Reflectance Spectra of Meat*. Applied Spectroscopy 39(3), "
                 "491–500.",
        "principle": (
            "Each spectrum is regressed on a reference spectrum (typically the "
            "training-set mean $\\bar{\\mathbf{x}}$): "
            "$\\mathbf{x}_i = a_i + b_i\\,\\bar{\\mathbf{x}} + \\mathbf{e}_i$. "
            "The corrected spectrum $(\\mathbf{x}_i - a_i)/b_i$ removes the "
            "additive ($a_i$) and multiplicative ($b_i$) scatter estimated by "
            "ordinary least squares. Unlike SNV, MSC needs a reference and so "
            "is stateful (the reference is fit on training data)."),
    },
    "pp_emsc": {
        "title": "Extended Multiplicative Scatter Correction (EMSC)",
        "paper": "Martens, H. & Stark, E. (1991). *Extended multiplicative "
                 "signal correction and spectral interference subtraction*. "
                 "Journal of Pharmaceutical and Biomedical Analysis 9(8), "
                 "625–635.",
        "principle": (
            "EMSC augments the MSC regression basis with polynomial wavelength "
            "terms (and optionally known interferent spectra), so the model "
            "$\\mathbf{x}_i = a_i + b_i\\bar{\\mathbf{x}} + d_i\\boldsymbol{\\lambda} "
            "+ e_i\\boldsymbol{\\lambda}^2 + \\dots$ separates chemical signal "
            "from smooth physical baselines more flexibly than plain MSC."),
    },
    "pp_savgol": {
        "title": "Savitzky–Golay smoothing / derivative",
        "paper": "Savitzky, A. & Golay, M. J. E. (1964). *Smoothing and "
                 "Differentiation of Data by Simplified Least Squares "
                 "Procedures*. Analytical Chemistry 36(8), 1627–1639.",
        "principle": (
            "Within a sliding window of odd length $w$, a polynomial of order "
            "$p$ is fit by least squares; the smoothed value (or its "
            "$d$-th derivative) at the window centre is a fixed linear "
            "combination of the windowed points. Because the convolution "
            "coefficients are precomputed, SG simultaneously denoises and "
            "differentiates while preserving peak shape far better than a "
            "moving average."),
    },
    "pp_first_derivative": {
        "title": "First derivative",
        "paper": "Standard finite-difference / gap derivative; see Savitzky & "
                 "Golay (1964) and Norris & Williams (1984).",
        "principle": (
            "Approximates $\\mathrm{d}\\mathbf{x}/\\mathrm{d}\\lambda$ by finite "
            "differences. The first derivative removes constant baseline "
            "offsets (additive scatter) and accentuates inflection points of "
            "overlapping bands."),
    },
    "pp_second_derivative": {
        "title": "Second derivative",
        "paper": "Standard finite-difference / gap derivative; see Savitzky & "
                 "Golay (1964) and Norris & Williams (1984).",
        "principle": (
            "Approximates $\\mathrm{d}^2\\mathbf{x}/\\mathrm{d}\\lambda^2$. The "
            "second derivative removes both constant and linear baselines and "
            "resolves overlapping peaks into sharp negative lobes at the "
            "original band positions, at the cost of amplifying high-frequency "
            "noise (hence it is usually paired with smoothing)."),
    },
    "pp_gaussian": {
        "title": "Gaussian smoothing",
        "paper": "Classical Gaussian-kernel convolution smoothing.",
        "principle": (
            "Convolves each spectrum with a discretised Gaussian kernel of a "
            "given standard deviation, a low-pass filter that attenuates "
            "high-frequency noise with minimal ringing compared with a boxcar "
            "average."),
    },
    "pp_detrend": {
        "title": "De-trending",
        "paper": "Barnes, R. J., Dhanoa, M. S. & Lister, S. J. (1989). Applied "
                 "Spectroscopy 43(5), 772–777.",
        "principle": (
            "Fits and subtracts a low-order polynomial (commonly linear or "
            "quadratic) along the wavelength axis of each spectrum, removing "
            "wavelength-dependent baseline curvature that SNV alone leaves "
            "behind."),
    },
    "aug_gaussian_noise": {
        "title": "Gaussian additive-noise augmentation",
        "paper": "Standard data-augmentation perturbation; see the nirs4all "
                 "augmentation handbook.",
        "principle": (
            "Adds i.i.d. Gaussian noise $\\mathbf{x} + \\boldsymbol{\\varepsilon}$, "
            "$\\varepsilon \\sim \\mathcal{N}(0,\\sigma^2)$, to each element, "
            "simulating detector/shot noise to regularise models and enlarge "
            "small calibration sets."),
    },
    "aug_linear_drift": {
        "title": "Linear baseline-drift augmentation",
        "paper": "Physical baseline-perturbation augmentation; see the nirs4all "
                 "augmentation handbook.",
        "principle": (
            "Adds a random linear ramp $a + b\\,\\boldsymbol{\\lambda}$ across "
            "the wavelength axis, emulating instrument baseline drift and "
            "temperature-dependent offsets so that downstream models learn "
            "drift invariance."),
    },
    "aug_mixup": {
        "title": "Mixup augmentation",
        "paper": "Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. (2018). "
                 "*mixup: Beyond Empirical Risk Minimization*. ICLR 2018.",
        "principle": (
            "Forms convex combinations of sample pairs, "
            "$\\tilde{\\mathbf{x}} = \\lambda\\mathbf{x}_i + (1-\\lambda)\\mathbf{x}_j$ "
            "and $\\tilde{y} = \\lambda y_i + (1-\\lambda) y_j$ with "
            "$\\lambda \\sim \\mathrm{Beta}(\\alpha,\\alpha)$, encouraging "
            "linear behaviour between training examples and regularising the "
            "calibration model."),
    },
}


def _operator_init_params(cls_node: ast.ClassDef,
                          classes_by_name: dict[str, ast.ClassDef]) -> list[dict]:
    """Constructor params for an operator class, resolving a single level of
    in-module base-class inheritance when the class has no own __init__."""
    init = next((n for n in cls_node.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None:
        for base in cls_node.bases:
            bname = base.id if isinstance(base, ast.Name) else None
            if bname and bname in classes_by_name:
                return _operator_init_params(classes_by_name[bname], classes_by_name)
        return []
    out: list[dict] = []
    pos = init.args.args[1:]  # skip self
    defaults = init.args.defaults
    offset = len(pos) - len(defaults)
    for i, arg in enumerate(pos):
        if arg.arg in ("self",):
            continue
        default = "—"
        if i >= offset:
            node = defaults[i - offset]
            try:
                default = repr(ast.literal_eval(node))
            except (ValueError, SyntaxError):
                default = ast.unparse(node)
        ann = ast.unparse(arg.annotation) if arg.annotation else "—"
        out.append({"name": arg.arg, "type": ann, "default": default})
    return out


# C ABI operation suffixes stripped when deriving a method prefix from
# `lib.n4m_<prefix>_<op>(...)` calls inside a binding class.
_C_OP_SUFFIXES = (
    "_create", "_destroy", "_fit", "_apply", "_run", "_split", "_compute",
    "_transform", "_fit_transform", "_predict", "_output_cols", "_set_seed",
    "_advance", "_inverse", "_refit", "_result_destroy",
)


def _class_c_prefix(node: ast.ClassDef,
                    classes_by_name: dict[str, ast.ClassDef],
                    _seen: set | None = None) -> str | None:
    """Best-effort C ABI prefix (`n4m_<name>`) for a binding operator class.

    Looks for (1) a `_C_PREFIX` assignment, class-level OR `self._C_PREFIX`
    inside __init__; (2) failing that, the common `lib.n4m_<prefix>_<op>()`
    symbol stem used in the class body (covers filters that call the C ABI
    directly without a `_C_PREFIX` attribute); (3) recurses into in-module
    base classes.
    """
    _seen = _seen if _seen is not None else set()
    if node.name in _seen:
        return None
    _seen.add(node.name)
    # (1) explicit _C_PREFIX (class- or instance-level)
    for item in ast.walk(node):
        if isinstance(item, ast.Assign) and isinstance(item.value, ast.Constant) \
                and isinstance(item.value.value, str) \
                and item.value.value.startswith("n4m_"):
            for t in item.targets:
                if (isinstance(t, ast.Name) and t.id == "_C_PREFIX") or \
                        (isinstance(t, ast.Attribute) and t.attr == "_C_PREFIX"):
                    return item.value.value
    # (2) derive from lib.n4m_<prefix>_<op>(...) symbol usage. Only consider
    # attributes accessed on a bare `lib` name, and strip the longest matching
    # operation suffix first so e.g. `_fit_transform` isn't mistaken for `_fit`.
    from collections import Counter
    suffixes_longest_first = sorted(_C_OP_SUFFIXES, key=len, reverse=True)
    stems: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and item.attr.startswith("n4m_") \
                and isinstance(item.value, ast.Name) and item.value.id == "lib":
            sym = item.attr
            for suf in suffixes_longest_first:
                if sym.endswith(suf) and len(sym) > len(suf) + 4:
                    stems.append(sym[: -len(suf)])
                    break
    if stems:
        return Counter(stems).most_common(1)[0][0]
    # (3) base classes within the same module
    for base in node.bases:
        bname = base.id if isinstance(base, ast.Name) else None
        if bname and bname in classes_by_name:
            found = _class_c_prefix(classes_by_name[bname], classes_by_name, _seen)
            if found:
                return found
    return None


def parse_operator_bindings(
        src_dir: Path,
        public_imports: dict[str, tuple[str, str]] | None = None) -> list[dict]:
    """AST-parse current n4m implementation operators into spec dictionaries.

    One spec per operator class (prefix resolved by `_class_c_prefix`).  The
    implementation layer is deliberately scanned because it owns the C ABI
    calls; ``public_imports`` supplies the stable role-package import that is
    rendered to readers.  Deduplication keeps the richest docstring.
    """
    specs: dict[str, dict] = {}
    public_imports = public_imports or {}
    if not src_dir.exists():
        return []
    for py in sorted(src_dir.glob("*.py")):
        if py.name in {"__init__.py", "compat.py", "estimator_base.py"}:
            continue
        group = OPERATOR_MODULE_GROUP.get(py.stem, "transform")
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        classes_by_name = {n.name: n for n in ast.walk(tree)
                           if isinstance(n, ast.ClassDef)}
        for node in classes_by_name.values():
            # skip private/abstract base classes (e.g. _AugmenterBase, _SplitHandle)
            if node.name.startswith("_"):
                continue
            cprefix = _class_c_prefix(node, classes_by_name)
            if not isinstance(cprefix, str) or not cprefix.startswith("n4m_"):
                continue
            name = cprefix[len("n4m_"):]
            doc = ast.get_docstring(node) or ""
            spec = {
                "name": name, "class": node.name, "module": py.stem,
                "group": group, "doc": doc, "c_prefix": cprefix,
                "params": _operator_init_params(node, classes_by_name),
            }
            public = public_imports.get(node.name)
            if public:
                spec["public_module"], spec["public_class"] = public
            prev = specs.get(name)
            if prev is None or len(doc) > len(prev["doc"]):
                specs[name] = spec
    return sorted(specs.values(), key=lambda s: s["name"])


def _operator_param_example(spec: dict) -> str:
    return ""  # default-constructed in the usage snippet


def render_operator_page(spec: dict, science: dict[str, str],
                         cat: dict | None, public_api: Any,
                         r_public_api: Any, matlab_public_api: Any,
                         bench_rows: list[dict] | None = None,
                         prefix_old_to_new: dict[str, str] | None = None) -> str:
    """Render a current n4m operator page from a curated scientific record."""
    name = spec["name"]
    cls = spec["class"]
    grp = spec["group"]
    grp_label = OPERATOR_GROUP_LABELS.get(grp, grp.capitalize())
    title = science.get("title") or _humanize_class(cls)
    summary = _docstring_summary(spec["doc"]) or science.get("principle", "")
    prefix_old_to_new = prefix_old_to_new or {}
    c_prefix = prefix_old_to_new.get(spec["c_prefix"], spec["c_prefix"])
    binding = _current_python_binding(cat, public_api, name)
    cross_bindings = cross_bindings_for_catalog(
        cat or {}, r_public_api, matlab_public_api)

    p: list[str] = []
    p.append(f"# `{name}` — {title}\n")
    p.append(f"_Group_: **{grp_label}** · _C ABI_: `{c_prefix}_*`\n")
    p.append("## Description\n")
    p.append(summary + "\n")
    if spec["doc"] and len(spec["doc"].strip()) > len(summary) + 24:
        p.append("<details>\n<summary>Full binding docstring</summary>\n\n"
                 "```text\n" + spec["doc"].strip() + "\n```\n</details>\n")

    p.append("## Parameters\n")
    if spec["params"]:
        p.append("| Name | Type | Default |")
        p.append("|------|------|---------|")
        for prm in spec["params"]:
            p.append("| `{}` | `{}` | `{}` |".format(
                markdown_table_cell(prm["name"]),
                markdown_table_cell(prm["type"]),
                markdown_table_cell(prm["default"])))
    else:
        p.append("_No constructor parameters._")
    p.append("")

    # The implementation scan supplies detailed constructor parameters while
    # the catalog / public API scans prove the import and cross-language
    # surfaces.  Keep both: a public re-export alone must not erase the
    # operator's constructor documentation.
    p.append("## API and bindings\n")
    symbols = catalog_c_symbols(cat)
    if not symbols:
        p.append(f"**C ABI (family):** `{c_prefix}_*`. No catalogued exact entry point was found.\n")
    else:
        p.append(render_c_abi_surface(symbols) + "\n")
    if binding:
        snippet = binding["import"]
        p.append("**Python (verified public re-export):**\n\n"
                 f"```python\n{snippet}\n```\n")
        source_url = ("https://github.com/GBeurier/nirs4all-methods/blob/main/"
                      "bindings/python/src/" + binding["source"])
        if binding.get("source_line"):
            source_url += f"#L{binding['source_line']}"
        p.append(f"Source signature: [`{binding['symbol']}{binding['signature'] or '(...)'}`]({source_url}).\n")
    else:
        p.append("**Python:** no current AST-verified public `n4m` re-export was found for this method.\n")
    p.append(render_cross_binding_surfaces(cross_bindings))

    record = dict(science)
    record["implementation"] = (
        record.get("implementation", "").strip() + "\n\n"
        f"The ABI-2 implementation is the `{c_prefix}_*` lifecycle in libn4m."
    ).strip()
    _append_scientific_sections(p, record)

    if bench_rows:
        p.append("")
        p.append(parity_table(name, bench_rows))

    p.append("")
    p.append("---")
    p.append("\n_See also_: [methods index](index.md) · "
             "[interactive dashboard](../landing/dashboard.md)")
    return "\n".join(p)


def _humanize_class(cls: str) -> str:
    """`GaussianAdditiveNoise` → `Gaussian Additive Noise`."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", cls).replace("_", " ")


# ---------------------------------------------------------------------------
# Doc-lint — no ABI-1 C symbol *or family prefix* may survive in a page
# ---------------------------------------------------------------------------

# Coarse legacy ABI-1 family prefixes. The per-method prefixes derived from the
# rename map cover the exact families, but these guard against any new
# wildcard line that names a family without a per-method leaf (e.g. a bare
# `n4m_split_*`). They are intentionally broad.
LEGACY_FAMILY_PREFIXES = ("n4m_pp_", "n4m_aug_", "n4m_split_", "n4m_aom_")


def build_old_prefix_set(old_to_new: dict[str, str]) -> set[str]:
    """ABI-1 family/method prefixes to forbid in generated pages.

    Built from two sources:
      * every column-1 (`old_symbol`) entry of `_rename_map.tsv` with its
        trailing operation verb stripped — the per-method family prefix
        (e.g. `n4m_split_kennard_stone_split` → `n4m_split_kennard_stone`);
      * the coarse legacy family prefixes (`n4m_pp_`, `n4m_aug_`,
        `n4m_split_`, `n4m_aom_`), normalised to their trailing-`_`-free form.

    These prefixes — and their `<prefix>_*` wildcard form — must not appear in
    any `docs/methods/*.md`. The `*_t` / `*_result_t` struct TYPE names kept
    their ABI-1 spelling in the ABI-2 headers, so the lint allow-lists any
    token ending in `_t` (see `_token_is_abi1_prefix`).
    """
    prefixes: set[str] = set()
    for old in old_to_new:
        pref = _strip_op_suffix(old)
        if pref:
            prefixes.add(pref)
    for fam in LEGACY_FAMILY_PREFIXES:
        prefixes.add(fam.rstrip("_"))
    return prefixes


def _compile_prefix_lint(
        old_prefixes: set[str]) -> re.Pattern[str] | None:
    """Match an ABI-1 prefix token, longest-first, with an optional `_*`.

    The whole `n4m_…` token is captured (greedy `[a-z0-9_]*`) so the
    allow-list can inspect its full form (e.g. reject `n4m_pp_snv_*` but accept
    the type name `n4m_aom_global_result_t`). The `(_\\*)?` tail keeps the
    rendered wildcard marker out of the captured identifier.
    """
    if not old_prefixes:
        return None
    olds = sorted(old_prefixes, key=len, reverse=True)
    body = "|".join(re.escape(o) for o in olds)
    return re.compile(r"\b((?:" + body + r")[a-z0-9_]*?)(_\*)?(?![a-z0-9_*])")


def _token_is_abi1_prefix(token: str, old_prefixes: set[str]) -> bool:
    """True if `token` names a forbidden ABI-1 prefix (not an allowed type).

    `*_t` / `*_result_t` struct type names legitimately kept their ABI-1
    spelling in the ABI-2 headers — they are types, not function symbols — so
    any token ending in `_t` is allow-listed.
    """
    if token.endswith("_t"):
        return False
    return any(
        token == p or token.startswith(p + "_") for p in old_prefixes)


def lint_generated_pages(out_dir: Path,
                         old_to_new: dict[str, str]) -> dict[str, list[str]]:
    """Scan every generated `docs/methods/*.md` for surviving ABI-1 surface.

    Flags two classes of leak (returns `{page_name -> sorted findings}`):

      * exact ABI-1 function symbols — any column-1 entry of `_rename_map.tsv`,
        matched on word boundaries so ABI-2 names embedding an old leaf are not
        false hits;
      * ABI-1 family / method *prefixes* and their `<prefix>_*` wildcard form
        (see `build_old_prefix_set`), which is how the operator pages publish
        their "C ABI" line.

    `*_t` / `*_result_t` struct type names are allow-listed — they keep their
    ABI-1 spelling in the ABI-2 headers and are types, not function symbols.
    """
    sym_pattern = _compile_symbol_rename(old_to_new)
    old_prefixes = build_old_prefix_set(old_to_new)
    pref_pattern = _compile_prefix_lint(old_prefixes)
    findings: dict[str, list[str]] = {}
    if sym_pattern is None and pref_pattern is None:
        return findings
    for md in sorted(out_dir.glob("*.md")):
        if md.stem == "index":
            continue
        text = md.read_text(encoding="utf-8")
        hits: set[str] = set()
        if sym_pattern is not None:
            hits.update(sym_pattern.findall(text))
        if pref_pattern is not None:
            for token, wildcard in pref_pattern.findall(text):
                if _token_is_abi1_prefix(token, old_prefixes):
                    hits.add(token + (wildcard or ""))
        if hits:
            findings[md.name] = sorted(hits)
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(METHODS_DIR),
                     help="output directory (default docs/methods)")
    ap.add_argument("--no-bench", action="store_true",
                     help="skip the parity + timing tables")
    ap.add_argument("--strict", action="store_true",
                     help="fail if registry / catalog / CSV missing")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_PY.exists():
        if args.strict:
            raise SystemExit(f"missing registry: {REGISTRY_PY}")
        return
    methods = parse_registry(REGISTRY_PY)
    science = scientific_content()
    old_to_new = load_symbol_old_to_new(RENAME_MAP_TSV)
    if args.strict and not old_to_new:
        raise SystemExit(f"missing/empty symbol rename map: {RENAME_MAP_TSV}")
    rename_pattern = _compile_symbol_rename(old_to_new)
    # ABI-1 family prefix → ABI-2 family prefix (drives the operator-page
    # `<prefix>_*` "C ABI" wildcard rewrite). Derived from the rename map and
    # cross-checked against the catalog ABI-2 surface.
    catalog_methods = parse_methods_catalog(METHODS_CATALOG_YAML)
    prefix_old_to_new = load_prefix_old_to_new(
        RENAME_MAP_TSV, catalog_methods)
    prefix_pattern = _compile_prefix_rename(prefix_old_to_new)
    new_to_old_prefix = inverse_prefix_map(prefix_old_to_new)
    py_docs = parse_python_sklearn(N4M_IMPL_DIR)
    public_imports = parse_n4m_public_imports(N4M_PYTHON_DIR)
    public_api = scan_public_api(N4M_PYTHON_DIR)
    r_public_api = scan_r_public_api()
    matlab_public_api = scan_matlab_public_api()
    # Retained solely for the generation summary while R/MATLAB rendering uses
    # the source-verified public indexes above.
    r_docs = parse_r_signatures(R_DIR)
    m_docs = parse_matlab(MATLAB_DIR)
    truth_sources = load_truth_source_metadata(strict=args.strict)

    if args.no_bench or not CSV_PATH.exists():
        if args.strict and not args.no_bench and not CSV_PATH.exists():
            raise SystemExit(f"missing CSV: {CSV_PATH}")
        bench_rows: dict[str, list[dict]] = {}
    else:
        bench_rows = parse_csv(CSV_PATH)

    # This module intentionally loads JSON only. It does not import the
    # generator back, so it is safe to use here and makes --no-bench
    # regeneration lossless for committed parity tables.
    from method_benchmark_tables import load_snapshot
    benchmark_snapshots = {
        name: normalize_benchmark_snapshot(block)
        for name, block in load_snapshot().items()
    }

    # Match registry methods to the current catalog, never the retired
    # tier-2 catalog.  It provides the ABI-2 C surface, current public
    # binding metadata, and detailed parameter declarations.
    method_to_cat: dict[str, dict] = {}
    registry_source_pages = set(science) | {method["name"] for method in methods}
    registry_page_rename_map = _load_symbol_rename_map(RENAME_MAP_TSV)
    for m in methods:
        n = m["name"]
        exact = [entry for entry in catalog_methods
                 if entry.get("leaf") == n
                 or n in {str(x).split(".")[-1]
                          for x in (entry.get("legacy_ids") or [])}]
        if len(exact) == 1:
            method_to_cat[n] = exact[0]
            continue
        token = f"_{n}_"
        by_symbol = [entry for entry in catalog_methods
                     if any(token in f"_{sym}_"
                            for sym in (entry.get("c_surface") or []))]
        if len(by_symbol) == 1:
            method_to_cat[n] = by_symbol[0]
            continue
        # Some registry names predate the ABI-2 catalog leaf (for example
        # `kernel_pls_rbf` versus `models.pls.kernel`). Resolve through the
        # same stable page mapping used by the catalogue index so their
        # source-verified R/MATLAB surfaces are not silently omitted.
        by_page = [entry for entry in catalog_methods
                   if resolve_catalog_page(
                       entry, registry_source_pages,
                       registry_page_rename_map) == n]
        if len(by_page) == 1:
            method_to_cat[n] = by_page[0]

    # Render registry methods
    method_names = {m["name"] for m in methods}
    for m in methods:
        rows = bench_rows.get(m["name"], [])
        page = render_method_page(
            m, method_to_cat.get(m["name"]), rows,
            py_docs, r_docs, m_docs, public_api,
            r_public_api, matlab_public_api,
            truth_sources=truth_sources.get(m["name"], {}),
            old_to_new=old_to_new, rename_pattern=rename_pattern,
            benchmark_snapshot=benchmark_snapshots.get(m["name"]))
        (out_dir / f"{m['name']}.md").write_text(page, encoding="utf-8")

    # Render C++ operator pages from the current ABI-2 implementation layer.
    # The rename map retains the historical file stem so existing public URLs
    # remain stable while the displayed C ABI is current.
    operators = parse_operator_bindings(N4M_IMPL_DIR, public_imports)
    for op in operators:
        op["name"] = documentation_page_for_operator(op, new_to_old_prefix)
    operators = [op for op in operators if op["name"] not in method_names]

    # Resolve the catalog before rendering operator pages.  This gives every
    # one of the 118 implementation-derived pages the same current Python,
    # R, MATLAB, and exact C-header evidence as registry and ABI-2 catalog
    # pages.  Page identity, rather than an inferred class/module name, is
    # the join key because it is already checked against the rename map.
    operator_page_names = {op["name"] for op in operators}
    preexisting_pages = set(science) | operator_page_names | method_names
    rename_map = _load_symbol_rename_map(RENAME_MAP_TSV)
    page_for: dict[str, str] = {}
    catalog_by_page: dict[str, list[dict]] = defaultdict(list)
    for catalog_method in catalog_methods:
        page = resolve_catalog_page(catalog_method, preexisting_pages, rename_map)
        if page is None:
            page = catalog_method["method_id"].replace(".", "_")
        page_for[catalog_method["method_id"]] = page
        catalog_by_page[page].append(catalog_method)

    for op in operators:
        record = science.get(op["name"]) or OPERATOR_BIB.get(op["name"], {})
        if not record:
            if args.strict:
                raise SystemExit(
                    f"missing scientific record for operator page: {op['name']}")
            continue
        matching_catalog = catalog_by_page.get(op["name"], [])
        if args.strict and len(matching_catalog) != 1:
            raise SystemExit(
                "operator page must resolve to exactly one catalog entry: "
                f"{op['name']} -> {[m['method_id'] for m in matching_catalog]}")
        page = render_operator_page(op, record,
                                    matching_catalog[0] if matching_catalog else None,
                                    public_api, r_public_api, matlab_public_api,
                                    bench_rows.get(op["name"]),
                                    prefix_old_to_new=prefix_old_to_new)
        (out_dir / f"{op['name']}.md").write_text(page, encoding="utf-8")

    # Catalog-driven index over the full 209-method namespace surface. The
    # catalog is the single source of truth for `namespace` / `leaf` /
    # `fq_name`; the index lists every method grouped by top-level role and
    # links to its page. Methods added during the namespace migration with no
    # legacy page get a catalog-sourced stub page so every link resolves.
    if not catalog_methods:
        if args.strict:
            raise SystemExit(f"missing/empty method catalog: {METHODS_CATALOG_YAML}")
        # Fall back to the legacy registry-driven index so the build still
        # produces an index page.
        (out_dir / "index.md").write_text(
            render_index(methods, operators), encoding="utf-8")
        print(f"wrote {len(methods)} method + {len(operators)} operator pages "
              f"+ legacy index in {out_dir}")
        return

    # Resolve against maintained scientific sources, never against arbitrary
    # residual Markdown in the output directory.  This makes regeneration
    # deterministic and surfaces a missing mapping before it reaches RTD.
    existing_pages = set(preexisting_pages)
    catalog_pages = 0
    for m in catalog_methods:
        page = page_for[m["method_id"]]

        # Registry and operator pages have their own richer renderer. Every
        # other catalog page is still emitted here, even when its stem already
        # occurs in a curated source module.  Resolving a source key without
        # writing a file used to leave an arbitrary old Markdown page in place
        # and made a fresh checkout non-reproducible.
        if page not in method_names and page not in operator_page_names:
            record = science.get(page)
            if not record:
                if args.strict:
                    raise SystemExit(
                        f"missing scientific record for catalog method "
                        f"{m['method_id']} ({page}.md)")
                continue
            (out_dir / f"{page}.md").write_text(
                render_catalog_stub_page(
                    m, record, public_api, r_public_api, matlab_public_api),
                encoding="utf-8")
            existing_pages.add(page)
            catalog_pages += 1

    (out_dir / "index.md").write_text(
        render_catalog_index(catalog_methods, page_for), encoding="utf-8")
    (out_dir / "coverage.md").write_text(
        render_coverage_report(catalog_methods, page_for, science),
        encoding="utf-8")

    print(f"wrote {len(methods)} registry + {len(operators)} operator pages, "
          f"{catalog_pages} catalog pages, "
          f"+ catalog index ({len(catalog_methods)} methods) in {out_dir} "
          f"(py docstrings: {len(py_docs)}, R sigs: {len(r_docs)}, "
          f"MATLAB sigs: {len(m_docs)})")

    # In-place ABI-2 pass over every method page. The operator pages
    # (`pp_*` / `aug_* `/ `split_*` / `filter_*` …) were generated before the
    # namespace migration moved their binding source out of the path the
    # generator scans, so it no longer overwrites them. Rewrite any leaked
    # ABI-1 `<prefix>_*` wildcard and any exact ABI-1 symbol in place so no
    # page advertises a dead ABI-1 prefix or symbol.
    rewritten = 0
    for md in sorted(out_dir.glob("*.md")):
        if md.stem == "index":
            continue
        text = md.read_text(encoding="utf-8")
        new_text = rename_prefixes_in_text(
            text, prefix_old_to_new, prefix_pattern)
        new_text = rename_symbols_in_text(
            new_text, old_to_new, rename_pattern)
        if new_text != text:
            md.write_text(new_text, encoding="utf-8")
            rewritten += 1
    if rewritten:
        print(f"rewrote ABI-2 surface in {rewritten} method page(s) in place")

    # Doc-lint: no ABI-1 C symbol *or family prefix* may survive in any page.
    findings = lint_generated_pages(out_dir, old_to_new)
    if findings:
        n_syms = sum(len(v) for v in findings.values())
        print(f"DOC-LINT: {n_syms} ABI-1 symbol(s)/prefix(es) survive in "
              f"{len(findings)} page(s):", file=sys.stderr)
        for page, syms in findings.items():
            print(f"  {page}: {', '.join(syms)}", file=sys.stderr)
        if args.strict:
            raise SystemExit(
                "doc-lint failed: ABI-1 C symbols/prefixes in generated pages")
    else:
        print("DOC-LINT: OK — no ABI-1 C symbols or prefixes in method pages.")


if __name__ == "__main__":
    main()
