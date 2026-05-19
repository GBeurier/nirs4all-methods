#!/usr/bin/env python3
"""Generate one markdown page per pls4all method.

Reads three sources:
  * `benchmarks/parity_timing/registry.py` (AST-parsed) — canonical method
    list with descriptions, notes, parity-reference flags, parity tolerance.
  * `bindings/_catalog/sklearn_tier2.yaml` — per-method binding catalog
    (C function name, parameters with types/defaults, extras flags).
  * `benchmarks/cross_binding/results/full_matrix.csv` — parity + timing
    cells per method and per backend.

Emits:
  * `docs/methods/<name>.md`  — one page per method (71 total).
  * `docs/methods/index.md`   — catalogue / index table.

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
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PY = ROOT / "benchmarks" / "parity_timing" / "registry.py"
CATALOG_YAML = ROOT / "bindings" / "_catalog" / "sklearn_tier2.yaml"
CSV_PATH = ROOT / "benchmarks" / "cross_binding" / "results" / "full_matrix.csv"
METHODS_DIR = ROOT / "docs" / "methods"

# Source dirs scanned for docstrings to enrich per-method pages.
PY_SKLEARN_DIR = (
    ROOT / "bindings" / "python" / "src" / "pls4all" / "sklearn")
R_DIR = ROOT / "bindings" / "r" / "pls4all" / "R"
MATLAB_DIR = ROOT / "bindings" / "matlab" / "+pls4all"

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
    from method_param_docs import lookup as _param_doc_lookup
except ImportError:  # pragma: no cover - depends on import path
    def _param_doc_lookup(method: str, param: str) -> str:
        return ""

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
            "`p4a_pls_fit` in libp4a, dispatched through "
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
            "`Algorithm.PCR` + `Solver.SVD` in libp4a, sharing the same "
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
            "`p4a_sparse_simpls_fit` — MethodResult entry point; "
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
            "`p4a_di_pls_fit` — needs `X_target` at fit time. Reference: "
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
            "`p4a_recursive_pls_run`. References: sklearn rolling-window "
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
            "`p4a_cppls_fit`. NOTE: R `pls::cppls` is **Liland 2009 "
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
            "`p4a_weighted_pls_fit` — in-sample only (no global coef "
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
            "`p4a_robust_pls_fit`. Reference: R `chemometrics::prm`."),
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
        "implementation": "`p4a_ridge_pls_fit` (in-sample only).",
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
        "implementation": "`p4a_continuum_regression_fit`.",
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
            "`p4a_n_pls_fit` — takes a flattened X plus `mode_j` and "
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
            "`p4a_kernel_pls_fit`. Predict-on-new-X is currently "
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
            "`p4a_o2pls_fit`. Reference: CRAN `OmicsPLS` 2.1.0."),
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
        "implementation": "`p4a_approximate_press_compute`.",
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
        "implementation": "`p4a_pls_diagnostics_compute` with stat=\"t2\".",
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
        "implementation": "`p4a_pls_diagnostics_compute` with stat=\"q\".",
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
        "implementation": "`p4a_pls_monitoring_run`.",
    },
    "one_se_rule": {
        "group": "diagnostic",
        "title": "One-SE rule for component count",
        "paper": "Hastie, T., Tibshirani, R. & Friedman, J. (2009). "
                 "*The Elements of Statistical Learning*, 2nd ed., §7.10.",
        "principle": (
            "Pick the smallest `k` whose CV-RMSE is within one "
            "standard error of the minimum. Reduces over-fitting."),
        "implementation": "`p4a_one_se_rule_compute`.",
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
            "`p4a_so_pls_fit`. Reference: R `multiblock` package."),
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
            "`p4a_on_pls_fit`. The vendored `OnPLS` Python port is "
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
            "`p4a_rosa_fit`. Reference: R `multiblock`."),
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
        "implementation": "`p4a_vissa_select`.",
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
            "`p4a_pso_select`. Reference: Python `pyswarms`."),
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
        "implementation": "`p4a_gpr_pls_fit`.",
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
            "`p4a_bagging_pls_fit`. Reference: R `enpls`."),
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
        "implementation": "`p4a_boosting_pls_fit`.",
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
        "implementation": "`p4a_random_subspace_pls_fit`.",
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
        "implementation": "`p4a_pls_glm_fit`.",
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
        "implementation": "`p4a_pls_qda_fit`.",
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
            "`p4a_pls_cox_fit`. Reference: CRAN `plsRcox`."),
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
            "`p4a_pds_fit` — TransformerMixin in tier 2. Reference: "
            "R `prospectr::pds`."),
    },
    "ds": {
        "group": "calibration-transfer",
        "title": "Direct Standardisation",
        "paper": "Wang, Y. et al. (1991), as above.",
        "principle": (
            "Single global transfer matrix between two instruments. "
            "Simpler than PDS, sometimes higher variance."),
        "implementation": "`p4a_ds_fit`. Reference: R `chemometrics::stdize`.",
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
        "implementation": "`p4a_mir_pls_fit`.",
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
            "`p4a_missing_aware_nipals_fit`. Reference: R `softImpute` "
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
            "`p4a_sparse_pls_da_fit`. Reference: Bioconductor `mixOmics::splsda`."),
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
            "`p4a_group_sparse_pls_fit`. Reference: CRAN `sgPLS`."),
    },
    "fused_sparse_pls": {
        "group": "sparse",
        "title": "Fused-sparse PLS",
        "paper": "Tibshirani, R. et al. (2005). *Sparsity and "
                 "smoothness via the fused lasso*. JRSS B 67(1), 91–108.",
        "principle": (
            "L1 + fused-L1 penalty on PLS loadings — neighbouring "
            "features (along the wavelength axis) tend to share weights."),
        "implementation": "`p4a_fused_sparse_pls_fit`.",
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
        "implementation": "`p4a_pls_diagnostics_compute` with stat=\"dmodx\".",
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
            "`p4a_mb_pls_fit`. Reference: sanctioned git-pinned port "
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
            "`p4a_lw_pls_fit`. Reference: sanctioned git-pinned port "
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
        "implementation": "`p4a_pls_lda_fit`. Reference: composite (PLSRegression + LDA).",
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
        "implementation": "`p4a_pls_logistic_fit`. Reference: R `plsRglm`.",
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
        "implementation": "`p4a_aom_preprocess_fit`.",
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
        "implementation": "`p4a_aom_global_select`.",
    },
    "pop_pls": {
        "group": "adaptive",
        "title": "POP-PLS",
        "paper": "Beurier et al. (2026), POP-PLS ablation.",
        "principle": (
            "Per-component adaptive operator selection (nirs4all "
            "POPPLSRegressor) over the compact "
            "strict-linear nirs4all bank, followed by best-prefix selection."),
        "implementation": "`p4a_aom_per_component_select`.",
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
        "implementation": "`p4a_variable_select_rank` with metric=VIP.",
    },
    "variable_select_coef": {
        "group": "selector",
        "title": "Coefficient-magnitude selection",
        "paper": "Standard chemometrics convention; see Martens & Næs "
                 "(1989).",
        "principle": (
            "Rank features by the magnitude of their PLS regression "
            "coefficient in the original feature scale."),
        "implementation": "`p4a_variable_select_rank` with metric=COEF.",
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
        "implementation": "`p4a_variable_select_rank` with metric=SR.",
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
        "implementation": "`p4a_interval_select`. Reference: R `plsVarSel`.",
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
        "implementation": "`p4a_bipls_select`. Reference: R `plsVarSel`.",
    },
    "sipls_select": {
        "group": "selector",
        "title": "Synergy interval PLS (siPLS)",
        "paper": "Nørgaard, L. et al. (2000), as for `interval_select`.",
        "principle": (
            "Exhaustively score every combination of `m` fixed-size "
            "intervals; pick the combination with the lowest CV-RMSE."),
        "implementation": "`p4a_sipls_select`. Reference: R `plsVarSel`.",
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
        "implementation": "`p4a_stability_select`. Reference: R `plsVarSel`.",
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
        "implementation": "`p4a_uve_select`. Reference: R `plsVarSel`.",
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
        "implementation": "`p4a_spa_select`.",
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
        "implementation": "`p4a_cars_select`. Reference: R `enpls`.",
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
        "implementation": "`p4a_random_frog_select`.",
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
        "implementation": "`p4a_scars_select`.",
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
        "implementation": "`p4a_ga_select`.",
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
        "implementation": "`p4a_shaving_select`.",
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
        "implementation": "`p4a_bve_select`.",
    },
    "t2_select": {
        "group": "selector",
        "title": "Hotelling T² loading selection",
        "paper": "Wold, S. et al. (2001), as for `variable_select_vip`.",
        "principle": (
            "Compute Hotelling T² on PLS loading weights, threshold by "
            "an α-specific upper control limit, fall back to top-k."),
        "implementation": "`p4a_t2_select`.",
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
        "implementation": "`p4a_wvc_select`.",
    },
    "wvc_threshold_select": {
        "group": "selector",
        "title": "WVC-threshold selection",
        "paper": "Andries & Vander Heyden (2011), as above.",
        "principle": (
            "Fixed-threshold and factor-of-mean rules over WVC scores; "
            "minimum-selected fallback."),
        "implementation": "`p4a_wvc_threshold_select`.",
    },
    "emcuve_select": {
        "group": "selector",
        "title": "EMCUVE — Ensemble MC-UVE",
        "paper": "Cai, W. et al. (2008), as for `stability_select`.",
        "principle": (
            "Ensemble MC-UVE rounds with a deterministic vote rule; "
            "robust against single-bag instability."),
        "implementation": "`p4a_emcuve_select`.",
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
        "implementation": "`p4a_randomization_select`.",
    },
    "rep_select": {
        "group": "selector",
        "title": "REP — Recursive Elimination of Predictors",
        "paper": "Mehmood, T. et al. (2012), as for `shaving_select`.",
        "principle": (
            "Remove a fixed count of weak coefficient-score variables "
            "per recursive step; keep the lowest-CV-error subset."),
        "implementation": "`p4a_rep_select`.",
    },
    "ipw_select": {
        "group": "selector",
        "title": "IPW — Iterative Predictor Weighting",
        "paper": "Forina, M. et al. (2004), as for `bve_select`.",
        "principle": (
            "Iteratively reweight coefficient scores; expose score "
            "and weight paths; keep the lowest-CV-error top-k subset."),
        "implementation": "`p4a_ipw_select`.",
    },
    "st_select": {
        "group": "selector",
        "title": "ST-PLS — Score-threshold selection",
        "paper": "Mehmood, T. et al. (2012), as above.",
        "principle": (
            "Apply deterministic score thresholds with min-selected "
            "fallbacks; keep the lowest-CV-error subset."),
        "implementation": "`p4a_st_select`.",
    },
    "ecr": {
        "group": "calibration-transfer",
        "title": "ECR — Empirical Calibration-transfer Regression",
        "paper": "Custom kernel for empirical calibration transfer; "
                 "see registry notes.",
        "principle": (
            "Penalised regression matching empirical calibration "
            "between two instruments, balanced by an α coefficient."),
        "implementation": "`p4a_ecr_fit`.",
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
        "implementation": "`p4a_iriv_select`.",
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
        "implementation": "`p4a_irf_select`.",
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
        "implementation": "`p4a_vip_spa_select`.",
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
        reason = (row.get("reason") or "").lower()
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
    try:
        d = float(row.get("binding_parity_max_diff",
                          row.get("parity_max_diff", "nan")))
    except (TypeError, ValueError):
        return "drift"
    if d != d:
        return "drift"
    return "drift" if d < 1e-3 else "divergent"


def parity_metric(row: dict, cid: str) -> tuple[str, str]:
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
                    n = int(r["n"]); p = int(r["p"]); t = int(r["threads"])
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
    """Return {ClassName: {docstring, init_params}} extracted from every
    `_*.py` file in `bindings/python/src/pls4all/sklearn/`."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for py in sorted(directory.glob("_*.py")):
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
                    n_args = len(args.args)
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
# Registry method → binding entry-point lookup tables
# ---------------------------------------------------------------------------

# Registry method name → Python sklearn class name in `pls4all.sklearn`.
METHOD_TO_PY_SKLEARN: dict[str, str] = {
    "pls": "PLSRegression",
    "pcr": "PCR",
    "opls": "OPLSRegression",
    "sparse_simpls": "SparseSimplsRegression",
    "di_pls": "DIPLSRegression",
    "cppls": "CPPLSRegression",
    "weighted_pls": "WeightedPLSRegression",
    "robust_pls": "RobustPLSRegression",
    "ridge_pls": "RidgePLSRegression",
    "continuum_regression": "ContinuumRegression",
    "n_pls": "NPLSRegression",
    "kernel_pls_rbf": "KernelPLSRegression",
    "o2pls": "O2PLSRegression",
    "recursive_pls": "RecursivePLSRegression",
    "mb_pls": "MBPLSRegression",
    "lw_pls": "LWPLSRegression",
    "mir_pls": "MIRPLSRegression",
    "missing_aware_nipals": "MissingAwareNipalsRegression",
    "ecr": "ECRegression",
    "bagging_pls": "BaggingPLSRegression",
    "boosting_pls": "BoostingPLSRegression",
    "random_subspace_pls": "RandomSubspacePLSRegression",
    "gpr_pls": "GPRPLSRegression",
    "so_pls": "SOPLSRegression",
    "rosa": "ROSARegression",
    "group_sparse_pls": "GroupSparsePLSRegression",
    "fused_sparse_pls": "FusedSparsePLSRegression",
    "pls_glm": "PLSGLMRegressor",
    "pls_cox": "PLSCoxRegressor",
    "pls_lda": "PLSLDAClassifier",
    "pls_qda": "PLSQDAClassifier",
    "pls_logistic": "PLSLogisticClassifier",
    "sparse_pls_da": "SparsePLSDAClassifier",
    "pds": "PDSTransformer",
    "ds": "DSTransformer",
    # Selectors
    "variable_select_vip": "VIPSelector",
    "variable_select_coef": "CoefficientSelector",
    "variable_select_sr": "SelectivityRatioSelector",
    "spa_select": "SPASelector",
    "stability_select": "StabilitySelector",
    "uve_select": "UVESelector",
    "cars_select": "CARSSelector",
    "random_frog_select": "RandomFrogSelector",
    "scars_select": "SCARSSelector",
    "ga_select": "GASelector",
    "pso_select": "PSOSelector",
    "vissa_select": "VISSASelector",
    "shaving_select": "ShavingSelector",
    "bve_select": "BVESelector",
    "rep_select": "REPSelector",
    "ipw_select": "IPWSelector",
    "st_select": "STSelector",
    "interval_select": "IntervalSelector",
    "bipls_select": "BiPLSSelector",
    "sipls_select": "SiPLSSelector",
    "t2_select": "T2Selector",
    "wvc_select": "WVCSelector",
    "wvc_threshold_select": "WVCThresholdSelector",
    "emcuve_select": "EMCUVESelector",
    "randomization_select": "RandomizationSelector",
    "iriv_select": "IRIVSelector",
    "irf_select": "IRFSelector",
    "vip_spa_select": "VIPSPASelector",
}

# Diagnostics / module-level helpers: registry name → Python function name
# exposed by `pls4all.sklearn._diagnostics` (re-exported in
# `pls4all.sklearn.__all__`).
METHOD_TO_PY_FN: dict[str, str] = {
    "pls_diagnostic_t2":    "t2_score",
    "pls_diagnostic_q":     "q_score",
    "pls_diagnostic_dmodx": "dmodx_score",
    "approximate_press":    "approximate_press",
    "one_se_rule":          "one_se_rule",
    "pls_monitoring":       "pls_monitoring",
    "aom_preprocess":       "aom_preprocess",
    "on_pls":               "on_pls",
}

# Registry name → R tier-1 raw function (in methods.R / methods_extra.R
# / selectors.R / diagnostics.R). The dispatcher
# `pls4all_method(algo, X, Y, k, params=list(…))` covers all fits +
# selectors + diagnostics; the per-algo helpers below are the idiomatic
# tier-1 shortcuts. `pls`, `pcr`, `opls` deliberately have no raw helper
# — they are reached through the dispatcher (algo strings
# `"pls_nipals"`, `"pcr"`, `"opls_nipals"`) or their formula wrappers.
METHOD_TO_R_RAW: dict[str, str] = {
    # methods.R / methods_extra.R — *_fit family
    "sparse_simpls": "sparse_simpls_fit",
    "cppls": "cppls_fit",
    "weighted_pls": "weighted_pls_fit",
    "mb_pls": "mb_pls_fit",
    "pls_glm": "pls_glm_fit",
    "mir_pls": "mir_pls_fit",
    "ecr": "ecr_fit",
    "di_pls": "di_pls_fit",
    "robust_pls": "robust_pls_fit",
    "ridge_pls": "ridge_pls_fit",
    "continuum_regression": "continuum_regression_fit",
    "recursive_pls": "recursive_pls_fit",
    "n_pls": "n_pls_fit",
    "kernel_pls_rbf": "kernel_pls_fit",
    "o2pls": "o2pls_fit",
    "sparse_pls_da": "sparse_pls_da_fit",
    "group_sparse_pls": "group_sparse_pls_fit",
    "fused_sparse_pls": "fused_sparse_pls_fit",
    "so_pls": "so_pls_fit",
    "on_pls": "on_pls_fit",
    "rosa": "rosa_fit",
    "bagging_pls": "bagging_pls_fit",
    "boosting_pls": "boosting_pls_fit",
    "random_subspace_pls": "random_subspace_pls_fit",
    "gpr_pls": "gpr_pls_fit",
    "pls_qda": "pls_qda_fit",
    "pls_cox": "pls_cox_fit",
    "pds": "pds_fit",
    "ds": "ds_fit",
    "missing_aware_nipals": "missing_aware_nipals_fit",
    "lw_pls": "lw_pls_fit",
    "pls_lda": "pls_lda_fit",
    "pls_logistic": "pls_logistic_fit",
    "aom_preprocess": "aom_preprocess",
    "aom_pls": "aom_pls",
    "pop_pls": "pop_pls",
    "aom_preprocess": "aom_preprocess",
    # selectors.R / methods_extra.R — *_select family
    "variable_select_vip":  "vip_select",
    "variable_select_coef": "coefficient_select",
    "variable_select_sr":   "selectivity_ratio_select",
    "spa_select":          "spa_select",
    "cars_select":         "cars_select",
    "stability_select":    "stability_select",
    "uve_select":          "uve_select",
    "interval_select":     "interval_select",
    "random_frog_select":  "random_frog_select",
    "scars_select":        "scars_select",
    "ga_select":           "ga_select",
    "pso_select":          "pso_select",
    "vissa_select":        "vissa_select",
    "shaving_select":      "shaving_select",
    "bve_select":          "bve_select",
    "wvc_select":          "wvc_select",
    "wvc_threshold_select":"wvc_threshold_select",
    "emcuve_select":       "emcuve_select",
    "randomization_select":"randomization_select",
    "bipls_select":        "bipls_select",
    "sipls_select":        "sipls_select",
    "rep_select":          "rep_select",
    "ipw_select":          "ipw_select",
    "st_select":           "st_select",
    "iriv_select":         "iriv_select",
    "irf_select":          "irf_select",
    "vip_spa_select":      "vip_spa_select",
    "t2_select":           "t2_select",
    # diagnostics.R
    "approximate_press":    "approximate_press",
    "one_se_rule":          "one_se_rule",
    "pls_monitoring":       "pls_monitoring",
    "pls_diagnostic_t2":    "pls_diagnostics",
    "pls_diagnostic_q":     "pls_diagnostics",
    "pls_diagnostic_dmodx": "pls_diagnostics",
}

# Parsnip / mlr3 integrations were factored out into
# `bindings/r/archive/parsnip-mlr3/` and are NOT shipped with the CRAN
# pls4all R package. The doc generator no longer renders a parsnip / mlr3
# tab; leaving the variable here (empty) keeps the rendering code path
# explicit.
PARSNIP_MLR3_ALGOS: set[str] = set()

# Registry name → R tier-2 formula function (sklearn.R, sklearn_methods.R,
# sklearn_extra.R). These are the idiomatic sklearn-style formula wrappers
# (`<fn>(formula, data, ncomp, ...)`).
METHOD_TO_R_FORMULA: dict[str, str] = {
    "pls": "pls",
    "opls": "opls",
    "sparse_simpls": "sparse_pls",
    "cppls": "cppls",
    "di_pls": "di_pls",
    "weighted_pls": "weighted_pls",
    "mb_pls": "mb_pls",
    "pls_glm": "pls_glm",
    "mir_pls": "mir_pls",
    "ecr": "ecr",
    "robust_pls": "robust_pls",
    "ridge_pls": "ridge_pls",
    "continuum_regression": "continuum_regression",
    "recursive_pls": "recursive_pls",
    "bagging_pls": "bagging_pls",
    "boosting_pls": "boosting_pls",
    "random_subspace_pls": "random_subspace_pls",
    "o2pls": "o2pls",
    "missing_aware_nipals": "missing_aware_nipals",
}

# Registry name → CRAN-`pls`-package-compatible alias in
# `bindings/r/pls4all/R/pls_compat.R`. These mirror the `pls::plsr` /
# `pls::pcr` / `pls::mvr` signatures so existing chemometrics code that
# uses the CRAN `pls` package can swap pls4all in with no rewrite.
METHOD_TO_R_PLS_COMPAT: dict[str, str] = {
    "pls": "plsr",
    "pcr": "pcr",
}

# Registry name → `mdatools::pls(x, y, ...)`-compatible alias in
# `bindings/r/pls4all/R/mdatools_compat.R`. A single function
# `pls_mdatools(x, y, ncomp, method = ...)` covers PLS / PCR / CPPLS via
# its `method=` switch (mirroring the mdatools matrix-oriented API).
METHOD_TO_R_MDATOOLS: dict[str, tuple[str, str]] = {
    # method → (fn_name, method-arg string for the snippet)
    "pls": ("pls_mdatools", "simpls"),
    "pcr": ("pls_mdatools", "pcr"),
    "cppls": ("pls_mdatools", "cppls"),
}

# Registry method → libp4a C ABI function (the `p4a_<fn>` symbol exposed
# as `pls4all._methods.<fn>` on the Python side). Most methods follow the
# convention `<method>_fit`; the entries below cover the exceptions
# (variable-rank shared kernel, diagnostic / monitoring helpers,
# `_run` / `_compute` suffixes, and Model-based methods that have no
# per-method shim — they go through `Model.fit`).
METHOD_TO_C_FUNCTION: dict[str, str] = {
    # Variable-rank selectors share one C kernel.
    "variable_select_vip":  "variable_select_rank",
    "variable_select_coef": "variable_select_rank",
    "variable_select_sr":   "variable_select_rank",
    # Diagnostics share one C kernel.
    "pls_diagnostic_t2":    "pls_diagnostics_compute",
    "pls_diagnostic_q":     "pls_diagnostics_compute",
    "pls_diagnostic_dmodx": "pls_diagnostics_compute",
    # `_run` / `_compute` exceptions.
    "approximate_press":    "approximate_press_compute",
    "one_se_rule":          "one_se_rule_compute",
    "pls_monitoring":       "pls_monitoring_run",
    "recursive_pls":        "recursive_pls_run",
}
# Methods that have no tier-1 `_methods.py` shim — they reach the C core
# through `Model.fit` (Algorithm / Solver / Deflation on Config). Used by
# `usage_section` to pick the C / Python snippet variant.
MODEL_FIT_METHODS: set[str] = {"pls", "pcr", "opls"}


# Registry name → MATLAB tier-2 classdef name.
METHOD_TO_MATLAB_CLASS: dict[str, str] = {
    "pls": "Regression",
    "pcr": "PcrRegression",
    "opls": "OplsRegression",
    "cppls": "CpplsRegression",
    "sparse_simpls": "SparsePlsRegression",
    "weighted_pls": "WeightedPlsRegression",
    "mb_pls": "MbPlsRegression",
    "ecr": "EcrRegression",
    "di_pls": "DiPlsRegression",
    "robust_pls": "RobustPlsRegression",
    "ridge_pls": "RidgePlsRegression",
    "continuum_regression": "ContinuumRegression",
    "n_pls": "NPlsRegression",
    "kernel_pls_rbf": "KernelPlsRegression",
    "o2pls": "O2plsRegression",
    "bagging_pls": "BaggingPlsRegression",
    "boosting_pls": "BoostingPlsRegression",
    "mir_pls": "MirRegression",
    "missing_aware_nipals": "MissingAwareNipalsRegression",
    "recursive_pls": "RecursivePlsRegression",
    "pls_glm": "GlmRegression",
}

# Registry name → MATLAB tier-1 function (.m file, snake_case). Aliases
# below cover the cases where the file name and the registry method name
# differ (PLS → pls_fit.m, RBF kernel PLS → kernel_pls.m, the three
# diagnostics share pls_diagnostics.m, and the variable-rank selectors
# use distinct file names).
METHOD_TO_MATLAB_FN: dict[str, str] = {
    name: name for name in (
        "aom_preprocess", "aom_pls", "approximate_press", "bagging_pls",
        "bipls_select", "boosting_pls", "bve_select", "cars_select",
        "continuum_regression", "cppls", "di_pls", "ds", "ecr",
        "emcuve_select", "fused_sparse_pls", "ga_select", "gpr_pls",
        "group_sparse_pls", "interval_select", "ipw_select", "irf_select",
        "iriv_select", "lw_pls", "mb_pls", "mir_pls",
        "missing_aware_nipals", "n_pls", "o2pls", "on_pls", "one_se_rule",
        "opls", "pcr", "pds", "pls_cox", "pls_glm", "pls_lda",
        "pls_logistic", "pls_monitoring", "pls_qda", "pso_select",
        "pop_pls", "random_frog_select", "random_subspace_pls",
        "randomization_select", "recursive_pls", "rep_select",
        "ridge_pls", "robust_pls", "rosa", "scars_select",
        "shaving_select", "sipls_select", "so_pls", "spa_select",
        "sparse_pls_da", "sparse_simpls", "st_select", "stability_select",
        "t2_select", "uve_select", "vip_spa_select", "vissa_select",
        "weighted_pls", "wvc_select", "wvc_threshold_select",
    )
}
METHOD_TO_MATLAB_FN["kernel_pls_rbf"] = "kernel_pls"
METHOD_TO_MATLAB_FN["pls"] = "pls_fit"
METHOD_TO_MATLAB_FN["pls_diagnostic_t2"] = "pls_diagnostics"
METHOD_TO_MATLAB_FN["pls_diagnostic_q"] = "pls_diagnostics"
METHOD_TO_MATLAB_FN["pls_diagnostic_dmodx"] = "pls_diagnostics"
METHOD_TO_MATLAB_FN["variable_select_vip"] = "vip_select"
METHOD_TO_MATLAB_FN["variable_select_coef"] = "coefficient_select"
METHOD_TO_MATLAB_FN["variable_select_sr"] = "selectivity_ratio_select"


# ---------------------------------------------------------------------------
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
# Rendered in this order from top to bottom; the C++ native libp4a row
# is always first, then the pls4all language bindings, then external
# reference libraries.
# ---------------------------------------------------------------------------

BAND_ORDER = [
    ("cpp",            "C++ native · libp4a"),
    ("python-pls4all", "Python · pls4all"),
    ("r-pls4all",      "R · pls4all"),
    ("matlab-pls4all", "MATLAB · pls4all"),
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
            v = abs(float(d)) if d else 0.0
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        if v > worst_abs:
            worst_abs = v
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
            n = int(r["n"]); p = int(r["p"]); t = int(r["threads"])
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
        "Adaptive wall-clock per cell measured against "
        "[`full_matrix.csv`](../benchmarks/overview.md). "
        "Only backends that implement this method are listed; "
        "libraries without the method are omitted.",
        "",
        "**Verdict** &nbsp;·&nbsp; ✓ ref / ≈ ref / ~ shape mark a "
        "reference-gate pass at strict / relaxed / qualitative "
        "tolerance &nbsp;·&nbsp; ✓ bind = pls4all binding agrees "
        "with the C++ baseline &nbsp;·&nbsp; ✗ divergent "
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
                f"drift between pls4all and the external reference "
                f"(`rmse_rel_tol ≤ {method_tol:.0e}`).",
            ]
        elif method_quality == "qualitative":
            legend_lines += [
                "",
                f"**Reference gate**: qualitative — shape/smoke "
                f"comparison only. The external library and pls4all "
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
            "C++ and external rows show reference parity; pls4all "
            "language bindings show binding parity against the C++ "
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
                if v not in ("exact", "drift"):
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
        # REAL outcome — exact/drift/divergent/error. Cells whose
        # verdict is `not_available` (library doesn't ship the method)
        # or `not_run` (size deferred / timeout / not scheduled) are
        # absence-of-data, not evidence of presence.
        _PRESENCE_VERDICTS = {"exact", "drift", "divergent", "error"}
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


def _tab_open(label: str, sync: str, cls: str = "") -> str:
    """Open a sphinx-design tab-item with sync key + optional class."""
    line = f":::{{tab-item}} {label}\n:sync: {sync}"
    if cls:
        line += f"\n:class-label: lang-{cls}"
    return line + "\n"


def _tab_close() -> str:
    return ":::\n"


def usage_section(method: str, spec: dict, cat: dict | None,
                   py_docs: dict, r_docs: dict, m_docs: dict,
                   truth_sources: dict[str, dict] | None = None) -> str:
    """Build the multi-binding usage section. Pulls real signatures from
    the parsed Python sklearn classes, R roxygen blocks and MATLAB
    headers when available; falls back to a generic template otherwise.

    `truth_sources` is `{cid -> metadata}` from
    `benchmarks.parity_timing.registry.truth_source_metadata_for(method)`.
    The bottom "Registry parity references" card is rendered from this
    metadata when available — covering the resolved library, version and
    quality band for every reference that actually instantiates on the
    doc-build host. Falls back to the legacy boolean summary otherwise.
    """
    truth_sources = truth_sources or {}
    name = method
    nc = spec.get("cell_params", {}).get("n_components", 2)
    needs_xt = spec.get("needs_x_target")
    needs_sw = spec.get("needs_sample_weights")
    needs_y = spec.get("needs_labels")
    needs_g = spec.get("needs_group_assignment")

    # ---- native (libp4a C ABI) ----
    # Model-based methods (`pls`, `pcr`, `opls`) reach the C core through
    # `p4a_model_fit`, not a `p4a_<method>_fit` shim. Detect that and emit
    # the Model.fit path for both the C and Python snippets. For every
    # other method, prefer the explicit METHOD_TO_C_FUNCTION mapping; fall
    # back to the catalog's `c_function`; final fallback is
    # `<method>_fit` (which is correct for the 60+ "regular" fits).
    cat_c_function = (cat or {}).get("c_function")
    cat_algorithm = (cat or {}).get("algorithm")
    cat_solver = (cat or {}).get("solver")
    use_model_path = name in MODEL_FIT_METHODS
    c_fn = (METHOD_TO_C_FUNCTION.get(name)
            or cat_c_function
            or f"{name}_fit")

    if name in {"aom_pls", "pop_pls"}:
        selector_fn = ("p4a_aom_per_component_select"
                       if name == "pop_pls" else "p4a_aom_global_select")
        result_t = ("p4a_aom_per_component_result_t"
                    if name == "pop_pls" else "p4a_aom_global_result_t")
        destroy_fn = ("p4a_aom_per_component_result_destroy"
                      if name == "pop_pls" else "p4a_aom_global_result_destroy")
        native = textwrap.dedent(f"""\
            ```c
            /* C ABI — libp4a AOM/POP selector path */
            p4a_context_t* ctx = p4a_context_create();
            p4a_config_t*  cfg = p4a_config_create();
            p4a_operator_bank_t* bank = NULL;
            p4a_validation_plan_t* plan = NULL;
            {result_t}* res = NULL;
            p4a_operator_bank_create(&bank);
            /* add compact nirs4all-style operators: identity, SG, detrend, FD */
            p4a_validation_plan_create(&plan);
            /* fill CV folds on plan */
            {selector_fn}(ctx, cfg, bank, &x_view, &y_view, plan,
                          /* max_components */ {nc}, &res);
            /* read predictions and selection diagnostics via result getters */
            {destroy_fn}(res);
            p4a_validation_plan_destroy(plan);
            p4a_operator_bank_destroy(bank);
            p4a_config_destroy(cfg);
            p4a_context_destroy(ctx);
            ```""")
    elif use_model_path:
        # All three MODEL_FIT_METHODS (pls / pcr / opls) have a YAML
        # catalog entry exposing the right algorithm / solver enum values.
        algo_enum = cat_algorithm or "PLS_REGRESSION"
        solver_enum = cat_solver or "SIMPLS"
        native = textwrap.dedent(f"""\
            ```c
            /* C ABI — libp4a (Model.fit path) */
            p4a_context_t* ctx = p4a_context_create();
            p4a_config_t*  cfg = p4a_config_create();
            p4a_config_set_algorithm(cfg, P4A_ALGORITHM_{algo_enum});
            p4a_config_set_solver   (cfg, P4A_SOLVER_{solver_enum});
            p4a_config_set_n_components(cfg, {nc});
            p4a_model_t* mdl = NULL;
            p4a_model_fit(ctx, cfg, &x_view, &y_view, &mdl);
            p4a_model_predict(ctx, mdl, &x_test_view, &y_hat_view);
            p4a_model_destroy(mdl);
            p4a_config_destroy(cfg);
            p4a_context_destroy(ctx);
            ```""")
    else:
        native = textwrap.dedent(f"""\
            ```c
            /* C ABI — libp4a */
            p4a_context_t* ctx = p4a_context_create();
            p4a_config_t*  cfg = p4a_config_create();
            p4a_method_result_t* res = NULL;
            p4a_{c_fn}(ctx, cfg, &x_view, &y_view, /* hyperparams */, &res);
            /* … read coefficients / mask / scores via */
            /* p4a_method_result_get_double_matrix / vector / scalar … */
            p4a_method_result_destroy(res);
            p4a_config_destroy(cfg);
            p4a_context_destroy(ctx);
            ```""")

    # ---- pls4all.python (tier-1) ----
    py_kwargs = ""
    extras = []
    if needs_xt:
        extras.append("X_target=X_target")
    if needs_sw:
        extras.append("sample_weights=sample_w")
    if needs_y:
        extras.append("y_labels=y_labels")
    if needs_g:
        extras.append("group_assignment=groups")
    if "n_components" in spec.get("cell_params", {}):
        py_kwargs = f"n_components={nc}"
    args_str = ", ".join([a for a in [py_kwargs] + extras if a])
    if name in {"aom_pls", "pop_pls"}:
        py_selector = "aom_per_component_select" if name == "pop_pls" else "aom_global_select"
        python_raw = textwrap.dedent(f"""\
            ```python
            import pls4all

            with pls4all.Context() as ctx, pls4all.Config() as cfg:
                bank = pls4all.OperatorBank()
                plan = pls4all.ValidationPlan()
                # Add compact nirs4all-style operators and CV folds.
                res = pls4all.{py_selector}(
                    ctx, cfg, bank, X.ravel(), y.ravel(), plan,
                    max_components={nc},
                    x_rows=X.shape[0], x_cols=X.shape[1],
                    y_rows=y.shape[0], y_cols=1,
                )
                values, rows, cols = res.predictions
            ```""")
    elif use_model_path:
        algo_enum = cat_algorithm or "PLS_REGRESSION"
        solver_enum = cat_solver or "SIMPLS"
        python_raw = textwrap.dedent(f"""\
            ```python
            import pls4all
            from pls4all import Algorithm, Solver
            with pls4all.Context() as ctx, pls4all.Config() as cfg:
                cfg.algorithm = Algorithm.{algo_enum}
                cfg.solver = Solver.{solver_enum}
                cfg.n_components = {nc}
                with pls4all.Model.fit(ctx, cfg, X, y) as mdl:
                    y_hat = mdl.predict(X_test)
            ```""")
    else:
        python_raw = textwrap.dedent(f"""\
            ```python
            import pls4all
            from pls4all._methods import {c_fn}
            with pls4all.Context() as ctx, pls4all.Config() as cfg:
                res = {c_fn}(ctx, cfg, X, y{', ' + args_str if args_str else ''})
            # then: res.matrix("predictions"), res.matrix("coefficients"),
            # res.vector("mask"), res.scalar("intercept"), …
            ```""")

    # ---- pls4all.sklearn (tier-2 Python) — pull real __init__ params ----
    py_class = METHOD_TO_PY_SKLEARN.get(name)
    py_fn = METHOD_TO_PY_FN.get(name)
    if py_class and py_class in py_docs:
        meta = py_docs[py_class]
        params = meta.get("init_params") or []
        rendered = []
        for p in params:
            d = p.get("default")
            if d in (None, ""):
                rendered.append(p["name"])
            else:
                rendered.append(f"{p['name']}={d}")
        fit_args = ["X, y"]
        if needs_xt:
            fit_args.append("X_target=X_target")
        if needs_sw:
            fit_args.append("sample_weight=sample_w")
        python_sklearn = textwrap.dedent(f"""\
            ```python
            from pls4all.sklearn import {py_class}
            mdl = {py_class}({', '.join(rendered) or 'n_components=' + str(nc)})
            mdl.fit({', '.join(fit_args)})
            y_hat = mdl.predict(X_test)
            ```""")
    elif py_fn:
        python_sklearn = textwrap.dedent(f"""\
            ```python
            from pls4all.sklearn import {py_fn}
            result = {py_fn}(X, y, n_components={nc})
            ```""")
    elif name in {"aom_pls", "pop_pls"}:
        python_sklearn = (
            "_No tier-2 sklearn-style class yet — exposed via the "
            "`pls4all.aom_global_select` / "
            "`pls4all.aom_per_component_select` low-level ABI._")
    else:
        python_sklearn = (
            "_No tier-2 sklearn-style class — exposed only via "
            "`pls4all._methods`._")

    # ---- pls4all.R (tier-1 dispatcher) ----
    # Build params=list(...) for extra hyperparameters.
    extra_params = []
    for k, v in (spec.get("cell_params") or {}).items():
        if k in ("n_samples", "n_features", "n_components"):
            continue
        if isinstance(v, str):
            extra_params.append(f'{k} = "{v}"')
        elif isinstance(v, float):
            extra_params.append(f"{k} = {v}")
        else:
            extra_params.append(f"{k} = {v}L")
    params_arg = (", params = list(" + ", ".join(extra_params) + ")") \
        if extra_params else ""
    r_raw = textwrap.dedent(f"""\
        ```r
        library(pls4all)
        # Unified low-level dispatcher (May 2026 R cleanup):
        res <- pls4all_method("{name}", X, y,
                              n_components = {nc}L{params_arg})
        # res is a named list with MethodResult arrays/scalars.
        # selected_indices / top_k_intervals are 1-based.
        ```""")

    # ---- pls4all.R (tier-1 raw function from methods.R/methods_extra.R) ----
    r_raw_fn = METHOD_TO_R_RAW.get(name)
    r_raw_block = None
    if r_raw_fn and r_raw_fn in r_docs:
        sig = r_docs[r_raw_fn]["signature"]
        r_raw_block = textwrap.dedent(f"""\
            ```r
            library(pls4all)
            res  <- {sig}
            yhat <- pls4all_predict(res, X_test)
            ```""")

    # ---- pls4all.R (tier-2 formula + S3) ----
    r_formula_fn = METHOD_TO_R_FORMULA.get(name)
    r_formula_block = None
    if r_formula_fn and r_formula_fn in r_docs:
        r_formula_block = textwrap.dedent(f"""\
            ```r
            library(pls4all)
            fit  <- {r_formula_fn}(y ~ ., data = train, ncomp = {nc}L)
            yhat <- predict(fit, newdata = test)
            summary(fit)
            ```""")

    # ---- pls4all.R (CRAN `pls`-package compatibility) ----
    # `pls_compat.R` exports `plsr()`, `pcr()`, `mvr()` with the same
    # signatures as the CRAN `pls` package so existing chemometrics code
    # can swap pls4all in without a rewrite.
    r_pls_compat_fn = METHOD_TO_R_PLS_COMPAT.get(name)
    r_pls_compat_block = None
    if r_pls_compat_fn:
        r_pls_compat_block = textwrap.dedent(f"""\
            ```r
            library(pls4all)
            # Drop-in for CRAN `pls::{r_pls_compat_fn}` (same signature).
            fit  <- {r_pls_compat_fn}(y ~ ., ncomp = {nc}L, data = train,
                                       validation = "CV", segments = 10L)
            yhat <- predict(fit, newdata = test, ncomp = {nc}L)
            RMSEP(fit)
            ```""")

    # ---- pls4all.R (`mdatools::pls` matrix-API compatibility) ----
    # `mdatools_compat.R` exports `pls_mdatools(x, y, ncomp, method = ...)`
    # mirroring the matrix-oriented signature used by NIRS / chemometrics
    # workflows.
    r_mdatools_pair = METHOD_TO_R_MDATOOLS.get(name)
    r_mdatools_block = None
    if r_mdatools_pair:
        mfn, mmethod = r_mdatools_pair
        r_mdatools_block = textwrap.dedent(f"""\
            ```r
            library(pls4all)
            # Drop-in for `mdatools::pls(x, y, ncomp, method = "{mmethod}")`.
            fit  <- {mfn}(X, y, ncomp = {nc}L, method = "{mmethod}",
                           center = TRUE, scale = FALSE)
            yhat <- predict(fit, newdata = X_test, ncomp = {nc}L)
            ```""")

    # ---- pls4all.matlab (tier-1 function) ----
    m_fn = METHOD_TO_MATLAB_FN.get(name)
    if m_fn and m_fn in m_docs:
        # Use the actual signature header from the .m file.
        sig = m_docs[m_fn]["signature"].replace("function ", "").strip()
        matlab_tier1 = textwrap.dedent(f"""\
            ```matlab
            res = pls4all.{m_fn}(X, y, {nc});
            % see header of bindings/matlab/+pls4all/{m_fn}.m for full
            % parameter surface:
            %   {sig}
            yhat = predict(res, Xtest);
            ```""")
    else:
        matlab_tier1 = textwrap.dedent(f"""\
            ```matlab
            res  = pls4all.fit("{name}", X, y, "NumComponents", {nc});
            yhat = predict(res, Xtest);
            ```""")

    # ---- pls4all.matlab (tier-2 classdef) ----
    m_class = METHOD_TO_MATLAB_CLASS.get(name)
    if m_class:
        matlab_tier2 = textwrap.dedent(f"""\
            ```matlab
            mdl  = pls4all.fit("{name}", X, y, "NumComponents", {nc});
            yhat = predict(mdl, Xtest);
            ```""")
    else:
        matlab_tier2 = (
            "_No idiomatic classdef wrapper — invoke "
            f"`pls4all.fit(\"{name}\", X, y, …)` directly from the unified "
            "MEX factory._")

    # ---- external references ----
    py_ext: list[str] = []
    if spec.get("has_py_ref"):
        py_ext.append("`scikit-learn` (declared as `python_reference` in "
                       "the registry)")
    for r in spec.get("extra_refs", []):
        if r in {"ikpls"}:
            py_ext.append("`ikpls`")
    if not py_ext:
        py_ext.append("_No widely installable Python reference._")

    r_ext: list[str] = []
    if spec.get("has_r_ref"):
        r_ext.append("CRAN / Bioconductor reference declared in the "
                      "registry (see *Bibliographic source*).")
    for r in spec.get("extra_refs", []):
        if r in {"mixOmics", "ropls", "spls"}:
            r_ext.append(f"`{r}`")
    if not r_ext:
        r_ext.append("_No R reference declared for this method._")

    parts: list[str] = [
        "### Usage\n",
        "Every pls4all binding tab dispatches into the same C kernel; "
        "the external libraries listed at the bottom of the page are the "
        "parity references registered in "
        "`benchmarks.parity_timing.registry`. Switch tabs to read the "
        "same fit in your language. The R package now ships "
        "drop-in-compatible facades for the CRAN `pls` package "
        "(`plsr`, `pcr`, `mvr`) and for the `mdatools::pls(x, y, ...)` "
        "matrix idiom — those tabs appear only on the methods that have "
        "a meaningful equivalence.\n",
    ]

    # tab-set 1: pls4all bindings -------------------------------------
    parts.append("**pls4all bindings**\n")
    parts.append("::::{tab-set}\n:class: pls4all-bindings\n")
    parts.append(_tab_open("C ABI · libp4a", "c", "c"))
    parts.append(native + "\n")
    parts.append(_tab_close())
    parts.append(_tab_open(
        "Python · pls4all (raw)", "python-raw", "python"))
    parts.append(python_raw + "\n")
    parts.append(_tab_close())
    parts.append(_tab_open(
        "Python · pls4all.sklearn", "python-sklearn", "python"))
    parts.append(python_sklearn + "\n")
    parts.append(_tab_close())
    parts.append(_tab_open(
        "R · pls4all_method()", "r-dispatcher", "r"))
    parts.append(r_raw + "\n")
    parts.append(_tab_close())
    if r_raw_block:
        parts.append(_tab_open(
            "R · pls4all (raw fn)", "r-raw", "r"))
        parts.append(r_raw_block + "\n")
        parts.append(_tab_close())
    if r_formula_block:
        parts.append(_tab_open(
            "R · pls4all (formula+S3)", "r-formula", "r"))
        parts.append(r_formula_block + "\n")
        parts.append(_tab_close())
    if r_pls_compat_block:
        parts.append(_tab_open(
            "R · `pls` package compat", "r-pls-compat", "r"))
        parts.append(r_pls_compat_block + "\n")
        parts.append(_tab_close())
    if r_mdatools_block:
        parts.append(_tab_open(
            "R · `mdatools` compat", "r-mdatools", "r"))
        parts.append(r_mdatools_block + "\n")
        parts.append(_tab_close())
    parts.append(_tab_open(
        "MATLAB · pls4all (MEX)", "matlab-mex", "matlab"))
    parts.append(matlab_tier1 + "\n")
    parts.append(_tab_close())
    parts.append(_tab_open(
        "MATLAB · pls4all (classdef)", "matlab-classdef", "matlab"))
    parts.append(matlab_tier2 + "\n")
    parts.append(_tab_close())
    parts.append("::::\n")

    # Registry parity references block (separate card, not tabbed).
    #
    # Source of truth: the resolved metadata from
    # `truth_source_metadata_for(method)`. We render one bullet per
    # registry-declared reference (Python or R), each tagged with the
    # 📐 icon to match the parity table rows. Paper-only methods get a
    # 📜 marker explaining there is no executable reference.
    parts.append(f"\n**Registry parity references** "
                  f"{TRUTH_SOURCE_ICON}\n")
    parts.append(":::{card}\n:class-card: external-refs\n")
    if spec.get("paper_only"):
        parts.append(
            f"- {PAPER_ONLY_ICON} **Paper-only** — no executable parity "
            "reference; the `pls4all` implementation is verified by a "
            "smoke fit only. Canonical citation: "
            + spec["paper_only"]
        )
    if truth_sources:
        for cid in sorted(truth_sources):
            meta = truth_sources[cid]
            label = _truth_quality_label(
                meta.get("quality", "qualitative"),
                float(meta.get("tolerance") or 0.0),
            )
            note = (meta.get("notes") or "").strip()
            note_clause = f" — {note}" if note else ""
            parts.append(
                f"- {TRUTH_SOURCE_ICON} **`{cid}`** "
                f"({meta.get('language', '?')} · {meta.get('role', '?')}) — "
                f"`{meta.get('library', '?')}` "
                f"{meta.get('version', '')} · {label}{note_clause}"
            )
    elif not spec.get("paper_only"):
        # No resolved metadata available — fall back to the legacy
        # boolean summary. Happens when the doc-build env can't import
        # the registry (e.g. RTD without numpy fully provisioned).
        parts.append("- **Python** — " + "; ".join(py_ext))
        parts.append("- **R** — " + "; ".join(r_ext))
    parts.append(":::\n")
    return "\n".join(parts)


def parameters_section(method: str, spec: dict, cat: dict | None,
                        py_docs: dict) -> str:
    """Render the canonical parameters block (pls4all native names).

    Prefers the actual Python sklearn class __init__ signature (which
    *is* the canonical pls4all parameter surface) over the YAML catalog
    or the registry's cell_params dict."""
    rows: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()

    def add(n: str, t: str, d: str, note: str) -> None:
        if n and n not in seen:
            seen.add(n)
            rows.append((n, t, d, note))

    # 1) Python sklearn class __init__ signature — most authoritative.
    py_class = METHOD_TO_PY_SKLEARN.get(method)
    if py_class and py_class in py_docs:
        for p in py_docs[py_class].get("init_params") or []:
            add(p["name"], p.get("type", "") or "—",
                str(p.get("default", "")) or "—", "")

    # 2) Fall back to YAML catalog
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

    # 3) Registry cell_params — values used by the benchmark.
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
        # Markdown table cells can't contain raw newlines.
        nt = (nt or "").replace("\n", " ").replace("|", "\\|")
        d = d.replace("\n", " ")
        lines.append(f"| `{n}` | `{t}` | `{d}` | {nt} |")
    return "\n".join(lines)


def _docstring_summary(doc: str) -> str:
    """Take the first paragraph of a docstring, dropping Sphinx markers."""
    if not doc:
        return ""
    paragraphs = doc.strip().split("\n\n")
    return paragraphs[0].strip().replace("\n    ", " ").replace("\n  ", " ")


def render_method_page(spec: dict, cat: dict | None,
                       bench_rows: list[dict],
                       py_docs: dict, r_docs: dict, m_docs: dict,
                       truth_sources: dict[str, dict] | None = None) -> str:
    """Build the full markdown for one method."""
    name = spec["name"]
    truth_sources = truth_sources or {}
    bib = BIBLIOGRAPHY.get(name, {})
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

    # Python sklearn class docstring — when richer than the registry desc,
    # quote it.
    py_class = METHOD_TO_PY_SKLEARN.get(name)
    if py_class and py_class in py_docs:
        doc = py_docs[py_class]["docstring"]
        summary = _docstring_summary(doc)
        if summary and summary.lower() not in (spec.get("desc") or "").lower():
            parts.append(f"From the `pls4all.sklearn.{py_class}` "
                          f"docstring:\n\n> {summary}\n")
        if doc and "----------" in doc:
            # Full parameters block is typically below the summary —
            # surface it verbatim in a collapsed code block.
            parts.append("<details>\n<summary>Full Python "
                          "<code>sklearn</code>-wrapper docstring</summary>\n")
            parts.append("```text\n" + doc.strip() + "\n```\n")
            parts.append("</details>\n")

    if spec.get("notes"):
        parts.append("> **Registry note** — " + spec["notes"].replace("\n", " ")
                      + "\n")

    parts.append(parameters_section(name, spec, cat, py_docs) + "\n")

    parts.append("## Explanations\n")
    parts.append("### Bibliographic source\n")
    parts.append(bib.get("paper") or "_No curated reference; see registry "
                  "notes and the [benchmark methodology](../benchmarks/methodology.md)._")
    parts.append("\n### Mathematical principle\n")
    parts.append(bib.get("principle") or
                  "_Standard derivation — see the cited reference._")
    parts.append("\n### Implementation\n")
    parts.append(bib.get("implementation") or
                  f"`p4a_{name}_fit` in libp4a.")

    # R roxygen — pull the first paragraph of the formula-style wrapper
    # if available; otherwise the raw-function one.
    r_meta = None
    for r_name in (METHOD_TO_R_FORMULA.get(name),
                    METHOD_TO_R_RAW.get(name)):
        if r_name and r_name in r_docs:
            r_meta = r_docs[r_name]
            break
    if r_meta:
        roxy = r_meta["roxygen"].strip()
        # Keep only the first paragraph
        first_para = roxy.split("\n\n")[0]
        parts.append("\nR roxygen note (`" + r_meta["file"] + "::"
                      + r_meta["signature"].split("(")[0] + "`):\n")
        parts.append("> " + first_para.replace("\n", "\n> "))

    # MATLAB classdef / function header — first 5 lines.
    m_name = (METHOD_TO_MATLAB_CLASS.get(name)
              or METHOD_TO_MATLAB_FN.get(name))
    if m_name and m_name in m_docs:
        header = m_docs[m_name]["header"].strip()
        if header:
            short = "\n".join(header.splitlines()[:6])
            parts.append("\nMATLAB header (`bindings/matlab/+pls4all/"
                          + m_docs[m_name]["file"] + "`):\n")
            parts.append("```text\n" + short + "\n```")
    parts.append("")

    parts.append(usage_section(name, spec, cat, py_docs, r_docs, m_docs,
                                truth_sources=truth_sources))

    parts.append(parity_table(name, bench_rows,
                                truth_sources=truth_sources))
    parts.append("")
    parts.append("---")
    parts.append("\n_See also_: "
                  "[benchmark overview](../benchmarks/overview.md) · "
                  "[methods index](index.md) · "
                  "[interactive dashboard](../landing/dashboard.md)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def render_index(methods: list[dict]) -> str:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for m in methods:
        by_group[method_group(m["name"])].append(m)
    out: list[str] = [
        "# Methods catalogue\n",
        "Every algorithm in `benchmarks.parity_timing.registry.METHODS` "
        "documented with its parameters, bibliographic source, math "
        "principle, every binding's signature, and its parity + timing "
        "rows.\n",
        f"_Total methods_: **{len(methods)}**. Grouped by family below.\n",
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
    out.append("---")
    out.append("\nSee the [benchmark overview](../benchmarks/overview.md) "
                "for how parity and timing are measured, and the "
                "[GitHub Pages dashboard](../landing/dashboard.md) for an "
                "interactive cross-method comparison.")
    return "\n".join(out)


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
    cat = parse_catalog(CATALOG_YAML)
    py_docs = parse_python_sklearn(PY_SKLEARN_DIR)
    r_docs = parse_r_signatures(R_DIR)
    m_docs = parse_matlab(MATLAB_DIR)
    truth_sources = load_truth_source_metadata(strict=args.strict)

    if args.no_bench or not CSV_PATH.exists():
        if args.strict and not CSV_PATH.exists():
            raise SystemExit(f"missing CSV: {CSV_PATH}")
        bench_rows: dict[str, list[dict]] = {}
    else:
        bench_rows = parse_csv(CSV_PATH)

    # Match registry-method to YAML catalog entry — used only as the
    # secondary parameter source.
    cat_by_cfn: dict[str, dict] = {}
    for entry in cat.values():
        cfn = entry.get("c_function", "")
        if cfn:
            cat_by_cfn[cfn] = entry
    method_to_cat: dict[str, dict] = {}
    for m in methods:
        n = m["name"]
        for suffix in ("_fit", "_run", "_select", "_compute"):
            if (n + suffix) in cat_by_cfn:
                method_to_cat[n] = cat_by_cfn[n + suffix]
                break
        else:
            for e in cat.values():
                if e.get("name", "").lower().startswith(
                        n.replace("_", "")):
                    method_to_cat[n] = e
                    break

    # Render
    for m in methods:
        rows = bench_rows.get(m["name"], [])
        page = render_method_page(
            m, method_to_cat.get(m["name"]), rows,
            py_docs, r_docs, m_docs,
            truth_sources=truth_sources.get(m["name"], {}))
        (out_dir / f"{m['name']}.md").write_text(page, encoding="utf-8")

    (out_dir / "index.md").write_text(render_index(methods), encoding="utf-8")

    print(f"wrote {len(methods)} method pages + index in {out_dir} "
            f"(py docstrings: {len(py_docs)}, R sigs: {len(r_docs)}, "
            f"MATLAB sigs: {len(m_docs)})")


if __name__ == "__main__":
    main()
