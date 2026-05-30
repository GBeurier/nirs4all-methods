"""Presentation policy for dashboard parity notes.

The orchestrator records selector parity as Jaccard set overlap in
``reference_parity_note``. A subset of ``selection_mismatch`` cells is expected:
those cells compare an n4m legacy C++ selector against an external reference
that uses a different random stream, noise model, CV split, or selector
thresholding convention. They are informative cross-checks, not regressions.

The benchmark matrix can also contain non-selector API/facade cells that run a
legacy binding convention while the canonical registry/C++/tier-1 path is
already exact. Those cells keep their timing and numeric delta, but the
dashboard marks them as cross-checks instead of red parity failures.
"""

from __future__ import annotations

import re


JACCARD_RE = re.compile(r"jaccard\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


SELECTION_BYDESIGN_REASONS: dict[str, str] = {
    "bve_select": (
        "R plsVarSel::bve_pls uses its own sampling/VIP elimination path; "
        "n4m legacy C++ uses deterministic greedy CV-RMSE backward elimination."
    ),
    "cars_select": (
        "The executable reference is R enpls::enpls.fs, a Monte-Carlo "
        "importance analog rather than n4m's CARS kernel."
    ),
    "ga_select": (
        "R plsVarSel::ga_pls and n4m's legacy C++ GA use different binary "
        "population/operator designs and RNG streams."
    ),
    "interval_select": (
        "R mdatools::ipls uses venetian CV and its interval scoring; n4m "
        "legacy C++ uses a fixed contiguous-fold validation plan."
    ),
    "ipw_select": (
        "R plsVarSel::ipw_pls uses the RC filter and package thresholds; n4m "
        "legacy C++ uses top-k iterative scores with damping controls."
    ),
    "irf_select": (
        "auswahl IntervalRandomFrog uses numpy RandomState proposals; n4m "
        "legacy C++ uses a different splitmix64 interval proposal chain."
    ),
    "iriv_select": (
        "The NumPy IRIV port uses default_rng and scipy rank tests; n4m legacy "
        "C++ uses its own splitmix64 candidate-generation path."
    ),
    "pso_select": (
        "pyswarms BinaryPSO and n4m legacy C++ PSO use different swarm state "
        "updates, velocity handling, and RNG streams."
    ),
    "random_frog_select": (
        "auswahl RandomFrog uses its own MCMC proposal chain and numpy "
        "RandomState; n4m legacy C++ uses a splitmix64 proposal chain."
    ),
    "rep_select": (
        "R plsVarSel::rep_pls repeats VIP-thresholded Monte-Carlo selection; "
        "n4m legacy C++ uses bounded backward elimination steps."
    ),
    "scars_select": (
        "The NumPy SCARS port and n4m legacy C++ kernel differ in the "
        "subsampling, shrinkage, and random stream details."
    ),
    "shaving_select": (
        "R plsVarSel::shaving chooses a CV-error-minimising survivor path; n4m "
        "legacy C++ follows an explicit step-count trajectory."
    ),
    "spa_select": (
        "R plsVarSel::spa_pls uses random subsampling and a Wilcoxon-style "
        "permutation test; n4m legacy C++ SPA is deterministic."
    ),
    "st_select": (
        "R plsVarSel::stpls uses a relative shrink-ladder sweep; n4m legacy "
        "C++ uses absolute coefficient thresholds."
    ),
    "stability_select": (
        "R mcuve stability and n4m legacy C++ stability use different "
        "subsampling, standardisation, and RNG/noise conventions."
    ),
    "uve_select": (
        "R mcuve_pls adds ncol(X) uniform-noise columns with R-MT; n4m legacy "
        "C++ uses a fixed noise count, splitmix64, and contiguous folds."
    ),
    "variable_select_sr": (
        "R plsVarSel::SR scores a pls::plsr model; n4m legacy C++ uses its "
        "own score/loading reconstruction for the selectivity ratio."
    ),
    "vissa_select": (
        "auswahl VISSA uses numpy RandomState submodel sampling; n4m legacy "
        "C++ differs in the sampling and aggregation path."
    ),
    "wvc_threshold_select": (
        "R plsVarSel::WVC_pls uses a median-scaled WVC threshold; n4m legacy "
        "C++ applies its own min-selected thresholding convention."
    ),
}


_LEGACY_API_BACKENDS = {
    "python_tier2",
    "r_tier1",
    "r_tier2",
    "r_pls_compat",
    "r_mdatools_compat",
    "matlab_tier1",
    "matlab_tier2",
}

_PYTHON_COMPOSITE_METHODS = {
    "bagging_pls",
    "boosting_pls",
    "group_sparse_pls",
    "n_pls",
    "pls_cox",
    "pls_glm",
    "pls_lda",
    "pls_logistic",
    "pls_qda",
    "random_subspace_pls",
}

_AUSWAHL_DEPENDENT_METHODS = {
    "irf_select",
    "random_frog_select",
    "vip_spa_select",
    "vissa_select",
}

_NONCANONICAL_BINDING_REASONS: dict[str, tuple[set[str], str]] = {
    "composite": (
        _PYTHON_COMPOSITE_METHODS,
        "Canonical registry/C++/Python tier-1 rows use the reference-equivalent "
        "composite path; this API/facade cell exercises the legacy binding "
        "kernel or facade convention, so it is a timing cross-check rather "
        "than a canonical parity gate.",
    ),
    "lw_pls": (
        {"lw_pls"},
        "Canonical registry/C++/Python tier-1 rows use the sanctioned "
        "Gaussian-weighted LW-PLS convention; this API/facade cell uses the "
        "legacy local-kernel convention.",
    ),
    "one_se_rule": (
        {"one_se_rule"},
        "Canonical rows feed the pooled CV-RMSEP matrix used by the R "
        "one-standard-error oracle; this API/facade cell follows its legacy "
        "input convention.",
    ),
    "cppls_facade": (
        {"cppls"},
        "The canonical CPPLS rows apply the NIPALS CPPLS convention exactly; "
        "this R compatibility facade cell does not carry that canonical "
        "benchmark convention in this matrix.",
    ),
    "sparse_pls_da_tier2": (
        {"sparse_pls_da"},
        "The canonical sparse PLS-DA rows compare one-hot predictions; this "
        "Python sklearn-style cell exposes decision-score semantics.",
    ),
}

_CPPLS_FACADE_BACKENDS = {"r_pls_compat", "r_mdatools_compat"}
_SPARSE_PLS_DA_SCORE_BACKENDS = {"python_tier2"}


def jaccard_from_note(note: str) -> float | None:
    note = (note or "").strip().lower()
    if not note.startswith("selection"):
        return None
    match = JACCARD_RE.search(note)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if note.startswith("selection_exact"):
        return 1.0
    return None


def selection_bydesign_reason(algorithm: str | None, note: str) -> str | None:
    if not (note or "").strip().lower().startswith("selection_mismatch"):
        return None
    return SELECTION_BYDESIGN_REASONS.get(algorithm or "")


def selection_note_with_reason(algorithm: str | None, note: str) -> str:
    clean = (note or "").strip()
    reason = selection_bydesign_reason(algorithm, clean)
    if not reason:
        return clean
    return f"{clean} | by design: {reason}"


def binding_cross_check_reason(algorithm: str | None,
                               backend: str | None) -> str | None:
    """Reason for known noncanonical binding/facade timing cells.

    These are not selector Jaccard cases. They are cells where the canonical
    method path is already exact elsewhere in the matrix, while this backend
    exposes an older or compatibility API convention. Keeping them as
    cross-checks preserves their timing signal without classifying the method
    as a parity failure.
    """
    algo = algorithm or ""
    be = backend or ""
    if be in _LEGACY_API_BACKENDS and algo in _PYTHON_COMPOSITE_METHODS:
        return _NONCANONICAL_BINDING_REASONS["composite"][1]
    if be in _LEGACY_API_BACKENDS and algo == "lw_pls":
        return _NONCANONICAL_BINDING_REASONS["lw_pls"][1]
    if be in _LEGACY_API_BACKENDS and algo == "one_se_rule":
        return _NONCANONICAL_BINDING_REASONS["one_se_rule"][1]
    if be in _CPPLS_FACADE_BACKENDS and algo == "cppls":
        return _NONCANONICAL_BINDING_REASONS["cppls_facade"][1]
    if be in _SPARSE_PLS_DA_SCORE_BACKENDS and algo == "sparse_pls_da":
        return _NONCANONICAL_BINDING_REASONS["sparse_pls_da_tier2"][1]
    return None


def dependency_unavailable_reason(algorithm: str | None,
                                  reason: str | None) -> str | None:
    """Return a neutral not-available reason for known optional deps."""
    algo = algorithm or ""
    text = (reason or "").lower()
    if algo in _AUSWAHL_DEPENDENT_METHODS and "auswahl" in text:
        return "not_available: reference requires the optional 'auswahl' package"
    return None
