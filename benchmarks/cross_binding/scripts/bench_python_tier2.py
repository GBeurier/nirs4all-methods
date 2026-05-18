"""Python tier-2 — pls4all.sklearn idiomatic estimator API.

Maps each canonical method to its `pls4all.sklearn.<Class>` and instantiates
it with the same per-method parameters the registry uses, then runs the
sklearn-style `fit/predict` cycle. Registry parameters are mapped explicitly
onto constructor kwargs; unsupported algorithm parameters fail closed with
an actionable error instead of disappearing silently.

Methods without an idiomatic class emit `ok=False, reason="no sklearn
estimator for <algo>"` and render `—` in the dashboard, honestly.
"""
from __future__ import annotations

import inspect
import os

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   load_method)


# Algo → pls4all.sklearn class name. None means "no idiomatic class
# (yet)"; the script then fails the cell with a clear reason.
_SKLEARN_CLASS: dict[str, str | None] = {
    "pls":                    "PLSRegression",
    "pcr":                    "PCR",
    "opls":                   "OPLSRegression",
    "sparse_simpls":          "SparseSimplsRegression",
    "cppls":                  "CPPLSRegression",
    "ecr":                    "ECRegression",
    "mir_pls":                "MIRPLSRegression",
    "ridge_pls":              "RidgePLSRegression",
    "robust_pls":             "RobustPLSRegression",
    "missing_aware_nipals":   "MissingAwareNipalsRegression",
    "continuum_regression":   "ContinuumRegression",
    "kernel_pls_rbf":         "KernelPLSRegression",
    "weighted_pls":           "WeightedPLSRegression",
    "recursive_pls":          "RecursivePLSRegression",
    "di_pls":                 "DIPLSRegression",
    "mb_pls":                 "MBPLSRegression",
    "so_pls":                 "SOPLSRegression",
    "rosa":                   "ROSARegression",
    "n_pls":                  "NPLSRegression",
    "o2pls":                  "O2PLSRegression",
    "bagging_pls":            "BaggingPLSRegression",
    "boosting_pls":           "BoostingPLSRegression",
    "random_subspace_pls":    "RandomSubspacePLSRegression",
    "fused_sparse_pls":       "FusedSparsePLSRegression",
    "group_sparse_pls":       "GroupSparsePLSRegression",
    "lw_pls":                 "LWPLSRegression",
    "gpr_pls":                None,   # no idiomatic class yet
    "pls_glm":                "PLSGLMRegressor",
    "pls_cox":                "PLSCoxRegressor",
    "pls_lda":                "PLSLDAClassifier",
    "pls_logistic":           "PLSLogisticClassifier",
    "pls_qda":                "PLSQDAClassifier",
    "sparse_pls_da":          "SparsePLSDAClassifier",
    "pds":                    "PDSTransformer",
    "ds":                     "DSTransformer",
    "on_pls":                 None,   # paper-only / not yet wrapped
    "aom_preprocess":         None,
    "approximate_press":      None,
    "one_se_rule":            None,
    "pls_monitoring":         None,
    "pls_diagnostic_dmodx":   None,
    "pls_diagnostic_q":       None,
    "pls_diagnostic_t2":      None,
    # Selectors
    "bve_select":             "BVESelector",
    "bipls_select":           "BiPLSSelector",
    "cars_select":            "CARSSelector",
    "coefficient_select":     "CoefficientSelector",
    "emcuve_select":          "EMCUVESelector",
    "ga_select":              "GASelector",
    "interval_select":        "IntervalSelector",
    "ipw_select":             "IPWSelector",
    "irf_select":             "IRFSelector",
    "iriv_select":            "IRIVSelector",
    "pso_select":             "PSOSelector",
    "random_frog_select":     "RandomFrogSelector",
    "randomization_select":   "RandomizationSelector",
    "rep_select":             "REPSelector",
    "scars_select":           "SCARSSelector",
    "shaving_select":         "ShavingSelector",
    "sipls_select":           "SiPLSSelector",
    "spa_select":             "SPASelector",
    "st_select":              "STSelector",
    "stability_select":       "StabilitySelector",
    "t2_select":              "T2Selector",
    "uve_select":             "UVESelector",
    "vip_spa_select":         "VIPSPASelector",
    "vissa_select":           "VISSASelector",
    "variable_select_coef":   "CoefficientSelector",
    "variable_select_sr":     "SelectivityRatioSelector",
    "variable_select_vip":    "VIPSelector",
    "wvc_select":             "WVCSelector",
    "wvc_threshold_select":   "WVCThresholdSelector",
}


_REGISTRY_DATA_PARAMS = {
    "n_samples",
    "n_features",
    "n_targets",
    "n_classes",
    "n_blocks",
    "n_unique_per_block",
}

_PARAM_ALIASES: dict[str, dict[str, str]] = {
    "interval_select": {"interval_step": "step"},
    "iriv_select": {"fold": "n_folds"},
}

_REFERENCE_ONLY_PARAMS: dict[str, set[str]] = {
    # The pls4all REP kernel consumes n_steps/min_features/remove_count.
    # These knobs configure the external plsVarSel reference adapter only.
    "rep_select": {"rep_ratio", "rep_repeats", "rep_vip_threshold"},
    # The pls4all VIP-SPA kernel is deterministic; this seed belongs to
    # the auswahl reference adapter in the registry.
    "vip_spa_select": {"seed"},
}

_BENCH_CTOR_OVERRIDES: dict[str, dict[str, object]] = {
    # sklearn's UVESelector defaults to min_features=n_components so a
    # downstream Pipeline cannot receive an empty feature matrix. The
    # parity bench compares raw selector masks, so keep the C-kernel's
    # empty-selection semantics here explicitly.
    "uve_select": {"min_features": 0},
}


def _constructor_kwargs(algo: str, cls, params: dict) -> dict:
    """Map registry params to constructor kwargs and fail on leftovers."""
    sig = inspect.signature(cls.__init__)
    keep = set(sig.parameters) - {"self"}
    aliases = _PARAM_ALIASES.get(algo, {})
    reference_only = _REFERENCE_ONLY_PARAMS.get(algo, set())
    out: dict = {}
    unsupported: list[str] = []
    for key, value in params.items():
        target = aliases.get(key, key)
        if target in keep:
            out[target] = value
        elif key in _REGISTRY_DATA_PARAMS or key in reference_only:
            continue
        else:
            unsupported.append(key)
    if unsupported:
        accepted = ", ".join(sorted(keep)) or "<none>"
        raise RuntimeError(
            "python_tier2: unsupported registry parameter(s) for "
            f"{algo} -> {cls.__name__}: {', '.join(sorted(unsupported))}. "
            "Add an alias to _PARAM_ALIASES, mark reference-only parameters "
            "in _REFERENCE_ONLY_PARAMS, or expose them on the sklearn "
            f"constructor. Accepted constructor params: {accepted}"
        )
    return out


def _fit_kwargs(cls, extras: dict) -> dict:
    """Map registry extras (X_target, sample_weights, y_labels, …) onto
    the fit() signature for this class, only keeping recognised names."""
    sig = inspect.signature(cls.fit)
    keep = set(sig.parameters) - {"self"}
    out: dict = {}
    rename = {"sample_weights": "sample_weight"}
    for k, v in extras.items():
        target = rename.get(k, k)
        if target in keep:
            out[target] = v
    return out


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()
    class_name = _SKLEARN_CLASS.get(algo, ...)
    if class_name is None:
        # Honest miss: no idiomatic sklearn class is wired for this algo.
        raise RuntimeError(
            f"python_tier2: no sklearn estimator for '{algo}' "
            "(declare it in _SKLEARN_CLASS or add the class to pls4all.sklearn)")
    if class_name is ...:
        raise RuntimeError(
            f"python_tier2: algo '{algo}' not declared in _SKLEARN_CLASS")

    from pls4all import sklearn as p4a_sklearn
    from pls4all._context import Context

    cls = getattr(p4a_sklearn, class_name, None)
    if cls is None:
        raise RuntimeError(
            f"python_tier2: pls4all.sklearn.{class_name} missing for '{algo}'")

    method = load_method(algo)

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        params = adapted_params(method, n, p, nc)
        Y, extras = benchmark_inputs(method, X, y, params, seed)
        # Some array extras (group_assignment, block_sizes) are
        # constructor kwargs on the sklearn class, not fit kwargs.
        ctor_kwargs = _constructor_kwargs(algo, cls, params)
        ctor_sig_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        for key, value in _BENCH_CTOR_OVERRIDES.get(algo, {}).items():
            if key not in ctor_sig_params:
                raise RuntimeError(
                    f"python_tier2: bench override {key!r} is not accepted "
                    f"by {class_name}"
                )
            ctor_kwargs[key] = value
        ctor_kwargs.update(
            {k: v for k, v in extras.items() if k in ctor_sig_params}
        )
        # MBPLSRegression / ROSARegression / SOPLSRegression do
        # `if not self.block_sizes:` which is ambiguous on numpy arrays.
        # Convert to a plain Python list so the truth-value check works.
        if "block_sizes" in ctor_kwargs and hasattr(ctor_kwargs["block_sizes"], "tolist"):
            ctor_kwargs["block_sizes"] = list(ctor_kwargs["block_sizes"].tolist())
        if "n_components_per_block" in ctor_kwargs and hasattr(
                ctor_kwargs["n_components_per_block"], "tolist"):
            ctor_kwargs["n_components_per_block"] = list(
                ctor_kwargs["n_components_per_block"].tolist())
        # Inject n_components if the class accepts it and adapted_params
        # didn't already supply it.
        if "n_components" in ctor_sig_params and "n_components" not in ctor_kwargs:
            ctor_kwargs["n_components"] = nc
        # Force center_*=True, scale_*=False to match the cross-binding
        # convention (R/MATLAB dispatcher defaults, and what the rest of
        # the bench scripts now set explicitly on cfg). Without this,
        # sklearn classes like PCR / PLSRegression default scale_x=True
        # and their predictions diverge from cpp by the column-std
        # factor.
        for scale_kw in ("scale_x", "scale_y"):
            if scale_kw in ctor_sig_params:
                ctor_kwargs[scale_kw] = False
        for center_kw in ("center_x", "center_y"):
            if center_kw in ctor_sig_params:
                ctor_kwargs.setdefault(center_kw, True)
        est = cls(**ctor_kwargs)
        try:
            ctx = Context()
            ctx.num_threads = int(os.environ.get("BENCH_THREADS", "1"))
        except Exception:
            pass
        fit_sig_params = set(inspect.signature(cls.fit).parameters) - {"self"}
        # PLS-Cox-style: fit(X, survival_times, event_indicators) — y is
        # repurposed as survival data. Match the registry convention
        # exactly: `survival_times = abs(y) + 0.5` (already in extras
        # under "sample_weights"), `event_indicators = (y_labels > 0)`.
        if "survival_times" in fit_sig_params:
            sw = extras.get("sample_weights")
            yl = extras.get("y_labels")
            if sw is None:
                sw = np.abs(y.ravel()) + 0.5
            if yl is None:
                yl = np.ones(n, dtype=np.int32)
            ev = (np.asarray(yl, dtype=np.int32) > 0).astype(np.int32)
            est.fit(X, survival_times=np.asarray(sw, dtype=np.float64),
                    event_indicators=ev)
        # Classifiers carry labels in extras under "y_labels"; the
        # estimator either expects `y_labels=` or just `y` in label form.
        elif "y_labels" in extras and "y_labels" in fit_sig_params:
            fkw = _fit_kwargs(cls, extras)
            est.fit(X, **fkw)
        elif "y_labels" in extras and "y" in fit_sig_params:
            fkw = _fit_kwargs(cls, {k: v for k, v in extras.items()
                                    if k != "y_labels"})
            est.fit(X, extras["y_labels"], **fkw)
        else:
            fkw = _fit_kwargs(cls, extras)
            yarg = Y if Y.shape[1] > 1 else Y.ravel()
            # PDS/DS-style transformers want (X, y=None, X_target=...)
            if "y" in fit_sig_params and fit_sig_params == {"X", "y", "X_target"}:
                est.fit(X, y=yarg, **fkw)
            elif "y" in fit_sig_params:
                est.fit(X, yarg, **fkw)
            else:
                est.fit(X, **fkw)
        # Output: align with what the registry's parity comparison expects
        # (method.prediction_key tells us which result matrix the C kernel
        # would have populated). Selectors emit a 1×p mask; classifiers
        # emit continuous discriminant scores via `decision_function`, not
        # the integer labels `predict` returns; regressors use `predict`.
        pkey = method.prediction_key
        if pkey in ("selected_indices", "mask", "support"):
            # sklearn-style selectors expose `get_support()` (bool mask)
            # or `selected_indices_`. Build the 1×p float mask the C
            # parity reference produces.
            if hasattr(est, "support_"):
                support = np.asarray(est.support_, dtype=bool)
                mask = np.zeros((1, p), dtype=np.float64)
                mask[0, support[:p]] = 1.0
                return mask
            if hasattr(est, "selected_indices_"):
                sel = np.asarray(est.selected_indices_, dtype=np.int64)
                mask = np.zeros((1, p), dtype=np.float64)
                valid = sel[(sel >= 0) & (sel < p)]
                mask[0, valid] = 1.0
                return mask
            if hasattr(est, "get_support"):
                support = np.asarray(est.get_support(), dtype=bool)
                mask = np.zeros((1, p), dtype=np.float64)
                mask[0, support[:p]] = 1.0
                return mask
        if pkey in ("decision_scores", "decision_function") and hasattr(
                est, "decision_function"):
            return np.asarray(est.decision_function(X))
        # Classifier-style estimators with `decision_function`: prefer
        # the continuous scores over integer `predict` labels so the
        # parity matrix lines up with the C kernel's `predictions` field
        # (which IS continuous for pls_qda / sparse_pls_da / etc.).
        if hasattr(est, "decision_function") and hasattr(est, "classes_"):
            return np.asarray(est.decision_function(X))
        if hasattr(est, "predict"):
            return np.asarray(est.predict(X))
        if hasattr(est, "transform"):
            return np.asarray(est.transform(X))
        if hasattr(est, "selected_indices_"):
            mask = np.zeros((1, p), dtype=np.float64)
            sel = np.asarray(est.selected_indices_, dtype=np.int64)
            valid = sel[(sel >= 0) & (sel < p)]
            mask[0, valid] = 1.0
            return mask
        raise RuntimeError(
            f"python_tier2: {class_name} exposes no predict/transform/selected_indices_")

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    import pls4all
    versions = collect_versions("Python",
                                  pls4all=getattr(pls4all, "__version__", "?"),
                                  numpy=_safe_version("numpy"),
                                  sklearn_class=class_name)
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()
