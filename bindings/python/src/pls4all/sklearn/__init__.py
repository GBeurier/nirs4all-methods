"""pls4all.sklearn — scikit-learn-compatible wrappers over the pls4all C ABI.

Optional sub-module. Requires `scikit-learn` at runtime. Wrappers are
**thin reformatters** — no numerical logic lives here. The C kernel under
the binding is the source of truth; the wrapper exists only to honour
the sklearn estimator contract (`fit/predict/score`, `get_params/set_params`,
`Pipeline`, `GridSearchCV`).

Persistence strategy per family:

* **Model-based** estimators (`PLSRegression`, `OPLSRegression`,
  `PCR`, `SparsePLSRegression`, `PLSCanonical`, `PLSSVD`,
  `PLSDAClassifier`, `OPLSDAClassifier`) round-trip via the C ABI
  `.n4a` bundle (`p4a_model_export_to_buffer` /
  `p4a_model_import_from_buffer`). State survives pickle bit-exactly.
* **MethodResult-based regressors** that carry coefficients
  (`SparseSimplsRegression`, `CPPLSRegression`, `ECRegression`,
  `DIPLSRegression`, `MIRPLSRegression`, `MBPLSRegression`,
  `NPLSRegression`, `O2PLSRegression`, `PLSGLMRegressor`)
  serialize their (coef, x_mean, y_mean) state as plain NumPy.
* **In-sample-only regressors** (`WeightedPLSRegression`,
  `RobustPLSRegression`, `RidgePLSRegression`, `ContinuumRegression`,
  `RecursivePLSRegression`, `LWPLSRegression`,
  `MissingAwareNipalsRegression`, `GroupSparsePLSRegression`,
  `FusedSparsePLSRegression`, `BaggingPLSRegression`,
  `GPRPLSRegression`, `BoostingPLSRegression`,
  `RandomSubspacePLSRegression`, `SOPLSRegression`, `ROSARegression`)
  store the in-sample `predictions_` and refuse predict-on-new-X
  with an informative error pointing to tier 1. See
  :mod:`pls4all.sklearn._in_sample` for the contract.
* **Calibration transformers** (`PDSTransformer`, `DSTransformer`)
  expose ``TransformerMixin`` with ``fit(X, y=None, X_target=...)``.
* **Selectors** (28 SelectorMixin classes) — see
  :mod:`pls4all.sklearn._selection`.
* **Diagnostic functions** (T², Q, DModX, PRESS, one-SE,
  monitoring) — module-level functions in
  :mod:`pls4all.sklearn._diagnostics`, not BaseEstimator subclasses.
"""

from __future__ import annotations

try:
    import sklearn  # noqa: F401  — fail fast with a clean error message
except ImportError as exc:  # pragma: no cover - exercised manually
    raise ImportError(
        "pls4all.sklearn requires scikit-learn; install it with "
        "`pip install scikit-learn`."
    ) from exc

# Model-based regressors / classifiers
from ._regression import (
    OPLSRegression,
    PCR,
    PLSCanonical,
    PLSRegression,
    PLSSVD,
    SparsePLSRegression,
)
from ._classification import OPLSDAClassifier, PLSDAClassifier

# MethodResult-based regressors with predict-on-new-X
from ._method_result_regressors import (
    CPPLSRegression,
    DIPLSRegression,
    ECRegression,
    KernelPLSRegression,
    MBPLSRegression,
    MIRPLSRegression,
    NPLSRegression,
    O2PLSRegression,
    PLSGLMRegressor,
    SparseSimplsRegression,
)

# Additional classifiers / survival regressor
from ._classifiers_extras import (
    PLSCoxRegressor,
    PLSLDAClassifier,
    PLSLogisticClassifier,
    PLSQDAClassifier,
    SparsePLSDAClassifier,
)

# Calibration transfer (TransformerMixin)
from ._transformers import DSTransformer, PDSTransformer

# In-sample-only regressors
from ._in_sample import (
    BaggingPLSRegression,
    BoostingPLSRegression,
    ContinuumRegression,
    FusedSparsePLSRegression,
    GPRPLSRegression,
    GroupSparsePLSRegression,
    LWPLSRegression,
    MissingAwareNipalsRegression,
    RandomSubspacePLSRegression,
    RecursivePLSRegression,
    RidgePLSRegression,
    RobustPLSRegression,
    ROSARegression,
    SOPLSRegression,
    WeightedPLSRegression,
)

# Selectors (28)
from ._selection import (
    BVESelector,
    BiPLSSelector,
    CARSSelector,
    CoefficientSelector,
    EMCUVESelector,
    GASelector,
    IntervalSelector,
    IPWSelector,
    IRFSelector,
    IRIVSelector,
    PSOSelector,
    RandomFrogSelector,
    RandomizationSelector,
    REPSelector,
    SCARSSelector,
    SelectivityRatioSelector,
    ShavingSelector,
    SiPLSSelector,
    SPASelector,
    STSelector,
    StabilitySelector,
    T2Selector,
    UVESelector,
    VIPSelector,
    VIPSPASelector,
    VISSASelector,
    WVCSelector,
    WVCThresholdSelector,
)

# Diagnostic functions
from ._diagnostics import (
    aom_preprocess,
    approximate_press,
    dmodx_score,
    on_pls,
    one_se_rule,
    pls_monitoring,
    q_score,
    t2_score,
)


__all__ = [
    # Model-based regressors / classifiers
    "PLSRegression", "OPLSRegression", "PCR", "SparsePLSRegression",
    "PLSCanonical", "PLSSVD", "PLSDAClassifier", "OPLSDAClassifier",
    # MethodResult-based regressors (Python-side predict via coef)
    "SparseSimplsRegression", "CPPLSRegression", "ECRegression",
    "DIPLSRegression", "MIRPLSRegression", "MBPLSRegression",
    "NPLSRegression", "O2PLSRegression", "PLSGLMRegressor",
    "KernelPLSRegression",
    # Additional classifiers / survival
    "SparsePLSDAClassifier", "PLSLogisticClassifier",
    "PLSLDAClassifier", "PLSQDAClassifier", "PLSCoxRegressor",
    # Calibration transfer
    "PDSTransformer", "DSTransformer",
    # In-sample-only regressors
    "WeightedPLSRegression", "RobustPLSRegression",
    "RidgePLSRegression", "ContinuumRegression",
    "RecursivePLSRegression", "LWPLSRegression",
    "MissingAwareNipalsRegression",
    "GroupSparsePLSRegression", "FusedSparsePLSRegression",
    "BaggingPLSRegression", "GPRPLSRegression", "BoostingPLSRegression",
    "RandomSubspacePLSRegression", "SOPLSRegression",
    "ROSARegression",
    # 28 SelectorMixin variable-selection wrappers
    "BVESelector", "BiPLSSelector", "CARSSelector",
    "CoefficientSelector", "EMCUVESelector", "GASelector",
    "IntervalSelector", "IPWSelector", "IRFSelector",
    "IRIVSelector", "PSOSelector", "RandomFrogSelector",
    "RandomizationSelector", "REPSelector", "SCARSSelector",
    "SelectivityRatioSelector", "ShavingSelector", "SiPLSSelector",
    "SPASelector", "STSelector", "StabilitySelector",
    "T2Selector", "UVESelector", "VIPSelector",
    "VIPSPASelector", "VISSASelector", "WVCSelector",
    "WVCThresholdSelector",
    # Diagnostic functions
    "t2_score", "q_score", "dmodx_score", "pls_monitoring",
    "approximate_press", "one_se_rule", "aom_preprocess", "on_pls",
]
