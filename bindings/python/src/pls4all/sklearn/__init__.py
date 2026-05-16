"""pls4all.sklearn — scikit-learn-compatible wrappers over the pls4all C ABI.

Optional sub-module. Requires `scikit-learn` at runtime. Wrappers are
**thin reformatters** — no numerical logic lives here. The C kernel under
the binding is the source of truth; the wrapper exists only to honour
the sklearn estimator contract (`fit/predict/score`, `get_params/set_params`,
`Pipeline`, `GridSearchCV`).

Two persistence strategies are used depending on which C primitive the
method maps to:

* **Model-based** estimators (`PLSRegression`, `OPLSRegression`,
  `PLSDAClassifier`, …) round-trip via the C ABI `.n4a` bundle
  (`p4a_model_export_to_buffer` / `p4a_model_import_from_buffer`). The
  serialized state is bit-exact.
* **MethodResult-based** estimators (`VIPSelector`, `CARSSelector`, …)
  serialize the fitted arrays as plain NumPy state. Selectors do not
  produce a re-fittable `p4a_model_t`.
"""

from __future__ import annotations

try:
    import sklearn  # noqa: F401  — fail fast with a clean error message
except ImportError as exc:  # pragma: no cover - exercised manually
    raise ImportError(
        "pls4all.sklearn requires scikit-learn; install it with "
        "`pip install scikit-learn`."
    ) from exc

from ._regression import OPLSRegression, PLSRegression
from ._classification import PLSDAClassifier
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

__all__ = [
    # Model-based regressors / classifiers
    "PLSRegression",
    "OPLSRegression",
    "PLSDAClassifier",
    # 28 SelectorMixin variable-selection wrappers (Phase 54a)
    "BVESelector",
    "BiPLSSelector",
    "CARSSelector",
    "CoefficientSelector",
    "EMCUVESelector",
    "GASelector",
    "IntervalSelector",
    "IPWSelector",
    "IRFSelector",
    "IRIVSelector",
    "PSOSelector",
    "RandomFrogSelector",
    "RandomizationSelector",
    "REPSelector",
    "SCARSSelector",
    "SelectivityRatioSelector",
    "ShavingSelector",
    "SiPLSSelector",
    "SPASelector",
    "STSelector",
    "StabilitySelector",
    "T2Selector",
    "UVESelector",
    "VIPSelector",
    "VIPSPASelector",
    "VISSASelector",
    "WVCSelector",
    "WVCThresholdSelector",
]
