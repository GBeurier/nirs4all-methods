"""Runtime contract guard: the core surface imports with NumPy alone.

The ``nirs4all-methods`` / ``pls4all`` wheels declare only ``numpy`` as a runtime
dependency; ``scikit-learn`` (and ``scipy``) are optional. ``n4m._impl.compat``
provides fallback ``BaseEstimator`` / ``TransformerMixin`` so the estimator layer
still imports without scikit-learn, and the AOM-Ridge simplex solver falls back to
a projected-gradient path without SciPy.

Every test environment (the dev venv and the cibuildwheel test stage) installs
scikit-learn, so nothing here would otherwise catch a regression where a module
grows a hard ``import sklearn`` at module scope and silently breaks the advertised
"works with NumPy alone" contract. This test proves the contract directly by
importing the package in a child interpreter with ``sklearn`` and ``scipy``
blocked at the meta-path level, then exercising the AOM / moment facades and a
scikit-learn-style wrapper on the fallback base.
"""
from __future__ import annotations

import os
import subprocess
import sys

# Child program: blocks sklearn/scipy, then imports the core + AOM/moment facades
# and runs the direct function surface plus a sklearn-style wrapper. Kept as a
# subprocess so the block is clean even after the parent session imported either
# library. Uses only numpy + n4m. Prints a sentinel on success.
_CHILD = r'''
import sys

_BLOCK = {"sklearn", "scipy"}


class _Block:
    """Meta-path finder that makes sklearn/scipy look uninstalled."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCK:
            raise ModuleNotFoundError("blocked for test: " + name)
        return None


sys.meta_path.insert(0, _Block())
for _m in [m for m in list(sys.modules) if m.split(".")[0] in _BLOCK]:
    del sys.modules[_m]

import numpy as np

import n4m
from n4m._impl import aom_facade as aom
from n4m._impl import moment_facade as moment

# Importing the core surface must not have pulled in the optional deps.
assert "sklearn" not in sys.modules, sorted(m for m in sys.modules if m.startswith("sklearn"))
assert "scipy" not in sys.modules, sorted(m for m in sys.modules if m.startswith("scipy"))

# The estimator layer must be running on the dependency-light fallback base.
from n4m._impl.compat import BaseEstimator
assert BaseEstimator.__module__ == "n4m._impl.compat", BaseEstimator.__module__

assert n4m.abi_version()  # forces libn4m to load + respond

# Direct ABI-close function facades (NumPy-only).
rng = np.random.default_rng(0)
X = rng.standard_normal((40, 6))
y = X[:, 0] - 0.3 * X[:, 2] + 0.01 * rng.standard_normal(40)

assert moment.moments(X, y)["xtx"].shape == (6, 6)
r = moment.ridge(X, y, alpha=0.1, scale_x=False)
assert r["predictions"].shape[0] == 40
assert moment.cppls(X, y, n_components=2)["predictions"].shape[0] == 40
assert moment.ecr(X, y, n_components=2, alpha=0.5)["predictions"].shape[0] == 40
assert aom.aom_pls(X, y, max_components=3) is not None
assert aom.available_methods() and moment.available_methods()

# A sklearn-style wrapper still fits/predicts on the fallback base, matching the
# direct native head exactly.
from n4m.estimators.regression.regularized import Ridge
model = Ridge(alpha=0.1, scale_x=False).fit(X, y)
np.testing.assert_allclose(
    model.predict(X), r["predictions"].ravel(), rtol=1e-10, atol=1e-10
)

# The AOM-Ridge blender imports and its SciPy-optional simplex QP stays usable.
from n4m.ensemble import AOMRidgeBlender
assert AOMRidgeBlender is not None

print("SKLEARN_OPTIONAL_OK")
'''


def test_core_surface_imports_without_sklearn_or_scipy():
    # Replicate the parent's import paths so the child finds n4m wherever this
    # suite found it (PYTHONPATH source layout, editable install, or site-packages),
    # and inherits N4M_LIB_PATH / the dev-build fallback for libn4m discovery.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SKLEARN_OPTIONAL_OK" in proc.stdout, proc.stdout
