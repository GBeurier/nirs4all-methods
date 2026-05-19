"""Shared helpers for registry-driven cross-binding benchmark scripts."""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BINDING_SRC = REPO / "bindings" / "python" / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(BINDING_SRC) not in sys.path:
    sys.path.insert(0, str(BINDING_SRC))

from benchmarks.parity_timing.registry import (  # noqa: E402
    get_method,
    iter_reference_factories,
    reference_id,
)


def _cap_components(value: int, n: int, p: int) -> int:
    return int(max(1, min(int(value), int(n) - 1, int(p))))


def _factor_pair(p: int) -> tuple[int, int]:
    root = int(np.sqrt(max(1, p)))
    for a in range(root, 0, -1):
        if p % a == 0:
            return a, p // a
    return 1, p


def adapted_params(method, n: int, p: int, n_components: int) -> dict:
    """Adapt a registry MethodSpec cell to the orchestrator's (n, p)."""
    params = dict(method.cell_params)
    params["n_samples"] = int(n)
    params["n_features"] = int(p)
    use_registry_components = os.environ.get("BENCH_REGISTRY_CELLS") == "1"
    if "n_components" in params:
        requested = params["n_components"] if use_registry_components else n_components
        params["n_components"] = _cap_components(requested, n, p)
    if method.name in {"pls_lda", "pls_logistic", "pls_qda", "sparse_pls_da"}:
        if method.name in {"pls_lda", "pls_logistic"}:
            # The LDA/logistic C kernels are binary classifiers. Keep the
            # benchmark cell in that supported regime even if the registry
            # parity cell uses multiclass sklearn references.
            params["n_classes"] = 2
        n_classes = int(params.get("n_classes", 2))
        params["n_components"] = max(1, min(int(params["n_components"]),
                                            max(1, n_classes - 1)))
    if "max_components" in params:
        params["max_components"] = _cap_components(params["max_components"], n, p)
    if (method.name in {"ipw_select", "rep_select", "shaving_select"}
            and not use_registry_components):
        # These selectors are compared to compact plsVarSel survivor sets.
        # Keep their registry-tuned component counts during size sweeps instead
        # of inheriting the global dashboard n_components value.
        params["n_components"] = _cap_components(
            method.cell_params.get("n_components", params.get("n_components", n_components)),
            n, p)
    if "n_predictive" in params:
        params["n_predictive"] = _cap_components(
            min(params["n_predictive"], n_components), n, p)
    if method.name == "n_pls":
        j, k = _factor_pair(p)
        params["mode_j"] = int(j)
        params["mode_k"] = int(k)
        params["n_components"] = _cap_components(n_components, n, min(j, k, p))
    if "n_components_per_block" in params:
        n_blocks = int(params.get("n_blocks", len(params["n_components_per_block"])))
        params["n_blocks"] = max(1, min(n_blocks, p))
        params["n_components_per_block"] = np.full(
            params["n_blocks"], max(1, min(n_components, n - 1)),
            dtype=np.int32)
    if "n_unique_per_block" in params:
        n_blocks = int(params.get("n_blocks", len(params["n_unique_per_block"])))
        params["n_blocks"] = max(1, min(n_blocks, p))
        params["n_unique_per_block"] = np.ones(params["n_blocks"],
                                               dtype=np.int32)
    # Multi-block methods need a `block_sizes` partition of p. The
    # Python registry's `_split_into_blocks(X, n_blocks)` uses
    # `cols_per_block = p // n_blocks` for the first n_blocks-1 chunks
    # and gives the remainder to the LAST chunk (so p=50, n_blocks=3 →
    # [16, 16, 18]). Build the same distribution here so R/MATLAB
    # bindings consume the same partition the C kernel will see on the
    # Python path — otherwise cells like mb_pls / rosa / so_pls show a
    # spurious ~0.04 max_diff across bindings.
    n_blocks_val = params.get("n_blocks")
    if n_blocks_val is not None and "block_sizes" not in params:
        nb = max(1, min(int(n_blocks_val), p))
        base = p // nb
        sizes = [base] * (nb - 1) + [p - base * (nb - 1)] if nb >= 1 else [p]
        params["block_sizes"] = np.array(sizes, dtype=np.int32)
    # `min_selected` (T2/ST selectors) must respect n_components ≤
    # min_selected ≤ p. The registry's cell may have a small fixed value
    # that drops below n_components once the orchestrator overrides it.
    if "min_selected" in params and "n_components" in params:
        lo = int(params["n_components"])
        hi = int(p)
        params["min_selected"] = max(lo, min(int(params["min_selected"]), hi))
    if method.name == "bve_select":
        min_features = max(1, min(int(params.get("min_features", 5)), p))
        # BVE is defined by a backward trajectory. On non-registry size
        # sweeps, keep eliminating until the requested survivor floor so
        # the external plsVarSel reference and pls4all compare masks with
        # comparable cardinality instead of "almost all variables" vs
        # "final survivor set".
        params["min_features"] = min_features
        params["n_steps"] = max(1, p - min_features)
    if method.name == "uve_select" and not use_registry_components:
        # The global dashboard size sweep overrides n_components. Keep a
        # deterministic UVE noise baseline that remains comparable to the
        # plsVarSel mcuve survivor set at the dashboard's 100x50 smoke
        # size, while leaving the MethodSpec registry cell untouched.
        params["n_components"] = _cap_components(3, n, p)
        params["noise_features"] = min(max(29, int(params.get("noise_features", 5))), p)
        params["noise_seed"] = 11
    return params


def benchmark_inputs(method, X: np.ndarray, y: np.ndarray, params: dict,
                     seed: int):
    """Return `(Y, extras)` using the same contract as parity_timing.runner."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, p = X.shape
    n_targets = int(params.get("n_targets", 1))
    if n_targets <= 1:
        Y = y.reshape(-1, 1)
    else:
        cols = [y]
        for j in range(1, n_targets):
            cols.append(0.5 * y + 0.1 * X[:, (j - 1) % p])
        Y = np.column_stack(cols)

    rng = np.random.default_rng(seed)
    X_target = X + 0.01 * rng.standard_normal(X.shape)
    sample_weights = np.abs(y) + 0.5

    n_classes = int(params.get("n_classes", 0))
    if n_classes >= 2:
        labels = np.zeros(n, dtype=np.int32)
        order = np.argsort(y)
        labels[order] = (np.arange(n, dtype=np.int32) % n_classes)
    else:
        labels = (y > np.median(y)).astype(np.int32)

    n_groups = max(1, min(5, p))
    width = max(1, int(np.ceil(p / n_groups)))
    group_assignment = np.minimum(np.arange(p) // width,
                                  n_groups - 1).astype(np.int32)

    extras = {}
    if method.needs_x_target:
        extras["X_target"] = X_target
    if method.needs_sample_weights:
        extras["sample_weights"] = sample_weights
    if method.needs_labels:
        extras["y_labels"] = labels
    if method.needs_group_assignment:
        extras["group_assignment"] = group_assignment
    return Y, extras


def find_reference(method, ref_id: str, params: dict):
    for _role, factory in iter_reference_factories(method):
        ref = factory(**params)
        if ref is None:
            continue
        if reference_id(ref.library_name, ref.language) == ref_id:
            return ref
    return None


def load_method(name: str):
    return get_method(name)
