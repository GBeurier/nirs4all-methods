"""Portable generic n4m selector dispatch through the native C ABI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from n4m.feature_selection import SELECTOR_METHODS, Selector

CASES = {
    "spa_select": {"top_k": 5},
    "wvc_select": {"top_k": 5, "normalize": False},
    "stability_select": {"top_k": 5},
    "interval_select": {"interval_width": 3, "step": 1},
    "cars_select": {"n_iterations": 8, "min_features": 3},
    "uve_select": {"noise_features": 12, "noise_seed": 7},
    "random_frog_select": {
        "n_iterations": 8,
        "initial_size": 6,
        "min_size": 2,
        "max_size": 10,
        "top_k": 5,
        "seed": 7,
    },
    "scars_select": {"n_iterations": 8, "min_features": 3, "seed": 7},
    "ga_select": {
        "n_generations": 5,
        "population_size": 8,
        "min_features": 2,
        "max_features": 10,
        "seed": 7,
    },
    "pso_select": {"n_swarm": 8, "n_iterations": 5, "seed": 7},
    "vissa_select": {"n_iterations": 3, "n_submodels": 8, "ratio_kept": 0.5, "seed": 7},
    "shaving_select": {"n_steps": 3, "min_features": 3},
    "bve_select": {"n_steps": 3, "min_features": 3},
    "t2_select": {"alpha_thresholds": [0.1, 0.3, 0.5], "min_selected": 2},
    "wvc_threshold_select": {"normalize": True, "threshold": 0.1, "min_selected": 2},
    "emcuve_select": {
        "noise_features": 12,
        "noise_seed": 7,
        "n_ensembles": 3,
        "vote_threshold": 0.5,
    },
    "randomization_select": {
        "n_permutations": 100,
        "randomization_seed": 7,
        "alpha": 0.5,
    },
    "bipls_select": {"interval_width": 3, "min_intervals": 1},
    "sipls_select": {"interval_width": 3, "combination_size": 2},
    "rep_select": {"n_steps": 3, "min_features": 3, "remove_count": 1},
    "ipw_select": {"n_iterations": 3, "top_k": 5},
    "st_select": {"thresholds": [0.1, 0.5, 1.0], "min_selected": 2},
    "iriv_select": {"max_rounds": 4, "seed": 7},
    "irf_select": {
        "n_iterations": 8,
        "window_size": 3,
        "initial_intervals": 3,
        "top_k": 3,
        "seed": 7,
    },
    "vip_spa_select": {"vip_threshold": 0.3, "top_k": 5},
}


@pytest.fixture(scope="module")
def samples() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, 38, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(rows * bands / 11) + np.cos(rows / 3 + bands / 7) + rows * bands / 170
    y = 0.9 + 0.6 * X[:, 2] - 0.4 * X[:, 8]
    return X, y


@pytest.mark.parametrize("method", sorted(CASES))
def test_generic_selector_native_fit_and_replay(
    method: str, samples: tuple[np.ndarray, np.ndarray]
) -> None:
    assert set(SELECTOR_METHODS) == set(CASES)
    X, y = samples
    selector = Selector(method, n_components=2, method_params=CASES[method]).fit(
        X[:28], y[:28]
    )
    indices = selector.selected_indices_
    assert indices.dtype == np.int64 and indices.ndim == 1
    assert indices.size > 0 and len(set(indices.tolist())) == indices.size
    assert np.all((0 <= indices) & (indices < X.shape[1]))
    np.testing.assert_allclose(
        selector.transform(X[28:]), X[28:, np.sort(indices)], rtol=0, atol=0
    )
    repeat = Selector(method, n_components=2, method_params=CASES[method]).fit(
        X[:28], y[:28]
    )
    np.testing.assert_array_equal(repeat.selected_indices_, indices)


@pytest.mark.parametrize(
    "method,params,pattern",
    [
        ("ridge", {}, "unsupported"),
        ("spa_select", {}, "requires explicit top_k"),
        ("ga_select", {"n_generations": 2}, "requires explicit seed"),
        ("t2_select", {}, "requires explicit alpha_thresholds"),
        ("wvc_select", {"top_k": 2, "normalize": 1}, "boolean"),
        ("spa_select", {"top_k": 13}, "exceeds input width"),
        ("spa_select", {"top_k": 2, "unknown": 1}, "unsupported"),
    ],
)
def test_generic_selector_refuses_nonportable_params(
    method: str,
    params: dict[str, object],
    pattern: str,
    samples: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = samples
    with pytest.raises(ValueError, match=pattern):
        Selector(method, method_params=params).fit(X[:28], y[:28])


@pytest.mark.parametrize(
    "n_train,n_components,cases",
    [
        (28, 2, CASES),
        (
            4,
            1,
            {
                "interval_select": CASES["interval_select"],
                "stability_select": {"top_k": 3},
            },
        ),
        (
            5,
            1,
            {
                "interval_select": CASES["interval_select"],
                "stability_select": {"top_k": 3},
            },
        ),
    ],
)
def test_generic_selector_matches_independent_r_native_dispatcher(
    samples: tuple[np.ndarray, np.ndarray],
    n_train: int,
    n_components: int,
    cases: dict[str, dict[str, object]],
) -> None:
    r_libs = os.environ.get("N4M_R_LIBS")
    rscript = os.environ.get("N4M_RSCRIPT") or shutil.which("Rscript")
    if not r_libs or not rscript:
        pytest.skip(
            "set N4M_R_LIBS and provide Rscript for cross-binding selector parity"
        )
    X, y = samples
    request = {
        "X": X[:n_train].tolist(),
        "heldout": X[n_train:].tolist(),
        "y": y[:n_train].tolist(),
        "cases": cases,
        "n_components": n_components,
    }
    script = r"""
request <- jsonlite::fromJSON(paste(readLines("stdin", warn = FALSE), collapse = ""),
                              simplifyVector = FALSE)
X <- do.call(rbind, lapply(request$X, function(row) as.numeric(unlist(row))))
heldout <- do.call(rbind, lapply(request$heldout,
                                function(row) as.numeric(unlist(row))))
y <- as.numeric(unlist(request$y))
output <- lapply(names(request$cases), function(method) {
  params <- lapply(request$cases[[method]], function(value)
    if (is.list(value)) as.numeric(unlist(value)) else value)
  indices <- n4m::n4m_method(method, X, y, request$n_components,
                             params = params)$selected_indices
  list(selected_indices = unname(as.list(as.integer(indices) - 1L)),
       heldout = unname(lapply(seq_len(nrow(heldout)), function(i)
         unname(as.list(heldout[i, sort(indices)])))))
})
names(output) <- names(request$cases)
cat(as.character(jsonlite::toJSON(output, auto_unbox = TRUE, digits = 17L)))
"""
    env = {**os.environ, "R_LIBS": r_libs}
    result = subprocess.run(
        [rscript, "--vanilla", "-e", script],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    reference = json.loads(result.stdout)
    for method, params in cases.items():
        selector = Selector(
            method, n_components=n_components, method_params=params
        ).fit(X[:n_train], y[:n_train])
        np.testing.assert_array_equal(
            selector.selected_indices_,
            reference[method]["selected_indices"],
            err_msg=f"R/Python n4m selected_indices differ for {method}",
        )
        np.testing.assert_allclose(
            selector.transform(X[n_train:]),
            reference[method]["heldout"],
            rtol=0,
            atol=0,
            err_msg=f"R/Python heldout projection differs for {method}",
        )
