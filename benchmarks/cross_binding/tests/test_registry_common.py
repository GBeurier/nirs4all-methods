from __future__ import annotations

from dataclasses import replace

from benchmarks.cross_binding.scripts.bench_registry_common import (
    adapted_params,
    load_method,
)


def test_recursive_pls_window_is_bounded_by_30x30_rows() -> None:
    params = adapted_params(load_method("recursive_pls"), 30, 30, 5)

    assert params["n_samples"] == 30
    assert params["n_features"] == 30
    assert params["n_components"] == 5
    assert params["window_size"] == 29


def test_selector_size_params_stay_inside_30x30_features() -> None:
    method = load_method("irf_select")
    params = dict(method.cell_params)
    params["window_size"] = 60
    params["initial_intervals"] = 60
    params["top_k"] = 60
    method = replace(method, cell_params=params)

    adapted = adapted_params(method, 30, 30, 5)

    assert adapted["window_size"] == 30
    assert adapted["initial_intervals"] == 30
    assert adapted["top_k"] == 30
