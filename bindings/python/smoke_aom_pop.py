"""Python parity smoke test for the AOM global and POP public C ABI.

Drives the new p4a_aom_global_select / p4a_aom_per_component_select entry
points through the ctypes Python binding and compares every output against
the canonical bench-AOM_v0 fixtures committed under `parity/fixtures/`.

Run after building libp4a::

    cmake --build --preset dev-release --parallel
    python bindings/python/smoke_aom_pop.py

Exits 0 on success, non-zero on first mismatch. No mocked data and no
hard-coded reference values — every expected number comes from the JSON
fixture so the script tracks the canonical oracle automatically.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BINDING_SRC = REPO_ROOT / "bindings" / "python" / "src"
FIXTURE_ROOT = REPO_ROOT / "parity" / "fixtures"

sys.path.insert(0, str(BINDING_SRC))

import pls4all  # noqa: E402

ABS_TOL = 1e-8
REL_TOL = 1e-8


def _approx_equal(actual: float, expected: float) -> bool:
    diff = abs(actual - expected)
    scale = max(abs(actual), abs(expected), 1.0)
    return diff <= ABS_TOL or diff <= REL_TOL * scale


def _check_vec(label: str, actual: list[float], expected: list[float]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: size {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if not _approx_equal(a, e):
            raise AssertionError(
                f"{label}[{i}]: {a!r} != {e!r} (diff={abs(a-e):.3g})")


def _check_ints(label: str, actual: list[int], expected: list[int]) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: size {len(actual)} != {len(expected)}")
    for i, (a, e) in enumerate(zip(actual, expected)):
        if int(a) != int(e):
            raise AssertionError(
                f"{label}[{i}]: {int(a)} != {int(e)}")


def _build_bank(fixture: dict) -> pls4all.OperatorBank:
    expected = fixture["expected"]
    kinds = expected["operator_kinds"]["values"]
    offsets = expected["operator_param_offsets"]["values"]
    params = expected["operator_params"]["values"]
    bank = pls4all.OperatorBank()
    for op_idx, kind in enumerate(kinds):
        start = offsets[op_idx]
        stop = offsets[op_idx + 1]
        bank.add(int(kind), params[start:stop] if stop > start else None)
    return bank


def _build_plan(fixture: dict) -> pls4all.ValidationPlan:
    expected = fixture["expected"]
    train_off = expected["fold_train_offsets"]["values"]
    train_idx = expected["fold_train_indices"]["values"]
    test_off = expected["fold_test_offsets"]["values"]
    test_idx = expected["fold_test_indices"]["values"]
    plan = pls4all.ValidationPlan()
    n_samples = int(fixture["data"]["X"]["shape"][0])
    plan.n_samples = n_samples
    for fold in range(len(train_off) - 1):
        ts = train_idx[train_off[fold] : train_off[fold + 1]]
        es = test_idx[test_off[fold] : test_off[fold + 1]]
        plan.add_fold(ts, es)
    return plan


def _make_simpls_config() -> pls4all.Config:
    cfg = pls4all.Config()
    cfg.algorithm = pls4all.Algorithm.PLS_REGRESSION
    cfg.solver = pls4all.Solver.SIMPLS
    cfg.deflation = pls4all.Deflation.REGRESSION
    cfg.center_x = True
    cfg.scale_x = False
    cfg.center_y = True
    cfg.scale_y = False
    cfg.store_scores = False
    return cfg


def _flat_y(fixture: dict) -> tuple[list[float], int, int]:
    y_shape = fixture["data"]["Y"]["shape"]
    y_values = fixture["data"]["Y"]["values"]
    y_rows = int(y_shape[0])
    y_cols = int(y_shape[1]) if len(y_shape) > 1 else 1
    if len(y_values) != y_rows * y_cols:
        raise AssertionError(
            f"fixture Y shape {y_rows}x{y_cols} but {len(y_values)} values")
    return y_values, y_rows, y_cols


def smoke_global(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text())
    expected = fixture["expected"]
    x_shape = fixture["data"]["X"]["shape"]
    x_rows, x_cols = int(x_shape[0]), int(x_shape[1])
    x_values = fixture["data"]["X"]["values"]
    y_values, y_rows, y_cols = _flat_y(fixture)

    max_components = int(fixture["generator"]["params"]["max_components"])

    with pls4all.Context() as ctx, _make_simpls_config() as cfg, \
         _build_bank(fixture) as bank, _build_plan(fixture) as plan:
        result = pls4all.aom_global_select(
            ctx, cfg, bank, x_values, y_values, plan, max_components,
            x_rows=x_rows, x_cols=x_cols, y_rows=y_rows, y_cols=y_cols,
        )
        try:
            n_operators = result.n_operators
            assert n_operators == len(expected["operator_kinds"]["values"]), (
                f"n_operators: {n_operators} != "
                f"{len(expected['operator_kinds']['values'])}")
            assert result.max_components == max_components
            assert result.selected_operator_index == int(
                expected["selected_operator_index"])
            assert result.selected_n_components == int(
                expected["selected_n_components"])
            if not _approx_equal(result.best_score, expected["best_score"]):
                raise AssertionError(
                    f"best_score: {result.best_score!r} != {expected['best_score']!r}")
            _check_ints("operator_kinds", result.operator_kinds,
                        expected["operator_kinds"]["values"])
            _check_vec("operator_scores", result.operator_scores,
                       expected["operator_scores"]["values"])
            curves, curve_rows, curve_cols = result.rmse_curves
            assert curve_rows == n_operators
            assert curve_cols == max_components
            _check_vec("rmse_curves", curves, expected["rmse_curves"]["values"])
            preds, pred_rows, pred_cols = result.predictions
            assert pred_rows == x_rows
            assert pred_cols == y_cols
            _check_vec("predictions", preds, expected["predictions"]["values"])
        finally:
            result.close()
    print(f"  OK {fixture_path.name}")


def smoke_pop(fixture_path: Path) -> None:
    fixture = json.loads(fixture_path.read_text())
    expected = fixture["expected"]
    x_shape = fixture["data"]["X"]["shape"]
    x_rows, x_cols = int(x_shape[0]), int(x_shape[1])
    x_values = fixture["data"]["X"]["values"]
    y_values, y_rows, y_cols = _flat_y(fixture)

    max_components = int(fixture["generator"]["params"]["max_components"])

    with pls4all.Context() as ctx, _make_simpls_config() as cfg, \
         _build_bank(fixture) as bank, _build_plan(fixture) as plan:
        result = pls4all.aom_per_component_select(
            ctx, cfg, bank, x_values, y_values, plan, max_components,
            x_rows=x_rows, x_cols=x_cols, y_rows=y_rows, y_cols=y_cols,
        )
        try:
            assert result.n_operators == len(expected["operator_kinds"]["values"])
            assert result.max_components == max_components
            assert result.selected_n_components == int(
                expected["selected_n_components"])
            if not _approx_equal(result.best_score, expected["best_score"]):
                raise AssertionError(
                    f"best_score: {result.best_score!r} != {expected['best_score']!r}")
            _check_ints("operator_kinds", result.operator_kinds,
                        expected["operator_kinds"]["values"])
            _check_ints("selected_operator_indices",
                        result.selected_operator_indices,
                        expected["selected_operator_indices"]["values"])
            comps, comp_rows, comp_cols = result.component_scores
            assert comp_rows == max_components
            assert comp_cols == result.n_operators
            _check_vec("component_scores", comps,
                       expected["component_scores"]["values"])
            _check_vec("prefix_scores", result.prefix_scores,
                       expected["prefix_scores"]["values"])
            preds, pred_rows, pred_cols = result.predictions
            assert pred_rows == x_rows
            assert pred_cols == y_cols
            _check_vec("predictions", preds, expected["predictions"]["values"])
        finally:
            result.close()
    print(f"  OK {fixture_path.name}")


def main() -> int:
    print(f"pls4all version: {pls4all.version()}")
    print(f"ABI:             {pls4all.abi_version()}")
    print(f"Library path:    {pls4all.lib._name}")
    print("\nGlobal AOM-SIMPLS fixtures:")
    global_fixtures = sorted(FIXTURE_ROOT.glob("synthetic_aom_global_*_cv_v1.json"))
    for path in global_fixtures:
        smoke_global(path)
    print("\nPOP per-component fixtures:")
    pop_fixtures = sorted(FIXTURE_ROOT.glob("synthetic_aom_pop_*_v1.json"))
    for path in pop_fixtures:
        smoke_pop(path)
    print("\nPython AOM/POP smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
