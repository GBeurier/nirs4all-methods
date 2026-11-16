# SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
"""Explainability bindings (``n4m.explain``) over the native ``n4m/explain.h`` ABI.

Native checks use analytic references (linear models, a two-factor interaction
and a hand-built tree ensemble). Optional parity tests against the ``shap``
package run only when ``shap`` and scikit-learn are installed. The small
scikit-learn -> flat-array tree converter below is a TEST fixture; production
converters belong to the SDK host.
"""

from __future__ import annotations

import numpy as np
import pytest

from n4m import N4MError
from n4m import explain as ex


def _linear_model(w, b):
    def predict(rows):
        assert rows.dtype == np.float64 and rows.flags.c_contiguous and rows.ndim == 2
        return rows @ w + b

    return predict


def _interaction(rows):
    return rows[:, 0] * rows[:, 1] + 2.0 * rows[:, 2]


@pytest.fixture(scope="module")
def linear_case():
    rng = np.random.default_rng(3)
    w = np.array([0.5, -1.25, 2.0, 0.0, 0.75, -0.3])
    X = rng.normal(size=(4, 6))
    background = rng.normal(size=(5, 6))
    return w, 0.4, X, background


def _analytic_interaction(X, background):
    """Interventional Shapley values of f = x0*x1 + 2*x2 averaged over background."""
    phi = np.zeros((X.shape[0], 3))
    for b in background:
        phi[:, 0] += 0.5 * (X[:, 0] - b[0]) * (X[:, 1] + b[1])
        phi[:, 1] += 0.5 * (X[:, 1] - b[1]) * (X[:, 0] + b[0])
        phi[:, 2] += 2.0 * (X[:, 2] - b[2])
    phi /= background.shape[0]
    expected = np.mean(background[:, 0] * background[:, 1] + 2.0 * background[:, 2])
    return phi, expected


X3 = np.array([[1.5, -2.0, 0.5], [0.25, 3.0, -1.0]])
BG3 = np.array([[0.5, 1.0, 0.0], [-1.0, 0.5, 2.0]])


# ---------------------------------------------------------------------------
# Session explainers
# ---------------------------------------------------------------------------


def test_permutation_matches_linear_shap(linear_case):
    w, b, X, background = linear_case
    seen = []
    result = ex.explain(
        _linear_model(w, b),
        X,
        background=background,
        n_permutations=3,
        seed=7,
        max_batch_rows=50,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert (result.kind, result.exactness, result.additive) == ("permutation_shap", "sampled", True)
    assert result.group_aggregation == "group_game"
    assert result.baseline == "background"
    assert result.attributions.shape == (4, 6, 1)
    assert result.std_error is not None and result.std_error.shape == (4, 6, 1)
    assert result.additivity_max_abs_error < 1e-10
    assert seen and seen[-1][0] == seen[-1][1] == result.settings["total_evaluations"]
    assert result.evaluations == seen[-1][1] + 1  # + the n_outputs probe row
    assert result.settings["n_outputs_inferred"] is True
    assert result.engine["name"] == "nirs4all-methods"
    assert len(result.engine["abi"].split(".")) == 3

    linear = ex.linear_shap(X, w, reference=background.mean(axis=0), intercept=[b])
    assert (linear.kind, linear.exactness, linear.group_aggregation) == ("linear_shap", "exact", "none")
    assert linear.baseline is None and linear.evaluations == 0
    np.testing.assert_allclose(result.attributions, linear.attributions, atol=1e-10)
    np.testing.assert_allclose(result.expected, linear.expected, atol=1e-10)
    np.testing.assert_allclose(linear.attributions[:, :, 0], w * (X - background.mean(axis=0)), atol=1e-12)
    np.testing.assert_allclose(linear.predictions[:, 0], X @ w + b, atol=1e-12)


def test_linear_group_sums_are_group_game(linear_case):
    w, b, X, background = linear_case
    groups, n_groups = ex.groups_fixed_width(6, 4)
    assert n_groups == 2
    per_feature = ex.linear_shap(X, w, reference=background.mean(axis=0), intercept=[b])
    grouped = ex.linear_shap(
        X, w, reference=background.mean(axis=0), intercept=[b], feature_group=groups
    )
    assert grouped.group_aggregation == "group_game"
    np.testing.assert_allclose(
        grouped.attributions[:, 0, 0], per_feature.attributions[:, :4, 0].sum(axis=1), atol=1e-12
    )


def test_strided_input_matches_contiguous(linear_case):
    w, b, X, background = linear_case
    wide = np.zeros((X.shape[0], 12))
    wide[:, ::2] = X
    strided = ex.explain(_linear_model(w, b), wide[:, ::2], background=background, n_outputs=1,
                         n_permutations=2, seed=1)
    contiguous = ex.explain(_linear_model(w, b), X, background=background, n_outputs=1,
                            n_permutations=2, seed=1)
    np.testing.assert_array_equal(strided.attributions, contiguous.attributions)


def test_exact_shap_matches_analytic_shapley():
    phi, expected = _analytic_interaction(X3, BG3)
    result = ex.explain(_interaction, X3, background=BG3, method="exact_shap")
    assert result.exactness == "exact" and result.std_error is None
    np.testing.assert_allclose(result.attributions[:, :, 0], phi, atol=1e-12)
    np.testing.assert_allclose(result.expected[:, 0], expected, atol=1e-12)


def test_kernel_with_full_enumeration_is_exact():
    phi, _ = _analytic_interaction(X3, BG3)
    result = ex.explain(_interaction, X3, background=BG3, method="kernel_shap")
    assert result.exactness == "exact" and result.additive
    np.testing.assert_allclose(result.attributions[:, :, 0], phi, atol=1e-8)


def test_non_additive_explainers_report_their_kind():
    lime = ex.explain(_interaction, X3, background=BG3, method="lime", seed=3)
    assert (lime.exactness, lime.additive, lime.additivity_max_abs_error) == ("surrogate", False, None)
    occlusion = ex.explain(_interaction, X3, method="occlusion", baseline="interpolate")
    assert (occlusion.exactness, occlusion.additive, occlusion.baseline) == (
        "perturbation",
        False,
        "interpolate",
    )
    assert occlusion.settings["n_background"] == 0


def test_multi_output_inference_and_grouped_game():
    phi, _ = _analytic_interaction(X3, BG3)
    two = ex.explain(
        lambda rows: np.column_stack([_interaction(rows), -_interaction(rows)]),
        X3,
        background=BG3,
        method="exact_shap",
    )
    assert two.attributions.shape == (2, 3, 2)
    np.testing.assert_allclose(two.attributions[:, :, 0], phi, atol=1e-12)
    np.testing.assert_allclose(two.attributions[:, :, 1], -phi, atol=1e-12)
    groups, n_groups, _ = ex.groups_from_ranges(3, [(0, 2)])
    grouped = ex.explain(_interaction, X3, background=BG3, method="exact_shap",
                         feature_group=groups, n_groups=n_groups)
    assert grouped.attributions.shape == (2, 2, 1)
    np.testing.assert_allclose(grouped.attributions[:, 1, 0], phi[:, 2], atol=1e-12)


def test_failing_predictor_propagates_and_session_is_released(linear_case):
    w, b, X, background = linear_case

    def broken(rows):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        ex.explain(broken, X, background=background, n_outputs=1)
    again = ex.explain(_linear_model(w, b), X, background=background, n_outputs=1,
                       n_permutations=3, seed=7, max_batch_rows=50)
    assert again.additivity_max_abs_error < 1e-10


# ---------------------------------------------------------------------------
# TreeSHAP on a hand-built ensemble
# ---------------------------------------------------------------------------

# Tree 0: x0 <= .5 ? (x1 <= .5 ? 1 : 2) : 5, covers 4 / 2 / 1 / 1 / 2.
# Tree 1: x2 <= .5 ? 0 : 4, covers 4 / 1 / 3.
HAND_TREES = ex.TreeEnsemble(
    tree_offsets=[0, 5, 8],
    feature=[0, 1, -1, -1, -1, 2, -1, -1],
    threshold=[0.5, 0.5, 0, 0, 0, 0.5, 0, 0],
    left=[1, 2, -1, -1, -1, 1, -1, -1],
    right=[4, 3, -1, -1, -1, 2, -1, -1],
    cover=[4, 2, 1, 1, 2, 4, 1, 3],
    value=[0, 0, 1, 2, 5, 0, 0, 4],
    n_outputs=1,
    base_offset=[0.5],
)
XT = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])


def test_tree_shap_hand_computed():
    result = ex.tree_shap(XT, HAND_TREES)
    assert (result.kind, result.exactness, result.group_aggregation) == ("tree_shap", "exact", "none")
    # Path-dependent values from the cover-weighted conditional expectations.
    np.testing.assert_allclose(
        result.attributions[:, :, 0], [[-1.625, 0.375, -3.0], [1.875, -0.125, 1.0]], atol=1e-12
    )
    np.testing.assert_allclose(result.expected[:, 0], [6.75, 6.75], atol=1e-12)
    np.testing.assert_allclose(result.predictions[:, 0], [2.5, 9.5], atol=1e-12)
    assert result.additivity_max_abs_error < 1e-12
    groups, n_groups, _ = ex.groups_from_ranges(3, [(0, 2)])
    grouped = ex.tree_shap(XT, HAND_TREES, feature_group=groups, n_groups=n_groups)
    assert grouped.group_aggregation == "sum_of_feature_values"
    np.testing.assert_allclose(grouped.attributions[:, :, 0], [[-1.25, -3.0], [1.75, 1.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# Grouping / summary helpers and validation
# ---------------------------------------------------------------------------


def test_grouping_and_summary_helpers():
    groups, n_groups = ex.groups_fixed_width(5, 2)
    assert groups.tolist() == [0, 0, 1, 1, 2] and n_groups == 3
    groups, n_groups, range_of_group = ex.groups_from_ranges(6, [(3, 5), (0, 1)])
    assert groups.tolist() == [1, 2, 2, 0, 0, 3]
    assert n_groups == 4 and range_of_group.tolist() == [0, 1, -1, -1]
    with pytest.raises(N4MError):
        ex.groups_from_ranges(6, [(0, 3), (2, 4)])
    a = np.random.default_rng(0).normal(size=(4, 3, 2))
    mean_abs, mean_signed = ex.summarize(a, output_index=1)
    np.testing.assert_allclose(mean_abs, np.abs(a[:, :, 1]).mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(mean_signed, a[:, :, 1].mean(axis=0), atol=1e-14)
    with pytest.raises(ValueError, match="output_index"):
        ex.summarize(a, output_index=2)


def test_input_validation(linear_case):
    w, b, X, background = linear_case
    model = _linear_model(w, b)
    with pytest.raises(ValueError, match="method"):
        ex.explain(model, X, background=background, method="deep_shap")
    with pytest.raises(ValueError, match="baseline"):
        ex.explain(model, X, background=background, baseline="zero")
    with pytest.raises(ValueError, match="requires background"):
        ex.explain(model, X)
    with pytest.raises(ValueError, match="columns"):
        ex.explain(model, X, background=background[:, :3])
    with pytest.raises(ValueError, match="feature_group"):
        ex.explain(model, X, background=background, feature_group=[0, 1])
    with pytest.raises(N4MError):  # group 1 is empty: not a partition
        ex.explain(model, X, background=background, feature_group=[0, 0, 0, 2, 2, 2], n_groups=3)
    with pytest.raises(ValueError, match="expected"):
        ex.explain(lambda rows: np.zeros(rows.shape[0] + 1), X, background=background, n_outputs=1)
    with pytest.raises(ValueError, match="nonfinite"):
        ex.explain(lambda rows: np.full(rows.shape[0], np.nan), X, background=background, n_outputs=1)
    with pytest.raises(N4MError):  # exact enumeration is limited to 16 groups
        ex.explain(lambda rows: rows.sum(axis=1), np.ones((1, 17)), background=np.zeros((1, 17)),
                   method="exact_shap")
    with pytest.raises(TypeError):
        ex.explain(model, X, background=background, seed=1.5)
    with pytest.raises(ValueError, match="coefficients"):
        ex.linear_shap(X, np.ones(5), reference=np.zeros(6))
    with pytest.raises(ValueError, match="reference"):
        ex.linear_shap(X, w, reference=np.zeros(5))
    bad = ex.TreeEnsemble(**{**HAND_TREES.__dict__, "tree_offsets": [0, 5, 9]})
    with pytest.raises(ValueError, match="tree_offsets"):
        ex.tree_shap(XT, bad)
    bad = ex.TreeEnsemble(**{**HAND_TREES.__dict__, "decision_type": 2})
    with pytest.raises(ValueError, match="decision_type"):
        ex.tree_shap(XT, bad)


# ---------------------------------------------------------------------------
# Optional parity with the shap package
# ---------------------------------------------------------------------------


def _affine_from_predict(predict, n_features):
    """Recover (B, c) of an affine predictor from its own predictions.

    Robust to scikit-learn's changing PLSRegression ``coef_`` orientation and
    intercept conventions across versions.
    """
    zero = np.atleast_2d(np.asarray(predict(np.zeros((1, n_features))), dtype=np.float64))
    eye = np.atleast_2d(np.asarray(predict(np.eye(n_features)), dtype=np.float64))
    if eye.shape[0] != n_features:
        eye = eye.T
    return eye - zero, zero[0]


@pytest.mark.parametrize("estimator", ["ridge", "pls"])
def test_linear_parity_with_shap(estimator):
    shap = pytest.importorskip("shap")
    linear_model = pytest.importorskip("sklearn.linear_model")
    cross_decomposition = pytest.importorskip("sklearn.cross_decomposition")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 8))
    y = X @ rng.normal(size=8) + 0.1 * rng.normal(size=60)
    model = (
        linear_model.Ridge(alpha=0.5).fit(X, y)
        if estimator == "ridge"
        else cross_decomposition.PLSRegression(n_components=3).fit(X, y)
    )
    B, c = _affine_from_predict(model.predict, 8)
    explained, background = X[:5], X[10:50]
    np.testing.assert_allclose(explained @ B + c, np.reshape(model.predict(explained), (5, -1)), atol=1e-10)
    ours = ex.linear_shap(explained, B, reference=background.mean(axis=0), intercept=c)
    reference = shap.LinearExplainer((B[:, 0], c[0]), background)
    np.testing.assert_allclose(ours.attributions[:, :, 0], reference.shap_values(explained), atol=1e-10)
    np.testing.assert_allclose(ours.expected[:, 0], np.ravel(reference.expected_value)[0], atol=1e-10)


def _sklearn_tree_arrays(trees):
    """Concatenate fitted sklearn regression trees into flat local-index arrays (test fixture)."""
    offsets, feature, threshold, left, right, cover, value = [0], [], [], [], [], [], []
    for tree in trees:
        t = tree.tree_
        offsets.append(offsets[-1] + t.node_count)
        feature.append(np.where(t.children_left < 0, -1, t.feature))
        threshold.append(t.threshold)
        left.append(t.children_left)
        right.append(t.children_right)
        cover.append(t.weighted_n_node_samples)
        value.append(t.value[:, 0, 0])
    return dict(
        tree_offsets=offsets,
        feature=np.concatenate(feature),
        threshold=np.concatenate(threshold),
        left=np.concatenate(left),
        right=np.concatenate(right),
        cover=np.concatenate(cover),
        value=np.concatenate(value),
        n_outputs=1,
    )


@pytest.mark.parametrize("kind", ["tree", "forest", "gbr"])
def test_tree_parity_with_shap(kind):
    shap = pytest.importorskip("shap")
    tree_mod = pytest.importorskip("sklearn.tree")
    ensemble_mod = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(5)
    # sklearn compares float32 features; pre-round so both libraries route identically.
    X = rng.normal(size=(200, 6)).astype(np.float32).astype(np.float64)
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.1 * rng.normal(size=200)
    if kind == "tree":
        model = tree_mod.DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y)
        flat = ex.TreeEnsemble(**_sklearn_tree_arrays([model]))
    elif kind == "forest":
        model = ensemble_mod.RandomForestRegressor(n_estimators=12, max_depth=5, random_state=0).fit(X, y)
        flat = ex.TreeEnsemble(**_sklearn_tree_arrays(model.estimators_),
                               tree_scale=1.0 / len(model.estimators_))
    else:
        model = ensemble_mod.GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
        trees = list(model.estimators_[:, 0])
        x0 = X[:1]
        base = model.predict(x0)[0] - model.learning_rate * sum(t.predict(x0)[0] for t in trees)
        flat = ex.TreeEnsemble(**_sklearn_tree_arrays(trees), tree_scale=model.learning_rate,
                               base_offset=[base])
    explained = X[:10]
    ours = ex.tree_shap(explained, flat)
    np.testing.assert_allclose(ours.predictions[:, 0], model.predict(explained), atol=1e-9)
    reference = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    np.testing.assert_allclose(ours.attributions[:, :, 0], reference.shap_values(explained), atol=1e-7)
    np.testing.assert_allclose(ours.expected[:, 0], np.ravel(reference.expected_value)[0], atol=1e-7)


def test_permutation_parity_with_shap_permutation_explainer():
    shap = pytest.importorskip("shap")
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 7))
    model = linear_model.Ridge(alpha=1.0).fit(X, X @ rng.normal(size=7))
    background, explained = X[:20], X[30:34]
    ours = ex.explain(model.predict, explained, background=background, n_permutations=20, seed=0)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    reference = shap.PermutationExplainer(model.predict, masker)(explained, max_evals=20 * (2 * 7 + 1))
    # Permutation SHAP is exact for an additive model, whatever the sampled orders.
    np.testing.assert_allclose(ours.attributions[:, :, 0], reference.values, atol=1e-9)
    np.testing.assert_allclose(ours.expected[:, 0], reference.base_values, atol=1e-9)


def test_exact_and_sampled_parity_with_shap_exact_explainer():
    shap = pytest.importorskip("shap")
    rng = np.random.default_rng(4)

    def f(rows):
        return rows[:, 0] * rows[:, 1] + np.sin(rows[:, 2]) * rows[:, 3] + rows[:, 4] ** 2

    background, explained = rng.normal(size=(5, 5)), rng.normal(size=(3, 5))
    masker = shap.maskers.Independent(background, max_samples=len(background))
    reference = shap.ExactExplainer(f, masker)(explained)
    exact = ex.explain(f, explained, background=background, method="exact_shap")
    np.testing.assert_allclose(exact.attributions[:, :, 0], reference.values, atol=1e-10)
    np.testing.assert_allclose(exact.expected[:, 0], reference.base_values, atol=1e-10)
    # Sampled permutation SHAP converges to the exact values within its own
    # Monte-Carlo standard error (statistical tolerance, fixed seed).
    sampled = ex.explain(f, explained, background=background, n_permutations=200, seed=1)
    tolerance = 5.0 * sampled.std_error[:, :, 0] + 1e-9
    assert np.all(np.abs(sampled.attributions[:, :, 0] - reference.values) <= tolerance)
