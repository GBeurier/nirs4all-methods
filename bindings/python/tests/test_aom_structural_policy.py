from __future__ import annotations

import numpy as np

import n4m
from n4m import (
    AOMControlSelector,
    AOMEndpointMarginStabilityGate,
    AOMFallbackBlendGate,
    AOMMidPEndpointStack,
    AOMOperatorPLSStack,
    AOMOperatorPLSSpec,
    AOMRidgeBlender,
    AOMRobustHPOCompact,
    AOMRobustHPORegressor,
    AOMRobustHPOWide,
    AOMStructuralPolicy,
    AOMStructuralPolicyWithP700BlockLocalAdmission,
    AOMStructuralPolicyWithP700ProtocolUnified,
    AOMStructuralPolicyWithPgt1200Admissions,
    AOMTrueBankEndpointPortfolio,
    build_aom_control_chain_bank,
)
from n4m.sklearn import AOMDatasetMetadata, EndpointStabilityDecision


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, X, y):
        self.n_features_ = np.asarray(X).shape[1]
        self.y_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.full(X.shape[0], self.value + 0.0 * self.y_mean_)


class FeatureRegressor:
    def __init__(self, column: int, scale: float = 1.0) -> None:
        self.column = int(column)
        self.scale = float(scale)

    def fit(self, X, y):
        self.n_features_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.scale * X[:, self.column]


class FitRowRecorder:
    def __init__(self, fit_sets) -> None:
        self.fit_sets = fit_sets

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.fit_ids_ = tuple(int(v) for v in X[:, 0])
        self.fit_sets.append(self.fit_ids_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, 1]


class SelectColumns:
    def __init__(self, columns):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.columns]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def _estimators():
    return {
        "p700_truebank_endpoint_portfolio": ConstantRegressor(1.0),
        "p700_protocol_unified_selector": ConstantRegressor(2.0),
        "p700_block_local_admission": ConstantRegressor(3.0),
        "mid_p_margin_stability_gate": ConstantRegressor(4.0),
        "pgt1200_control_selector": ConstantRegressor(5.0),
        "pgt1200_kernel_rff_admission": ConstantRegressor(6.0),
    }


def test_structural_policy_routes_by_feature_count_only():
    policy = AOMStructuralPolicy(_estimators(), enable_pgt1200_admissions=True)

    assert policy.choose_route(n_features=128) == "p700_truebank_endpoint_portfolio"
    assert policy.choose_route(n_features=900) == "mid_p_margin_stability_gate"
    assert policy.choose_route(n_features=1500) == "pgt1200_kernel_rff_admission"


def test_structural_policy_metadata_does_not_change_route_or_predictions():
    X = np.arange(40, dtype=float).reshape(5, 8)
    y = np.linspace(0.0, 1.0, 5)
    estimators = _estimators()
    meta_a = AOMDatasetMetadata(dataset_id="Beer_OriginalExtract_60_KS", database_name="BEER")
    meta_b = {"dataset_id": "CompletelyDifferent", "database_name": "OTHER"}

    model_a = AOMStructuralPolicy(estimators).fit(X, y, metadata=meta_a)
    model_b = AOMStructuralPolicy(estimators).fit(X, y, metadata=meta_b)

    assert model_a.route_ == model_b.route_ == "p700_truebank_endpoint_portfolio"
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))
    assert model_a.routing_report_["metadata_used_for_routing"] is False
    assert model_b.routing_report_["metadata_used_for_routing"] is False
    assert model_a.routing_report_["dataset_id"] != model_b.routing_report_["dataset_id"]


def test_structural_policy_requires_selected_estimator():
    X = np.arange(20, dtype=float).reshape(5, 4)
    y = np.linspace(0.0, 1.0, 5)
    policy = AOMStructuralPolicy({"mid_p_margin_stability_gate": ConstantRegressor(4.0)})

    try:
        policy.fit(X, y)
    except ValueError as exc:
        assert "missing estimator" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected missing estimator error")


def test_structural_policy_presets_are_source_free():
    estimators = _estimators()

    high_p = AOMStructuralPolicyWithPgt1200Admissions(estimators)
    assert high_p.choose_route(n_features=1500) == "pgt1200_kernel_rff_admission"

    block_local = AOMStructuralPolicyWithP700BlockLocalAdmission(estimators)
    assert block_local.choose_route(n_features=128) == "p700_block_local_admission"
    assert block_local.choose_route(n_features=1500) == "pgt1200_kernel_rff_admission"

    protocol = AOMStructuralPolicyWithP700ProtocolUnified(
        estimators,
        p700_protocol_min_features=100,
    )
    assert protocol.choose_route(n_features=80) == "p700_truebank_endpoint_portfolio"
    assert protocol.choose_route(n_features=128) == "p700_protocol_unified_selector"


def test_legacy_beer_admission_name_is_not_exported():
    assert not hasattr(n4m, "AOMStructuralPolicyWithBlockLocalBeerAdmissions")


def test_control_selector_selects_best_chain_with_train_cv_ridge():
    x0 = np.linspace(0.0, 1.0, 18)
    x1 = np.sin(np.linspace(0.0, 3.0, 18))
    X = np.column_stack([x0, x1])
    y = 2.0 * x0 + 0.25
    selector = AOMControlSelector(
        {
            "signal": SelectColumns([0]),
            "noise": SelectColumns([1]),
        },
        alphas=(1e-8, 1.0),
        cv=3,
        cv_mode="both",
        repeats=2,
        ensemble_top_k=1,
    ).fit(X, y)

    assert selector.selection_report_["selected"].startswith("signal__ridge")
    assert selector.selection_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(selector.predict(X), y, atol=1e-5)


def test_control_selector_chain_subset_and_rank_mean_ensemble():
    x0 = np.linspace(0.0, 1.0, 12)
    X = np.column_stack([x0, 1.0 - x0])
    y = x0
    selector = AOMControlSelector(
        {
            "signal": SelectColumns([0]),
            "unused": SelectColumns([1]),
        },
        chains=("signal",),
        alphas=(1e-8, 1e-7),
        cv=3,
        ensemble_top_k=2,
        ensemble_aggregation="rank_mean",
    ).fit(X, y)

    assert all(name.startswith("signal__ridge") for name in selector.selection_report_["ensemble"])
    assert len(selector.selection_report_["ensemble"]) == 2
    np.testing.assert_allclose(selector.predict(X), y, atol=1e-5)


def test_control_selector_supports_pls_head():
    x0 = np.linspace(0.0, 1.0, 16)
    x1 = np.cos(np.linspace(0.0, 3.0, 16))
    X = np.column_stack([x0, x1])
    y = 1.5 * x0 - 0.2
    selector = AOMControlSelector(
        {
            "signal": SelectColumns([0]),
            "noise": SelectColumns([1]),
        },
        heads=("pls",),
        pls_components=(1, 2),
        cv=4,
    ).fit(X, y)

    assert selector.selection_report_["selected"].startswith("signal__pls")
    assert selector.selection_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(selector.predict(X), y, atol=1e-10)


def test_control_selector_rejects_unknown_head():
    try:
        AOMControlSelector({"signal": SelectColumns([0])}, heads=("svr",))
    except ValueError as exc:
        assert "unknown heads" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected unknown head error")


def test_endpoint_margin_stability_gate_accepts_stable_endpoint():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 12),
        np.linspace(1.0, 0.0, 12),
    ])
    y = X[:, 0]
    gate = AOMEndpointMarginStabilityGate(
        stack_estimator=ConstantRegressor(9.0),
        endpoint_estimators={
            "good": FeatureRegressor(0),
            "bad": FeatureRegressor(1),
        },
        cv=4,
    ).fit(X, y)

    assert gate.decision_ == EndpointStabilityDecision(
        chosen_endpoint="good",
        cv_best_endpoint="good",
        oof_best_endpoint="good",
        accepted_endpoint=True,
        cv_margin_ratio=gate.decision_.cv_margin_ratio,
        oof_margin_ratio=gate.decision_.oof_margin_ratio,
    )
    assert gate.stability_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(gate.predict(X), y)


def test_endpoint_margin_stability_gate_falls_back_to_stack_when_margin_fails():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 8),
        np.linspace(1.0, 0.0, 8),
    ])
    y = X[:, 0]
    gate = AOMEndpointMarginStabilityGate(
        stack_estimator=ConstantRegressor(7.0),
        endpoint_estimators={"only": FeatureRegressor(0)},
        cv=4,
    ).fit(X, y)

    assert gate.decision_.chosen_endpoint == "stack"
    assert gate.decision_.accepted_endpoint is False
    np.testing.assert_allclose(gate.predict(X), np.full(X.shape[0], 7.0))


def test_truebank_endpoint_portfolio_selects_cv_best_without_metadata_route():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 16),
        np.linspace(1.0, 0.0, 16),
    ])
    y = X[:, 0]

    portfolio_a = AOMTrueBankEndpointPortfolio(
        {
            "good": FeatureRegressor(0),
            "bad": FeatureRegressor(1),
        },
        cv=4,
        cv_mode="both",
    ).fit(X, y, metadata=AOMDatasetMetadata(dataset_id="Beer", database_name="BEER"))
    portfolio_b = AOMTrueBankEndpointPortfolio(
        {
            "good": FeatureRegressor(0),
            "bad": FeatureRegressor(1),
        },
        cv=4,
        cv_mode="both",
    ).fit(X, y, metadata={"dataset_id": "Other", "database_name": "OTHER"})

    assert portfolio_a.selected_endpoint_ == portfolio_b.selected_endpoint_ == "good"
    assert portfolio_a.portfolio_report_["metadata_used_for_routing"] is False
    assert portfolio_b.portfolio_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(portfolio_a.predict(X), portfolio_b.predict(X))
    np.testing.assert_allclose(portfolio_a.predict(X), y)


def test_midp_endpoint_stack_learns_pairwise_oof_convex_blend():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 20),
        np.linspace(1.0, 0.0, 20),
    ])
    y = 0.7 * X[:, 0] + 0.3 * X[:, 1]

    stack = AOMMidPEndpointStack(
        {
            "left": FeatureRegressor(0),
            "right": FeatureRegressor(1),
        },
        cv=5,
        max_pair_oof_ratio=10.0,
    ).fit(X, y, metadata={"dataset_id": "Rice", "database_name": "AMYLOSE"})

    assert stack.selected_members_ == ("left", "right")
    assert stack.stack_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(stack.weights_, (0.7, 0.3), atol=1e-12)
    np.testing.assert_allclose(stack.predict(X), y, atol=1e-12)


def test_fallback_blend_gate_accepts_oof_gain_without_metadata_route():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 20),
        np.linspace(1.0, 0.0, 20),
    ])
    y = X[:, 0]

    gate_a = AOMFallbackBlendGate(
        base_estimator=ConstantRegressor(0.5),
        candidate_estimator=FeatureRegressor(0),
        cv=5,
        blend_weights=(0.0, 1.0),
        min_relative_oof_gain=0.01,
    ).fit(X, y, metadata={"dataset_id": "Biscuit", "database_name": "BISCUIT"})
    gate_b = AOMFallbackBlendGate(
        base_estimator=ConstantRegressor(0.5),
        candidate_estimator=FeatureRegressor(0),
        cv=5,
        blend_weights=(0.0, 1.0),
        min_relative_oof_gain=0.01,
    ).fit(X, y, metadata={"dataset_id": "Other", "database_name": "OTHER"})

    assert gate_a.accepted_candidate_ is True
    assert gate_a.candidate_weight_ == 1.0
    assert gate_a.blend_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(gate_a.predict(X), gate_b.predict(X))
    np.testing.assert_allclose(gate_a.predict(X), y)


def test_fallback_blend_gate_keeps_base_when_candidate_is_not_better():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 16),
        np.linspace(1.0, 0.0, 16),
    ])
    y = X[:, 0]
    gate = AOMFallbackBlendGate(
        base_estimator=FeatureRegressor(0),
        candidate_estimator=FeatureRegressor(1),
        cv=4,
        blend_weights=(0.0, 0.5, 1.0),
        min_relative_oof_gain=0.01,
    ).fit(X, y)

    assert gate.accepted_candidate_ is False
    assert gate.candidate_weight_ == 0.0
    np.testing.assert_allclose(gate.predict(X), y)


def test_aom_control_chain_bank_profiles_cover_lab_preprocessings():
    compact = build_aom_control_chain_bank("compact")
    robust = build_aom_control_chain_bank("robust")
    wide = build_aom_control_chain_bank("wide")

    assert {"raw", "snv", "msc", "emsc2", "detrend1", "nw_s5_g5_d1"} <= set(compact)
    assert sum("savgol" in name for name in compact) >= 3
    assert len(compact) < len(robust) < len(wide)
    assert sum("savgol" in name for name in wide) >= 10

    X = np.linspace(0.0, 1.0, 40).reshape(4, 10)
    Xt = compact["snv"].fit_transform(X)
    assert Xt.shape == X.shape


def test_aom_robust_hpo_regressor_is_source_free_and_predicts():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((24, 16))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1]

    kwargs = dict(
        profile="compact",
        chains=("raw", "snv", "detrend1"),
        heads=("ridge", "pls"),
        alphas=(1e-8, 1e-4),
        pls_components=(1, 2),
        cv=3,
    )
    model_a = AOMRobustHPORegressor(**kwargs).fit(
        X,
        y,
        metadata=AOMDatasetMetadata(dataset_id="Beer", database_name="BEER"),
    )
    model_b = AOMRobustHPORegressor(**kwargs).fit(
        X,
        y,
        metadata={"dataset_id": "Other", "database_name": "OTHER"},
    )

    assert model_a.selection_report_["metadata_used_for_routing"] is False
    assert model_a.selection_report_["n_chains"] >= 3
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))
    assert model_a.predict(X).shape == (X.shape[0],)


def test_aom_robust_hpo_presets_are_exported_and_source_free():
    assert issubclass(AOMRobustHPOCompact, AOMRobustHPORegressor)
    assert issubclass(AOMRobustHPOWide, AOMRobustHPORegressor)
    assert AOMRobustHPOCompact(cv=3).profile == "compact"
    assert AOMRobustHPOWide(cv=3).profile == "wide"


def test_aom_ridge_blender_learns_simplex_oof_weights():
    x0 = np.linspace(0.0, 1.0, 24)
    x1 = np.cos(np.linspace(0.0, 2.0, 24))
    X = np.column_stack([x0, x1])
    y = 0.65 * X[:, 0] + 0.35 * X[:, 1]

    blender = AOMRidgeBlender(
        {
            "left": FeatureRegressor(0),
            "right": FeatureRegressor(1),
        },
        cv=4,
        shuffle=False,
        regularizer=0.0,
    ).fit(X, y)

    np.testing.assert_allclose(blender.weights_.sum(), 1.0, atol=1e-12)
    assert np.all(blender.weights_ >= -1e-12)
    assert blender.selected_variant_label_ == "left"
    np.testing.assert_allclose(blender.weights_, [0.65, 0.35], atol=1e-10)
    np.testing.assert_allclose(blender.predict(X), y, atol=1e-10)
    assert blender.blend_report_["metadata_used_for_routing"] is False


def test_aom_ridge_blender_oof_fits_are_fold_local():
    fit_sets = []
    X = np.column_stack([np.arange(12, dtype=float), np.linspace(0.0, 1.0, 12)])
    y = X[:, 1]

    AOMRidgeBlender(
        {"probe": lambda: FitRowRecorder(fit_sets)},
        cv=3,
        shuffle=False,
    ).fit(X, y)

    expected_cv_fit_sets = [
        tuple(range(4, 12)),
        tuple(list(range(0, 4)) + list(range(8, 12))),
        tuple(range(0, 8)),
    ]
    assert fit_sets[:3] == expected_cv_fit_sets
    assert fit_sets[3] == tuple(range(12))


def test_aom_ridge_blender_metadata_is_audit_only():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 20),
        np.linspace(1.0, 0.0, 20),
    ])
    y = X[:, 0]
    kwargs = dict(
        candidates={"good": FeatureRegressor(0), "bad": FeatureRegressor(1)},
        cv=5,
        shuffle=True,
        regularizer=0.01,
    )

    model_a = AOMRidgeBlender(**kwargs).fit(
        X,
        y,
        metadata=AOMDatasetMetadata(dataset_id="Beer", database_name="BEER"),
    )
    model_b = AOMRidgeBlender(**kwargs).fit(
        X,
        y,
        metadata={"dataset_id": "Other", "database_name": "OTHER"},
    )

    assert model_a.blend_report_["metadata_used_for_routing"] is False
    assert model_b.blend_report_["metadata_used_for_routing"] is False
    np.testing.assert_allclose(model_a.weights_, model_b.weights_)
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))


def test_aom_ridge_blender_default_chain_ridge_pool_predicts():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((18, 6))
    y = 1.25 * X[:, 0] - 0.2 * X[:, 1]

    model = AOMRidgeBlender(
        chains=("raw",),
        alphas=(1e-8, 1.0),
        cv=3,
        shuffle=False,
    ).fit(X, y)

    assert all(label.startswith("raw__ridge") for label in model.candidate_labels_)
    assert len(model.candidate_labels_) == 2
    assert model.predict(X).shape == (X.shape[0],)
    assert hasattr(n4m, "AOMRidgeBlender")


def test_aom_ridge_blender_predict_requires_fit():
    try:
        AOMRidgeBlender({"one": FeatureRegressor(0)}).predict(np.zeros((2, 1)))
    except RuntimeError as exc:
        assert "before fit" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected predict-before-fit error")


def test_aom_operator_pls_stack_selects_and_predicts_with_custom_operators():
    rng = np.random.default_rng(55)
    X = rng.standard_normal((36, 4))
    y = 1.7 * X[:, 0] - 0.4 * X[:, 2]
    op0 = np.array([[1.0], [0.0], [0.0], [0.0]])
    op2 = np.array([[0.0], [0.0], [1.0], [0.0]])

    stack = AOMOperatorPLSStack(
        {"x0": op0, "x2": op2},
        components=(1, 2),
        alphas=(1e-8, 1e-2),
        cv=3,
        cv_mode="contiguous",
    ).fit(X, y)

    assert isinstance(stack.selected_spec_, AOMOperatorPLSSpec)
    assert stack.accepted_operator_stack_ is True
    assert stack.stack_report_["metadata_used_for_routing"] is False
    assert stack.transform(X).shape[0] == X.shape[0]
    np.testing.assert_allclose(stack.predict(X), y, atol=1e-5)


def test_aom_operator_pls_stack_baseline_gate_rejects_false_positive():
    X = np.column_stack([
        np.linspace(0.0, 1.0, 24),
        np.linspace(1.0, 0.0, 24),
    ])
    y = X[:, 0]
    noise_only = np.array([[0.0], [1.0]])

    stack = AOMOperatorPLSStack(
        {"noise_only": noise_only},
        components=(1,),
        alphas=(1e-8,),
        baseline_estimator=FeatureRegressor(0),
        min_relative_oof_gain=0.01,
        cv=4,
        cv_mode="contiguous",
    ).fit(X, y)

    assert stack.accepted_operator_stack_ is False
    assert stack.stack_report_["accepted_operator_stack"] is False
    np.testing.assert_allclose(stack.predict(X), y)


def test_aom_operator_pls_stack_metadata_is_audit_only():
    rng = np.random.default_rng(77)
    X = rng.standard_normal((30, 5))
    y = X[:, 0] + 0.2 * X[:, 1]
    op = np.eye(5)
    kwargs = dict(
        operator_bank={"raw": op},
        components=(1, 2),
        alphas=(1e-6, 1.0),
        cv=3,
        cv_mode="shuffled",
        random_state=42,
    )

    model_a = AOMOperatorPLSStack(**kwargs).fit(
        X,
        y,
        metadata=AOMDatasetMetadata(dataset_id="Beer", database_name="BEER"),
    )
    model_b = AOMOperatorPLSStack(**kwargs).fit(
        X,
        y,
        metadata={"dataset_id": "Other", "database_name": "OTHER"},
    )

    assert model_a.stack_report_["metadata_used_for_routing"] is False
    assert model_b.stack_report_["metadata_used_for_routing"] is False
    assert model_a.selected_spec_ == model_b.selected_spec_
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))


def test_aom_operator_pls_stack_default_operator_bank_smoke():
    rng = np.random.default_rng(88)
    X = rng.standard_normal((20, 12))
    y = X[:, 0] - X[:, 3]

    stack = AOMOperatorPLSStack(
        components=(1,),
        alphas=(1e-4,),
        cv=2,
        cv_mode="contiguous",
    ).fit(X, y)

    assert "raw" in stack.operator_names_
    assert len(stack.operator_names_) >= 3
    assert stack.predict(X).shape == (X.shape[0],)
    assert hasattr(n4m, "AOMOperatorPLSStack")


def test_aom_operator_pls_stack_predict_requires_fit():
    try:
        AOMOperatorPLSStack({"raw": np.eye(2)}).predict(np.zeros((2, 2)))
    except RuntimeError as exc:
        assert "before fit" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected predict-before-fit error")
