# SPDX-License-Identifier: CECILL-2.1
"""Independent numerical and fold-local controls for the spectral encoders."""

import copy
import pickle

import numpy as np
import pytest
from n4m._errors import N4MError
from n4m.decomposition import GCU, LVSE
from n4m.model_selection.aom_calibration import (
    AOMPLSRegressor,
    AOMRidgeRegressor,
    FastAOMPLSRegressor,
    FastAOMRidgeRegressor,
)
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge


@pytest.mark.parametrize("cls", [LVSE, GCU])
def test_fitted_handle_cannot_be_copied_or_pickled(cls):
    model = cls(rank=1)
    for copier in (
        copy.copy,
        copy.deepcopy,
        lambda value: pickle.loads(pickle.dumps(value)),
    ):
        assert copier(model).get_params() == model.get_params()
    x = np.random.default_rng(93).normal(size=(12, 8))
    model.fit(x)
    expected = model.transform(x)
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="process-local native handle"):
            copier(model)
        np.testing.assert_array_equal(model.transform(x), expected)
    assert clone(model).get_params() == model.get_params()
    model.close()


def test_lvse_affine_and_independent_local_subspaces():
    rng = np.random.default_rng(81)
    x, query = rng.normal(size=(45, 19)), rng.normal(size=(8, 19))
    model = LVSE(width=8, rank=2).fit(np.asfortranarray(x))
    operator, offset = model.export_linear_operator()
    encoded = model.transform(query)
    assert_allclose(encoded, query @ operator.T + offset, atol=2e-14)
    scale = np.maximum(x.std(0), 1e-8)
    reference = []
    for start in range(0, 19, 8):
        sl = slice(start, min(start + 8, 19))
        _, _, vt = np.linalg.svd(
            (x[:, sl] - x[:, sl].mean(0)) / scale[sl], full_matrices=False
        )
        reference.append(((query[:, sl] - x[:, sl].mean(0)) / scale[sl]) @ vt[:2].T)
    reference = np.hstack(reference)
    # SVD signs are arbitrary; inner products are invariant.
    assert_allclose(encoded @ encoded.T, reference @ reference.T, rtol=1e-9, atol=1e-10)
    assert clone(model).get_params() == model.get_params()


def test_lvse_overlap_snv_and_failed_refit():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(30, 17))
    model = LVSE(width=8, rank=2, overlap=0.5, snv=True).fit(x)
    before = model.transform(x)
    assert before.shape == (30, 8)
    with pytest.raises(ValueError, match="affine"):
        model.export_linear_operator()
    with pytest.raises(N4MError):
        model.fit(np.full_like(x, np.nan))
    assert_allclose(model.transform(x), before)
    assert model.transform(x[:0]).shape == (0, 8)
    model.close()
    with pytest.raises(RuntimeError):
        model.transform(x)


def test_gcu_matches_nmf_rank_one_and_fixed_query_basis():
    # Rank one avoids approximate/randomized initialization differences.
    rng = np.random.default_rng(9)
    x = rng.normal(size=(30, 12))
    query = rng.normal(size=(7, 12))
    model = GCU(rank=1, max_iter=100, tol=1e-8).fit(x)
    positive = np.maximum((x - x.min(0)) / x.std(), 0)
    reference = NMF(
        n_components=1,
        init="nndsvda",
        solver="cd",
        max_iter=100,
        tol=1e-8,
        random_state=0,
    ).fit(positive)
    assert_allclose(
        model.transform(query),
        reference.transform(np.maximum((query - x.min(0)) / x.std(), 0)),
        rtol=2e-5,
        atol=2e-6,
    )
    assert np.all(model.transform(query) >= 0)
    one = model.transform(query[:1])
    assert_allclose(
        model.transform(np.vstack((query[:1], query[1:] * 100)))[:1],
        one,
        rtol=2e-5,
        atol=2e-6,
    )


def test_aom_lvse_ridge_cv_is_fold_local_and_folds_to_input():
    class IdentityLVSE(AOMRidgeRegressor):
        def _candidate_chains(self):
            return (
                (("identity", ("identity", ())), ("lvse", ("lvse", (8, 2, 0.0, 1)))),
            )

    rng = np.random.default_rng(23)
    x = rng.normal(size=(39, 17))
    x[-13:, :5] *= 4
    y = x[:, 0] - x[:, 8] + rng.normal(size=39) * 0.1
    ids = np.repeat(np.arange(3), 13)
    alphas = (0.1, 1.0, 10.0)
    model = IdentityLVSE(branches=("raw",), alphas=alphas, fold_ids=ids).fit(x, y)
    scores = np.zeros(3)
    for f in range(3):
        train, valid = ids != f, ids == f
        encoder = LVSE(width=8, rank=2).fit(x[train])
        z, v = encoder.transform(x[train]), encoder.transform(x[valid])
        for a, alpha in enumerate(alphas):
            fit = Ridge(alpha=alpha).fit(z, y[train])
            scores[a] += np.sqrt(np.mean((fit.predict(v) - y[valid]) ** 2)) / 3
    assert_allclose(model.cv_scores_.ravel(), scores, rtol=1e-8, atol=1e-9)
    encoder = LVSE(width=8, rank=2).fit(x)
    reference = Ridge(alpha=alphas[np.argmin(scores)]).fit(encoder.transform(x), y)
    query = rng.normal(size=(11, 17))
    assert_allclose(
        model.predict(query),
        reference.predict(encoder.transform(query)),
        rtol=1e-8,
        atol=1e-9,
    )
    assert_allclose(
        model.predict(query), query @ model.coef_ + model.intercept_, atol=1e-12
    )


@pytest.mark.parametrize(
    "cls",
    [AOMPLSRegressor, AOMRidgeRegressor, FastAOMPLSRegressor, FastAOMRidgeRegressor],
)
@pytest.mark.parametrize("overlap,standardize", [(0.0, True), (0.5, False)])
def test_lvse_bank_all_calibration_heads_and_nested_rank_cache(
    cls, overlap, standardize
):
    rng = np.random.default_rng(40)
    x = rng.normal(size=(30, 19))
    if not standardize:
        x[:, -1] = 1  # Include a constant channel in the overlapping tail.
    y = x[:, 0] + 0.2 * rng.normal(size=30)
    ids = np.arange(30, dtype=np.int32) % 3
    kwargs = {
        "branches": ("raw",),
        "max_components": 3,
        "alphas": (0.1, 1.0),
        "fold_ids": ids,
    }
    model = cls(lvse=[(8, k, overlap, standardize) for k in (1, 2, 4)], **kwargs).fit(
        x, y
    )
    query = rng.normal(size=(7, 19))
    assert_allclose(
        model.predict(query), query @ model.coef_ + model.intercept_, atol=1e-12
    )
    if not model._fast:
        # Same candidates with and without maximal-rank reuse must give the
        # same scores, including the final short (three-channel) block.
        single = cls(lvse=[(8, 1, overlap, standardize)], **kwargs).fit(x, y)
        assert_allclose(model.cv_scores_[:, :10], single.cv_scores_[:, :10], atol=1e-10)
        assert_allclose(
            model.cv_scores_[:, 10::3], single.cv_scores_[:, 10:], atol=1e-8
        )


@pytest.mark.parametrize(
    "kwargs", [{"width": 0}, {"rank": 0}, {"overlap": 1}, {"overlap": np.nan}]
)
def test_lvse_rejects_invalid_configuration(kwargs):
    with pytest.raises(N4MError):
        LVSE(**kwargs).fit(np.ones((5, 8)))


def test_aom_rejects_invalid_lvse_terminal_before_search():
    x = np.random.default_rng(913).normal(size=(18, 12))
    y = x[:, 0]
    with pytest.raises(ValueError, match="invalid LVSE configuration"):
        AOMPLSRegressor(lvse=[(1, 1, 0.0), (8, 2, 0.0)]).fit(x, y)
    with pytest.raises(ValueError, match="invalid LVSE configuration"):
        AOMPLSRegressor(lvse=[(8, 2, 0.0)]).fit(x[:, :1], y)


@pytest.mark.parametrize("rows", [12, 70])
def test_rank_deficient_windows_zero_unreliable_gram_modes(rows):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(rows, 2)) @ rng.normal(size=(2, 64))
    x += 1e-10 * rng.normal(size=x.shape)
    model = LVSE(width=64, rank=4, standardize=False).fit(x)
    a, _ = model.export_linear_operator()
    np.testing.assert_allclose(a[2:], 0, atol=1e-12)
    projector = a.T @ a
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-9)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    np.testing.assert_allclose(projector, vt[:2].T @ vt[:2], atol=1e-9)


@pytest.mark.parametrize("shape", [(30, 12), (12, 30)])
def test_gcu_multirank_matches_independent_exact_nndsvda_and_sklearn_cd(shape):
    rng = np.random.default_rng(971)
    x, query = rng.normal(size=shape), rng.normal(size=(6, shape[1]))
    z = np.maximum((x - x.min(0)) / x.std(), 0)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    w, h = np.zeros((len(x), 4)), np.zeros((4, shape[1]))
    w[:, 0], h[0] = np.sqrt(s[0]) * abs(u[:, 0]), np.sqrt(s[0]) * abs(vt[0])
    for a in range(1, 4):
        candidates = []
        for sign in (1, -1):
            left, right = np.maximum(sign * u[:, a], 0), np.maximum(sign * vt[a], 0)
            ln, rn = np.linalg.norm(left), np.linalg.norm(right)
            candidates.append((ln * rn, left / ln, right / rn))
        magnitude, left, right = max(candidates, key=lambda item: item[0])
        w[:, a], h[a] = (
            np.sqrt(s[a] * magnitude) * left,
            np.sqrt(s[a] * magnitude) * right,
        )
    w[w < 1e-6], h[h < 1e-6] = z.mean(), z.mean()
    ref = NMF(n_components=4, init="custom", max_iter=60, tol=1e-3)
    ref.fit_transform(z, W=w, H=h)
    model = GCU(rank=4).fit(x)
    assert_allclose(
        model.transform(query),
        ref.transform(np.maximum((query - x.min(0)) / x.std(), 0)),
        rtol=1e-8,
        atol=1e-9,
    )


def test_fast_lvse_screen_matches_explicit_fold_local_covariance_ratio():
    class IdentityBank(FastAOMRidgeRegressor):
        def _candidate_chains(self):
            return ((("identity", ("identity", ())),),) + tuple(
                (("identity", ("identity", ())), (f"lvse{k}", ("lvse", (8, k, 0.5, 0))))
                for k in (1, 2, 4)
            )

    rng = np.random.default_rng(36)
    x = rng.normal(size=(39, 19))
    y = x[:, 0] + 0.2 * rng.normal(size=39)
    ids = np.arange(len(x), dtype=np.int32) % 3
    model = IdentityBank(
        branches=("raw",), alphas=(0.1, 1), fold_ids=ids, rank=100
    ).fit(x, y)

    def candidates(train, query):
        values = [(train - train.mean(0), query - train.mean(0))]
        for k in (1, 2, 4):
            encoder = LVSE(width=8, rank=k, overlap=0.5, standardize=False).fit(train)
            values.append((encoder.transform(train), encoder.transform(query)))
        return values

    def select(values, target):
        yc = target - target.mean()
        return np.argmax(
            [
                np.sum((z.T @ yc) ** 2) / (np.sum(z * z) * (yc @ yc) + 1e-12)
                for z, _ in values
            ]
        )

    values = candidates(x, x)
    assert model.selected_chain_index_ == select(values, y)
    expected = np.zeros(2)
    for f in range(3):
        train, valid = ids != f, ids == f
        values = candidates(x[train], x[valid])
        z, v = values[select(values, y[train])]
        for j, alpha in enumerate((0.1, 1)):
            pred = Ridge(alpha=alpha).fit(z, y[train]).predict(v)
            expected[j] += np.sqrt(np.mean((pred - y[valid]) ** 2)) / 3
    assert_allclose(model.cv_scores_.ravel(), expected, atol=1e-9)
