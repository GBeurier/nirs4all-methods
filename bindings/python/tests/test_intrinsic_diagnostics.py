"""Independent scientific witnesses for the shared native descriptive protocols."""

import json

import numpy as np
import pytest
from n4m._errors import N4MError
from n4m.roles import method_class
from numpy.testing import assert_allclose, assert_array_equal


def run(name, x, target=None, **params):
    inputs = {} if target is None else {"X_target": target}
    return method_class("diagnostics." + name)(**params).run(
        np.asarray(x, dtype=float), **inputs
    )


def values(result, key):
    return np.asarray(result[key + "_values"]).ravel()


def robust_count(v, threshold=8):
    med = np.median(v)
    scale = 1.4826 * np.median(abs(v - med))
    if scale == 0:
        scale = np.std(v, ddof=1)
    return np.sum(abs(v - med) > threshold * scale) if scale > 0 else 0


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_reference_affine_and_angles_against_independent_ols(layout):
    r = np.array([0.2, 0.8, 1.4, 2.3, 3.7, 5.1])
    x = np.array(
        [r, 2 * r + 3, 0.7 * r - 1, r + np.array([0, 0.1, -0.2, 0.05, 0, 0.01])]
    )
    if layout == "F":
        x = np.asfortranarray(x)
    if layout == "strided":
        padded = np.zeros((len(x), 2 * x.shape[1]))
        padded[:, ::2] = x
        x = padded[:, ::2]
    out = run("spectral_reference_qc", x, r[None, :], reference_mode="supplied")
    coeff = np.linalg.lstsq(np.c_[np.ones(len(r)), r], x.T, rcond=None)[0]
    expected = np.sqrt(np.mean((x - r) ** 2, axis=1))
    assert_allclose(values(out, "rms"), expected, atol=1e-14)
    assert_allclose(values(out, "affine_offset"), coeff[0], atol=1e-14)
    assert_allclose(values(out, "affine_gain"), coeff[1], atol=1e-14)
    residual = x - coeff[0, :, None] - coeff[1, :, None] * r
    assert_allclose(
        values(out, "affine_residual_rms"),
        np.sqrt(np.mean(residual**2, axis=1)),
        atol=1e-14,
    )
    assert_allclose(out["rms_p95"], np.quantile(expected, 0.95), atol=1e-14)
    assert_allclose(
        out["affine_abs_offset_p95"], np.quantile(abs(coeff[0]), 0.95), atol=1e-14
    )
    angle = np.arccos(
        np.clip((x @ r) / np.linalg.norm(x, axis=1) / np.linalg.norm(r), -1, 1)
    )
    assert_allclose(values(out, "sam"), angle, atol=3e-8)


def test_reference_masks_and_sparse_availability_are_not_imputation():
    x = np.array([[1, 2, 4], [0, 0, 0], [5, 0, 0]], dtype=float)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
    out = run(
        "spectral_reference_qc",
        x,
        np.array([[1.0, 2, 4]]),
        reference_mode="supplied",
        valid_mask=mask.ravel().tolist(),
    )
    assert_array_equal(out["sam_status"], [0, 2, 1])
    assert_array_equal(out["sam_rows"], [0])
    assert_array_equal(out["common_valid_counts"], [3, 3, 1])
    assert_allclose(values(out, "rms"), [0, np.sqrt(7), 4])
    json.dumps({k: np.asarray(v).tolist() for k, v in out.items()}, allow_nan=False)


def test_dataset_mean_is_native_and_all_missing_remains_unavailable():
    x = np.array([[1.0, 0, 3], [5, 0, 7]])
    out = run("spectral_reference_qc", x, valid_mask=[1, 0, 1, 1, 0, 1])
    assert_allclose(out["reference"], [[3, 0, 5]])
    assert_array_equal(out["reference_counts"], [2, 0, 2])
    assert_array_equal(out["reference_valid_mask"], [1, 0, 1])
    empty = run("spectral_reference_qc", x, valid_mask=[0] * 6)
    assert_array_equal(empty["rms_status"], [1, 1])
    assert "rms_values" not in empty and "rms_median" not in empty


def test_events_complete_legacy_parity_and_native_pooled_denominators():
    rng = np.random.default_rng(29)
    x = rng.normal(0, 0.01, (4, 256))
    x[0, 100] += 8
    x[1, 100:] += 3
    out = run("spectral_events_qc", x)
    jc = np.array([robust_count(v) for v in np.diff(x, axis=1)])
    sc = np.array([robust_count(v) for v in np.diff(x, n=2, axis=1)])
    assert jc.sum() > 0 and sc.sum() > 0
    assert_array_equal(values(out, "jump_count"), jc)
    assert_array_equal(values(out, "spike_count"), sc)
    assert out["global_jump_count"] == jc.sum()
    assert out["global_spike_count"] == sc.sum()
    assert_allclose(out["global_jump_fraction"], jc.sum() / np.diff(x, axis=1).size)
    assert_allclose(
        out["global_spike_fraction"], sc.sum() / np.diff(x, n=2, axis=1).size
    )
    assert_array_equal(out["declared_clip_count_status"], [5] * 4)
    assert "global_declared_clip_fraction" not in out


@pytest.mark.parametrize("descending", [False, True])
def test_physical_nonuniform_axis_is_not_silently_an_index_axis(descending):
    axis = np.geomspace(350, 2500, 80)
    x = np.array([axis * 3 + 5, axis * 0.7 - 2])
    if descending:
        axis = axis[::-1].copy()
        x = x[:, ::-1].copy()
    out = run("spectral_events_qc", x, axis_basis="physical", axis_values=axis.tolist())
    assert_allclose(values(out, "jump_count"), 0)
    assert_allclose(values(out, "spike_count"), 0)
    with pytest.raises((N4MError, ValueError)):
        run("spectral_events_qc", x, axis_basis="physical", axis_values=[1.0] * 80)


def test_sensor_clipping_is_distinct_from_observed_extrema_and_masks():
    x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]])
    out = run("spectral_events_qc", x, clip_lower=0, clip_upper=1)
    assert out["global_declared_clip_count"] == 0
    assert out["observed_extrema_occupancy_count"] == 2
    assert out["observed_extrema_occupancy_n"] == 10
    mask = [1] * 10
    mask[-1] = 0
    out = run("spectral_events_qc", x, clip_upper=0.5, valid_mask=mask)
    assert out["global_declared_clip_count"] == 2
    assert out["global_declared_clip_eligible"] == 9
    assert_allclose(out["global_declared_clip_fraction"], 2 / 9)
    with pytest.raises((N4MError, ValueError)):
        run("spectral_events_qc", x, clip_lower=1, clip_upper=0)


def test_masked_events_do_not_join_nonadjacent_bands():
    x = np.tile(np.arange(8.0), (2, 1))
    mask = np.ones(x.shape, dtype=int)
    mask[0, 3:5] = 0
    out = run("spectral_events_qc", x, valid_mask=mask.ravel().tolist())
    assert_array_equal(out["jump_eligible_counts"], [4, 7])
    assert_array_equal(out["spike_eligible_counts"], [2, 6])
    assert_array_equal(out["spike_count_status"], [1, 0])
    assert out["global_spike_eligible"] == 6


def test_noise_proxy_exact_mad_normalization_and_db():
    rng = np.random.default_rng(39)
    x = 2 + rng.normal(0, 0.02, (10, 60))
    second = np.diff(x, n=2, axis=1)
    sigma = (
        np.median(abs(second - np.median(second, axis=1)[:, None]), axis=1)
        * 1.4826
        / np.sqrt(6)
    )
    out = run("spectral_noise_proxy", x)
    assert_allclose(values(out, "noise_sigma"), sigma, rtol=1e-13)
    assert_allclose(out["global_noise_sigma"], np.median(sigma), rtol=1e-13)
    ratio = np.mean(abs(x)) / np.median(sigma)
    assert_allclose(out["global_signal_noise_ratio"], ratio, rtol=1e-13)
    assert_allclose(out["global_signal_noise_db"], 20 * np.log10(ratio), rtol=1e-13)
    smooth = run("spectral_noise_proxy", np.tile(np.arange(10.0), (3, 1)))
    assert smooth["global_noise_sigma"] == 0
    assert_array_equal(smooth["global_signal_noise_ratio_status"], [4])
    assert "global_signal_noise_ratio" not in smooth


@pytest.mark.parametrize("novelty", [False, True])
def test_true_lof_reuses_native_kernel_and_matches_independent_sklearn(novelty):
    from sklearn.neighbors import LocalOutlierFactor

    rng = np.random.default_rng(93)
    train = rng.normal(size=(40, 4))
    query = rng.normal(size=(7, 4)) * 3
    oracle = LocalOutlierFactor(n_neighbors=7, novelty=novelty).fit(train)
    out = run(
        "local_density",
        query if novelty else train,
        train if novelty else None,
        mode="novelty" if novelty else "training",
        n_neighbors=7,
    )
    expected = (
        -oracle.score_samples(query) if novelty else -oracle.negative_outlier_factor_
    )
    assert_allclose(values(out, "lof"), expected, rtol=1e-9, atol=1e-9)
    assert out["effective_neighbors"] == 7
    degenerate = run("local_density", np.ones((5, 3)))
    assert "lof_values" not in degenerate
    assert_array_equal(degenerate["lof_status"], [4] * 5)


def test_frozen_pca_reference_exports_no_truncated_full_moments():
    rng = np.random.default_rng(12)
    train = rng.normal(size=(20, 8))
    query = rng.normal(size=(5, 8))
    first = run("pca_reference_qc", query, train, n_components=3)
    changed = run("pca_reference_qc", query * 100 + 7, train, n_components=3)
    for key in [
        "reference_mean",
        "loadings",
        "eigenvalues",
        "reference_scores",
        "reference_q_values",
        "reference_t2_values",
    ]:
        assert_array_equal(first[key], changed[key])
    centered = train - train.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    eigen = s[:3] ** 2 / (len(train) - 1)
    scores = (query - train.mean(axis=0)) @ vt[:3].T
    q = np.sum((query - train.mean(axis=0) - scores @ vt[:3]) ** 2, axis=1)
    assert_allclose(first["eigenvalues"].ravel(), eigen, rtol=1e-10)
    assert_allclose(values(first, "pca_q"), q, rtol=1e-10, atol=1e-12)
    assert_allclose(
        values(first, "pca_t2"), np.sum(scores**2 / eigen, axis=1), rtol=1e-10
    )
    assert_array_equal(first["exact_full_moments_status"], [6])
    assert not any("moment" in k and k != "exact_full_moments_status" for k in first)
    json.dumps({k: np.asarray(v).tolist() for k, v in first.items()}, allow_nan=False)


def test_zero_rank_pca_publishes_q_and_explicit_unavailable_t2():
    out = run("pca_reference_qc", np.ones((2, 5)) * 2, np.ones((10, 5)), n_components=3)
    assert out["effective_components"] == 0
    assert_allclose(values(out, "pca_q"), 5)
    assert_array_equal(out["pca_t2_status"], [3, 3])
    assert "pca_t2_values" not in out


@pytest.mark.parametrize(
    "name",
    [
        "spectral_reference_qc",
        "spectral_noise_proxy",
        "spectral_events_qc",
        "local_density",
    ],
)
def test_nonfinite_values_are_rejected_without_silent_imputation(name):
    with pytest.raises((N4MError, ValueError)):
        run(name, [[1, 2, 3, 4, 5], [1, np.nan, 3, 4, 5]])


def test_pca_budget_refusal_is_actionable_and_distinct_from_missing_moments():
    with pytest.raises(N4MError, match="whole-computation budget.*reduce"):
        run(
            "pca_reference_qc", np.zeros((1, 900)), np.zeros((900, 900)), n_components=5
        )


def test_pca_rejects_nonfinite_reference_and_implicit_mask_imputation():
    train = np.arange(80.0).reshape(20, 4)
    bad = train.copy()
    bad[0, 1] = np.nan
    with pytest.raises(N4MError, match="reference must be finite"):
        run("pca_reference_qc", train[:2], bad, n_components=2)
    with pytest.raises(N4MError, match="missingness policy"):
        run(
            "pca_reference_qc",
            train[:2],
            train,
            n_components=2,
            valid_mask=[1, 0, 1, 1, 1, 1, 1, 1],
        )
