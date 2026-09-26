# SPDX-License-Identifier: CECILL-2.1
"""Seeded X-only native augmenter parity with R/WASM ABI-2.11 oracle."""

import numpy as np
import pytest

from n4m.augmentation.noise import GaussianAdditiveNoise
from n4m.augmentation.scattering import ScatterSimulationMSC
from n4m.augmentation.spectral import BandMasking
from n4m.augmentation.splines import SplineXPerturbationAugmenter


@pytest.mark.parametrize(
    ("augmenter", "sum_expected", "sumsq_expected", "prefix"),
    [
        (GaussianAdditiveNoise(sigma=0.03, seed=42),
         335.29953033891428, 447.93442481713896,
         [1.0016880886187998, 1.0142386382315953,
          1.0441573912549518, 1.0652105927054174]),
        (BandMasking(n_bands_lo=1, n_bands_hi=1, bw_lo=2, bw_hi=4,
                     mode="zero", seed=42),
         313.51999999999998, 417.7072, [1.0, 1.02, 1.04, 1.06]),
        (ScatterSimulationMSC(a_low=0.9, a_high=1.1,
                              b_low=-0.01, b_high=0.01, seed=42),
         264.88033265406824, 274.76085353952521,
         [1.0473534823647035, 1.0472047278177739,
          1.0470559732708440, 1.0469072187239143]),
        (SplineXPerturbationAugmenter(spline_degree=3,
                                      perturbation_density=0.2,
                                      perturbation_range_min=-0.1,
                                      perturbation_range_max=0.1, seed=42),
         335.32596744464371, 447.94217484919568,
         [0.99888977536385415, 1.01915259930947943,
          1.03941542325510450, 1.05967824720072956]),
    ],
)
def test_native_augmenter_matches_r_wasm_seeded_oracle(
    augmenter, sum_expected, sumsq_expected, prefix,
):
    x = np.tile(1.0 + 0.02 * np.arange(32), (8, 1))
    original = x.copy()
    out = augmenter.transform(x)
    assert out.shape == x.shape
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(np.sum(out), sum_expected, rtol=1e-12)
    np.testing.assert_allclose(np.sum(out * out), sumsq_expected, rtol=1e-12)
    np.testing.assert_allclose(out.ravel()[:4], prefix, rtol=1e-12)
