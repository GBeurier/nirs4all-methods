# SPDX-License-Identifier: CECILL-2.1
"""
Frozen NumPy reference for the Phase 15 augmenters (`c4a_aug_*`).

Validated once against the corresponding operators in
``nirs4all/operators/augmentation/{spectral,synthesis}.py`` (nirs4all 0.8.x).
After validation these modules become the canonical parity floor for
chemometrics4all's `c4a_aug_*` ABI category - nirs4all itself is no longer
in the build path of the parity fixture generator.

Phase 15 (7 augmenters):
  - GaussianAdditiveNoise         (c4a_aug_gaussian_noise_*)
  - MultiplicativeNoise           (c4a_aug_multiplicative_noise_*)
  - SpikeNoise                    (c4a_aug_spike_noise_*)
  - HeteroscedasticNoiseAugmenter (c4a_aug_hetero_noise_*)
  - LinearBaselineDrift           (c4a_aug_linear_drift_*)
  - PolynomialBaselineDrift       (c4a_aug_poly_drift_*)
  - PathLengthAugmenter           (c4a_aug_path_length_*)

RNG contract (locked in ``roadmap/phase-15-18-augmenters-abi-contract.md``):
each reference function takes an ``np.random.Generator`` seeded with PCG64
(``np.random.default_rng(seed)``) and consumes the stream via
``standard_normal`` and ``random`` primitives only - these have bit-exact
C counterparts in ``c4a_pcg64_engine_standard_normal_fill`` /
``c4a_pcg64_engine_next_double``. The uniform / integer arithmetic that
nirs4all itself uses via ``rng.uniform`` / ``rng.integers`` is unrolled to
explicit ``floor(u * range)`` mappings so the C side can match without
implementing NumPy's Lemire rejection.
"""

from .gaussian_noise import gaussian_noise_apply
from .hetero_noise import hetero_noise_apply
from .linear_drift import linear_drift_apply
from .multiplicative_noise import multiplicative_noise_apply
from .path_length import path_length_apply
from .poly_drift import poly_drift_apply
from .spike_noise import spike_noise_apply

__all__ = [
    "gaussian_noise_apply",
    "hetero_noise_apply",
    "linear_drift_apply",
    "multiplicative_noise_apply",
    "path_length_apply",
    "poly_drift_apply",
    "spike_noise_apply",
]
