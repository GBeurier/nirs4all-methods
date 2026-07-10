# SPDX-License-Identifier: CECILL-2.1
"""Tier-A parity test: the native Sobol sampler must be bit-identical to
scipy.stats.qmc.Sobol(scramble=False).

Run:  N4M_LIB_PATH=/path/to/libn4m.so.2.1.0 PYTHONPATH=bindings/python/src \
      python -m pytest bindings/python/tests/test_sobol_parity.py -q
Skipped if scipy is not installed.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
qmc = pytest.importorskip("scipy.stats").qmc

from n4m.model_selection.optimizer import Optimizer, Sampler, SearchSpace


def _native_sobol(dim: int, n: int) -> "np.ndarray":
    space = SearchSpace()
    for j in range(dim):
        space.add_float(f"x{j}", 0.0, 1.0)  # unit coord passes straight through
    opt = Optimizer(space, sampler=Sampler.SOBOL, seed=0)
    pts = np.empty((n, dim))
    for i in range(n):
        t = opt.ask()
        for j in range(dim):
            pts[i, j] = t.get_float(f"x{j}")
        opt.tell(t.id, 0.0)
    return pts


@pytest.mark.parametrize("dim,n", [(1, 16), (3, 32), (6, 32), (10, 16)])
def test_sobol_matches_scipy(dim, n):
    mine = _native_sobol(dim, n)
    ref = qmc.Sobol(d=dim, scramble=False).random(n)
    assert np.array_equal(mine, ref), (
        f"native Sobol diverges from scipy at d={dim} "
        f"(max abs diff {np.max(np.abs(mine - ref)):.3e})"
    )


if __name__ == "__main__":
    for d, n in [(1, 16), (3, 32), (6, 32), (10, 16)]:
        m = _native_sobol(d, n)
        r = qmc.Sobol(d=d, scramble=False).random(n)
        assert np.array_equal(m, r), f"d={d} mismatch"
    print("sobol parity vs scipy: OK")
