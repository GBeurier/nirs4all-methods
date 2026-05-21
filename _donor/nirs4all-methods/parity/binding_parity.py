# SPDX-License-Identifier: CECILL-2.1
"""Gate 1 comparator: did the binding produce the same numbers as cpp?

This module mirrors :mod:`pls4all.benchmarks.parity_timing._comparator` so
that both projects use the same comparator contract:

  * **Binding parity** (this module, Gate 1) — every language binding
    (python, R, MATLAB, JS/WASM) must produce bit-identical outputs to
    libn4m (the C kernel reference). Tolerance: ``max|pred - ref| <=
    tolerance`` with default ``1e-6``. This gate is exercised once a
    binding ships; until then it is a no-op stub.

  * **Reference parity** (:mod:`parity.reference_parity`, Gate 2) —
    libn4m's algorithms must agree with the canonical external reference
    (numpy / scipy / sklearn / pywt / pybaselines / nirs4all) within a
    per-op tolerance recorded in ``parity/tolerances.md``. This gate is
    the existing Stage 2 of ``parity/python_generator/scripts/run_parity_gate.py``.

The two comparators are deliberately small and pure-numpy so they can be
shared by the runner and the per-binding orchestrators without pulling in
the rest of the parity-gate plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BindingParityResult:
    """Bit-exact agreement between two prediction arrays.

    Attributes
    ----------
    ok
        ``True`` iff ``shape match AND max_abs_diff <= tolerance``. NaN
        in either side always sets ``ok = False``.
    max_abs_diff
        Maximum element-wise absolute difference. ``inf`` on shape
        mismatch, ``nan`` if either input contains NaN.
    max_rel_diff
        Maximum element-wise relative difference,
        ``max_abs_diff / max(|ref|, 1e-300)``. Mirrors the same special
        cases as ``max_abs_diff``.
    shape_mismatch
        True when ``pred.shape != ref.shape``. The comparator does **not**
        attempt to reshape — bindings must produce arrays in the same
        shape as libn4m.
    detail
        Short human-readable summary suitable for log lines.
    """

    ok: bool
    max_abs_diff: float
    max_rel_diff: float
    shape_mismatch: bool
    detail: str


def binding_parity(pred: np.ndarray, ref: np.ndarray,
                   tolerance: float = 1e-6) -> BindingParityResult:
    """Bit-exact comparator: ``max|pred - ref| <= tolerance``.

    Default tolerance ``1e-6`` matches pls4all's binding-consistency
    contract. Use ``tolerance=0.0`` for strict bit-equality on
    deterministic ops where the binding is a thin ctypes wrapper that
    forwards the libn4m result verbatim.

    Parameters
    ----------
    pred
        Predicted array (from the binding under test).
    ref
        Reference array (from libn4m).
    tolerance
        Maximum allowed absolute difference. Inclusive.

    Returns
    -------
    BindingParityResult
        See the dataclass docstring.
    """
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)

    if p.shape != r.shape:
        return BindingParityResult(
            ok=False,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            shape_mismatch=True,
            detail=f"shape mismatch: pred {p.shape} vs ref {r.shape}",
        )
    if np.any(np.isnan(p)) or np.any(np.isnan(r)):
        return BindingParityResult(
            ok=False,
            max_abs_diff=float("nan"),
            max_rel_diff=float("nan"),
            shape_mismatch=False,
            detail="NaN in pred or ref",
        )

    if p.size == 0:
        return BindingParityResult(
            ok=True, max_abs_diff=0.0, max_rel_diff=0.0,
            shape_mismatch=False, detail="empty",
        )

    abs_d = float(np.max(np.abs(p - r)))
    rel_d = abs_d / max(float(np.max(np.abs(r))), 1e-300)
    ok = abs_d <= tolerance
    return BindingParityResult(
        ok=ok,
        max_abs_diff=abs_d,
        max_rel_diff=rel_d,
        shape_mismatch=False,
        detail=f"max_abs={abs_d:.3e}",
    )


__all__ = ["BindingParityResult", "binding_parity"]
