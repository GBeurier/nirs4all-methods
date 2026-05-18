# SPDX-License-Identifier: CECILL-2.1
"""Gate 2 comparator: does the C kernel match the canonical external reference?

This module mirrors :mod:`pls4all.benchmarks.parity_timing._comparator` so
that both projects use the same comparator contract. See
:mod:`parity.binding_parity` for the companion Gate-1 comparator.

Used when bit-exactness is impossible: the C engine and the canonical
external implementation (numpy / scipy / sklearn / pywt / pybaselines /
nirs4all) follow different solvers, iteration tolerances, or summation
orders — they should agree on the *values* but not bit-for-bit. The
tolerance is per-op in ``parity/tolerances.md``.

Shape-mismatch handling mirrors :func:`binding_parity` exactly so both
gates report drift the same way. When ``||ref||_RMS`` is essentially
zero, the comparator falls back to absolute RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceParityResult:
    """Tolerance-based agreement against an external reference.

    Attributes
    ----------
    ok
        ``True`` iff shape match AND the appropriate threshold passes.
        Threshold is ``rmse_rel <= tolerance`` for non-zero references,
        ``rmse_abs <= tolerance`` for references whose RMS magnitude is
        below ``1e-15``.
    rmse_abs
        Absolute RMSE: ``sqrt(mean((pred - ref)^2))``.
    rmse_rel
        Relative RMSE: ``rmse_abs / ||ref||_RMS``. Set to ``0.0`` when
        the reference is essentially zero.
    shape_mismatch
        ``True`` when ``pred.shape != ref.shape``.
    detail
        Short human-readable summary suitable for log lines.
    """

    ok: bool
    rmse_abs: float
    rmse_rel: float
    shape_mismatch: bool
    detail: str


def reference_parity(pred: np.ndarray, ref: np.ndarray,
                      tolerance: float) -> ReferenceParityResult:
    """Relative-RMSE comparator: ``||pred - ref||_RMS / ||ref||_RMS <= tol``.

    Falls back to absolute RMSE when the reference is essentially zero
    (RMS magnitude below ``1e-15``), avoiding division-by-zero false
    negatives for ops whose canonical reference legitimately returns
    near-zero arrays.

    Parameters
    ----------
    pred
        Predicted array (from libc4a).
    ref
        Reference array (from the canonical external implementation).
    tolerance
        Per-op tolerance. Recorded in ``parity/tolerances.md``.

    Returns
    -------
    ReferenceParityResult
        See the dataclass docstring.
    """
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)

    if p.shape != r.shape:
        return ReferenceParityResult(
            ok=False,
            rmse_abs=float("inf"),
            rmse_rel=float("inf"),
            shape_mismatch=True,
            detail=f"shape mismatch: pred {p.shape} vs ref {r.shape}",
        )
    if np.any(np.isnan(p)) or np.any(np.isnan(r)):
        return ReferenceParityResult(
            ok=False,
            rmse_abs=float("nan"),
            rmse_rel=float("nan"),
            shape_mismatch=False,
            detail="NaN in pred or ref",
        )
    if p.size == 0:
        return ReferenceParityResult(
            ok=True, rmse_abs=0.0, rmse_rel=0.0,
            shape_mismatch=False, detail="empty",
        )

    diff = p - r
    rmse_abs = float(np.sqrt(np.mean(diff * diff)))
    rmse_ref = float(np.sqrt(np.mean(r * r)))

    if rmse_ref < 1e-15:
        # Reference essentially zero; fall back to absolute RMSE.
        ok = rmse_abs <= tolerance
        return ReferenceParityResult(
            ok=ok,
            rmse_abs=rmse_abs,
            rmse_rel=0.0,
            shape_mismatch=False,
            detail=f"abs rmse {rmse_abs:.3e} (ref ~0)",
        )

    rmse_rel = rmse_abs / rmse_ref
    ok = rmse_rel <= tolerance
    return ReferenceParityResult(
        ok=ok,
        rmse_abs=rmse_abs,
        rmse_rel=rmse_rel,
        shape_mismatch=False,
        detail=f"rmse_rel={rmse_rel:.3e}",
    )


__all__ = ["ReferenceParityResult", "reference_parity"]
