"""Shared parity comparators used by the parity-timing gate AND the
cross-binding bench's reference-parity check.

Two distinct notions of "parity" live in the repo:

1. **Binding consistency** — every pls4all language binding (cpp,
   python_tier1, python_tier2, registry_pls4all, r_tier1, r_tier2,
   matlab_tier1, matlab_tier2) should produce bit-identical outputs to
   the C kernel. Tolerance: 1e-6 on `max|pred - ref|`. Used by the
   cross-binding orchestrator on the per-cell predictions cache.

2. **External-reference validity** — pls4all's algorithm implementation
   should agree with an external "ground truth" (sklearn /
   ropls / R::pls / ikpls / mixOmics / …) to within
   `MethodSpec.rmse_rel_tol`. Tolerance: relative RMSE with an
   absolute-RMSE escape near zero. Used by both `parity_timing.runner`
   (its primary purpose) and the cross-binding orchestrator's new
   second parity gate.

The functions in this module are pure-numpy, side-effect-free, and
deliberately small so the two callers can share them without pulling
the rest of `parity_timing.runner`'s plumbing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BindingParityResult:
    """Bit-exact agreement between two prediction arrays.

    `max_abs_diff` is the max element-wise absolute difference. `ok` is
    True when `max_abs_diff <= tolerance`. NaN diff (shape mismatch /
    missing) → ok=False.
    """
    max_abs_diff: float
    tolerance: float
    ok: bool
    note: str = ""


@dataclass(frozen=True)
class ReferenceParityResult:
    """Tolerance-based agreement against an external reference.

    `rmse_abs = sqrt(mean((pred - ref)^2))`. `rmse_rel = rmse_abs /
    ||ref||_RMS`. The "absolute escape" `rmse_abs <= 1e-9` short-circuits
    to ok=True for methods whose reference is essentially zero (avoids
    division-by-zero false negatives).
    """
    rmse_abs: float
    rmse_rel: float
    tolerance: float
    ok: bool
    note: str = ""


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Element-wise root-mean-squared difference between two arrays."""
    diff = np.asarray(a) - np.asarray(b)
    return float(math.sqrt(float(np.mean(diff * diff))))


def rms(arr: np.ndarray) -> float:
    """Root-mean-square magnitude of an array."""
    a = np.asarray(arr)
    return float(math.sqrt(float(np.mean(a * a))))


def _reshape_vector_compatible(p: np.ndarray,
                               r: np.ndarray) -> tuple[np.ndarray, str]:
    """Allow only 1-D vs column/row-vector shape quirks, never matrices."""
    if p.shape == r.shape:
        return p, ""
    if p.size != r.size:
        return p, f"shape mismatch ({p.shape} vs {r.shape})"
    p_is_vector = p.ndim == 1 or (p.ndim == 2 and 1 in p.shape)
    r_is_vector = r.ndim == 1 or (r.ndim == 2 and 1 in r.shape)
    if p_is_vector and r_is_vector:
        return p.reshape(r.shape), ""
    return p, f"shape mismatch ({p.shape} vs {r.shape})"


def binding_parity(pred: np.ndarray, ref: np.ndarray,
                    tolerance: float = 1e-6) -> BindingParityResult:
    """Bit-exact comparator: max|pred - ref| ≤ tolerance.

    Used by the cross-binding orchestrator to verify every binding
    produces the same numbers as the cpp reference. Only 1-D vs row/column
    vector quirks are reshaped; matrix-vs-matrix shape mismatches fail.
    """
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if p.size == 0 or r.size == 0:
        return BindingParityResult(
            max_abs_diff=float("nan"), tolerance=tolerance, ok=False,
            note=f"empty predictions ({p.shape} vs {r.shape})")
    if p.shape != r.shape:
        p, note = _reshape_vector_compatible(p, r)
        if note:
            return BindingParityResult(
                max_abs_diff=float("nan"), tolerance=tolerance, ok=False,
                note=note)
    diff = float(np.max(np.abs(p - r)))
    return BindingParityResult(
        max_abs_diff=diff, tolerance=tolerance, ok=diff <= tolerance)


def reference_parity(pred: np.ndarray, ref: np.ndarray,
                      tolerance: float) -> ReferenceParityResult:
    """Relative-RMSE comparator: ||pred - ref||_RMS / ||ref||_RMS ≤ tol.

    Falls through to ok=True when `rmse_abs ≤ 1e-9` (reference is
    essentially zero, no useful relative scale). Shape mismatch handling
    mirrors `binding_parity`.
    """
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if p.size == 0 or r.size == 0:
        return ReferenceParityResult(
            rmse_abs=float("nan"), rmse_rel=float("nan"),
            tolerance=tolerance, ok=False,
            note=f"empty predictions ({p.shape} vs {r.shape})")
    if p.shape != r.shape:
        p, note = _reshape_vector_compatible(p, r)
        if note:
            return ReferenceParityResult(
                rmse_abs=float("nan"), rmse_rel=float("nan"),
                tolerance=tolerance, ok=False,
                note=note)
    abs_rmse = rmse(p, r)
    ref_rms = rms(r) or 1.0
    rel_rmse = abs_rmse / ref_rms
    return ReferenceParityResult(
        rmse_abs=abs_rmse, rmse_rel=rel_rmse, tolerance=tolerance,
        ok=bool(rel_rmse <= tolerance or abs_rmse <= 1e-9))
