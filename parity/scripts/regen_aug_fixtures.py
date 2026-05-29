# SPDX-License-Identifier: CECILL-2.1
"""Regenerate (or check) the libn4m self-fixtures for the stochastic edge /
spline / random augmenters.

Ten ``parity/fixtures/aug_*_v1.json`` files shipped with ``output_hex`` byte-
identical to ``input_hex`` — identity placeholders that lock nothing. These
augmenters are *self-fixtures*: the oracle is libn4m's own deterministic output
at the canonical seed (donor bit-parity vs nirs4all is explicitly out of scope
for stochastic augmenters — nirs4all's numpy RNG ≠ n4m's PCG64, see
``benchmarks/cross_binding/donor_ops.py``). This tool freezes that output so the
C++ replay in ``cpp/tests/test_augmenters_edge_splines_random.cpp`` becomes a
real regression-lock.

``spline_smooth`` is deliberately excluded: it is the only FITPACK-gated
augmenter (a no-op when N4M_HAVE_FITPACK=0, real smoothing when =1), so its
output is build-dependent and a single committed self-fixture cannot lock it.
``detector_rolloff`` stays in: its fixture wavelengths (1000–1700 nm) sit inside
detector model 4's optimal band, so it is deterministically pass-through here —
the replay locks that in-band signal is not corrupted.

The libn4m call path (PCG64 seed → ``*_create`` → ``*_apply``) is identical to
the C++ test, so regenerated values match the test bit-for-bit by construction.

Usage (libn4m must be built; point N4M_LIB_PATH at it)::

    N4M_LIB_PATH=build/blas-omp/cpp/src/libn4m.so.1.9.0 \
    PYTHONPATH=bindings/python/src \
        python parity/scripts/regen_aug_fixtures.py            # rewrite in place
    ... regen_aug_fixtures.py --check                          # verify only (CI)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
from pathlib import Path

import numpy as np

from n4m.python import _aug_apply, lib  # noqa: E402  (after sys.path / env are set)


def _build_info() -> str:
    """Return libn4m's build_info string (e.g. 'fitpack=1')."""
    fn = lib.n4m_get_build_info
    fn.restype = ctypes.c_char_p
    return (fn() or b"").decode()


def _have_fitpack() -> bool:
    return "fitpack=1" in _build_info()

# Canonical augmenter parity seed — mirrors `kSeed` in
# cpp/tests/test_augmenters_edge_splines_random.cpp.
K_SEED = 3298921130

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _i32(v) -> ctypes.c_int32:
    return ctypes.c_int32(int(v))


def _f64(v) -> ctypes.c_double:
    return ctypes.c_double(float(v))


# Per-augmenter: libn4m symbol prefix, whether apply takes the wavelength axis,
# and how the case `params` map onto the C `*_create` argument list (order +
# type must match the C++ smoke/parity calls exactly).
def _detector_rolloff(p):
    return [_i32(p.get("detector_model", 4)), _f64(p.get("effect_strength", 1.0)),
            _f64(p.get("noise_amplification", 0.02)),
            _i32(p.get("include_baseline_distortion", 1))]


def _stray_light(p):
    return [_f64(p.get("stray_light_fraction", 0.001)), _f64(p.get("edge_enhancement", 2.0)),
            _f64(p.get("edge_width", 0.1)), _i32(p.get("include_peak_truncation", 1))]


def _edge_curve(p):
    return [_f64(p.get("curvature_strength", 0.02)), _i32(p.get("curvature_type", 1)),
            _f64(p.get("asymmetry", 0.0)), _f64(p.get("edge_focus", 0.7))]


def _truncated_peak(p):
    return [_f64(p.get("peak_probability", 0.5)), _f64(p.get("amplitude_min", 0.01)),
            _f64(p.get("amplitude_max", 0.1)), _f64(p.get("width_min", 50.0)),
            _f64(p.get("width_max", 200.0)), _i32(p.get("left_edge", 1)),
            _i32(p.get("right_edge", 1))]


def _edge_artifacts(p):
    return [_i32(p.get("enabled_flags", 15)), _f64(p.get("overall_strength", 1.0)),
            _i32(p.get("detector_model", 4))]


def _spline_x_perturb(p):
    return [_i32(p.get("spline_degree", 3)), _f64(p.get("perturbation_density", 0.05)),
            _f64(p.get("perturbation_range_min", -0.1)),
            _f64(p.get("perturbation_range_max", 0.1))]


def _spline_y_perturb(p):
    return [_i32(p.get("spline_points", -1)), _f64(p.get("perturbation_intensity", 0.005))]


def _rotate_translate(p):
    return [_f64(p.get("p_range", 2.0)), _f64(p.get("y_factor", 3.0))]


def _random_x_op(p):
    return [_i32(p.get("op_kind", 0)), _f64(p.get("operator_range_min", 0.97)),
            _f64(p.get("operator_range_max", 1.03))]


SPECS = [
    ("aug_detector_rolloff_v1.json", "n4m_aug_detector_rolloff", True,  _detector_rolloff),
    ("aug_stray_light_v1.json",      "n4m_aug_stray_light",      True,  _stray_light),
    ("aug_edge_curve_v1.json",       "n4m_aug_edge_curve",       True,  _edge_curve),
    ("aug_truncated_peak_v1.json",   "n4m_aug_truncated_peak",   True,  _truncated_peak),
    ("aug_edge_artifacts_v1.json",   "n4m_aug_edge_artifacts",   True,  _edge_artifacts),
    ("aug_spline_x_perturb_v1.json", "n4m_aug_spline_x_perturb", False, _spline_x_perturb),
    ("aug_spline_y_perturb_v1.json", "n4m_aug_spline_y_perturb", False, _spline_y_perturb),
    ("aug_rotate_translate_v1.json", "n4m_aug_rotate_translate", False, _rotate_translate),
    ("aug_random_x_op_v1.json",      "n4m_aug_random_x_op",      False, _random_x_op),
]


def _decode(hexes: list[str]) -> np.ndarray:
    """ieee754_binary64_be_hex (one big-endian hex string per double)."""
    return np.array([struct.unpack(">d", bytes.fromhex(h))[0] for h in hexes],
                    dtype=np.float64)


def _encode(arr: np.ndarray) -> list[str]:
    return [struct.pack(">d", float(v)).hex() for v in np.asarray(arr).reshape(-1)]


def _compute(fx: dict, prefix: str, needs_wl: bool, build, params: dict) -> np.ndarray:
    rows, cols = int(fx["rows"]), int(fx["cols"])
    X = _decode(fx["input_hex"]).reshape(rows, cols)
    wl = _decode(fx["wavelengths_hex"]).reshape(-1) if "wavelengths_hex" in fx else None
    seed = int(params.get("random_state", K_SEED))
    out = _aug_apply(prefix, X, *build(params), seed=seed,
                     wavelengths=wl, apply_wavelengths=needs_wl)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Phase-17 self-fixtures (parity/fixtures/aug_phase17_v1.json): 10 heterogeneous
# augmenters, one case each. Five take the wavelength axis as the LAST `*_create`
# args (wl_ptr, cols) — NOT as an apply arg — so the regen loop appends those to
# the create-arg list with apply_wavelengths left False. The other five do not.
# Five cases ship a frozen output_hex + a test_parity_* lock already; only the
# five EMPTY cases are (re)generated here, against the committed wavelength axis,
# at PHASE17_SEED (the n4m_rng_pcg64_create(0u, ...) the C++ replays use).
# ---------------------------------------------------------------------------
PHASE17_FIXTURE = "aug_phase17_v1.json"
PHASE17_SEED = 0


def _p17_particle_size(p):
    return [_f64(p.get("mean_size_um", 50.0)), _f64(p.get("size_variation_um", 15.0)),
            ctypes.c_int(0), _f64(0.0), _f64(0.0),
            _f64(p.get("reference_size_um", 50.0)), _f64(p.get("wavelength_exponent", 1.5)),
            _f64(p.get("size_effect_strength", 0.1)),
            ctypes.c_int(int(p.get("include_path_length", 1))),
            _f64(p.get("path_length_sensitivity", 0.5))]


def _p17_emsc_distort(p):
    return [_f64(p.get("mult_low", 0.9)), _f64(p.get("mult_high", 1.1)),
            _f64(p.get("add_low", -0.05)), _f64(p.get("add_high", 0.05)),
            _i32(p.get("polynomial_order", 2)), _f64(p.get("polynomial_strength", 0.02)),
            _f64(p.get("correlation", 0.3))]


def _p17_moisture(p):
    return [_f64(p.get("water_activity_delta", 0.1)), ctypes.c_int(0), _f64(0.0), _f64(0.0),
            _f64(p.get("reference_water_activity", 0.5)), _f64(p.get("free_water_fraction", 0.3)),
            _f64(p.get("bound_water_shift", 25.0)), _f64(p.get("moisture_content", 0.1)),
            ctypes.c_int(int(p.get("enable_shift", 1))), ctypes.c_int(int(p.get("enable_intensity", 1)))]


def _p17_mixup(p):
    return [_f64(p.get("alpha", 0.2))]


def _p17_local_mixup(p):
    return [_f64(p.get("alpha", 0.2)), _i32(p.get("k_neighbors", 5))]


# case-name -> (libn4m prefix, needs_wl, builder). Only the empty cases are listed;
# the 5 deterministic cases are already locked by their test_parity_* tests.
PHASE17 = {
    "particle_size": ("n4m_aug_particle_size", True,  _p17_particle_size),
    "emsc_distort":  ("n4m_aug_emsc_distort",  True,  _p17_emsc_distort),
    "moisture":      ("n4m_aug_moisture",      True,  _p17_moisture),
    "mixup_alpha02": ("n4m_aug_mixup",         False, _p17_mixup),
    "local_mixup":   ("n4m_aug_local_mixup",   False, _p17_local_mixup),
}


def _regen_phase17(check: bool) -> bool:
    """Fill the EMPTY per-case output_hex in aug_phase17_v1.json from libn4m.

    Cases that already carry a frozen output_hex (the 5 deterministic locks) are
    left untouched. The committed wavelengths_hex axis is reused as-is; the C++
    replay reads the same axis via load_det_case. Returns True if stale.
    """
    path = FIXTURE_DIR / PHASE17_FIXTURE
    fx = json.loads(path.read_text())
    rows, cols = int(fx["rows"]), int(fx["cols"])
    X = _decode(fx["input_hex"]).reshape(rows, cols)
    wl = _decode(fx["wavelengths_hex"]).reshape(-1)
    wl_ptr = wl.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    changed = False
    for case in fx["cases"]:
        if case.get("output_hex"):  # already locked — never recompute
            continue
        prefix, needs_wl, build = PHASE17[case["name"]]
        args = list(build(case.get("params", {})))
        if needs_wl:
            args += [wl_ptr, ctypes.c_int64(cols)]
        out = np.asarray(_aug_apply(prefix, X, *args, seed=PHASE17_SEED),
                         dtype=np.float64)
        new_hex = _encode(out)
        if case.get("output_hex", []) != new_hex:
            changed = True
            case["output_hex"] = new_hex
            case["output_rows"] = int(out.shape[0])
            case["output_cols"] = int(out.shape[1])

    if changed and not check:
        path.write_text(json.dumps(fx) + "\n")
        print(f"  rewrote {PHASE17_FIXTURE}")
    elif changed:
        print(f"  STALE   {PHASE17_FIXTURE}")
    else:
        print(f"  ok      {PHASE17_FIXTURE}")
    return changed


# spline_smooth is the only FITPACK-gated augmenter
# (cpp/src/core/augmentation/splines/spline_smoothing.c): with
# N4M_HAVE_FITPACK=0 (no Fortran compiler) it is an intentional no-op (memcpy),
# with =1 it performs real cubic spline smoothing. It takes no create-args and
# no wavelength axis. The oracle is libn4m's OWN deterministic FITPACK output
# at the canonical seed (a self-fixture, like the rest of this file). A single
# committed fixture can lock only one build config, so we touch it ONLY from a
# fitpack=1 build; under fitpack=0 we skip entirely (neither rewrite nor flag
# stale) so the non-Fortran dev/CI builds keep passing and never freeze the
# identity placeholder as the oracle.
SPLINE_SMOOTH_FIXTURE = "aug_spline_smooth_v1.json"


def _regen_spline_smooth(check: bool) -> bool:
    """Regenerate the FITPACK-gated spline_smooth self-fixture. Returns True
    if the stored output is stale (only possible under fitpack=1)."""
    if not _have_fitpack():
        print(f"  skip    {SPLINE_SMOOTH_FIXTURE} "
              f"(build_info='{_build_info()}'; needs fitpack=1)")
        return False
    path = FIXTURE_DIR / SPLINE_SMOOTH_FIXTURE
    fx = json.loads(path.read_text())
    changed = False
    for case in fx["cases"]:
        out = _compute(fx, "n4m_aug_spline_smooth", False, lambda p: [],
                       case.get("params", {}))
        new_hex = _encode(out)
        if new_hex != case.get("output_hex", []):
            changed = True
            case["output_hex"] = new_hex
            case["output_rows"] = int(out.shape[0])
            case["output_cols"] = int(out.shape[1])
    if changed and not check:
        path.write_text(json.dumps(fx, indent=2) + "\n")
        print(f"  rewrote {SPLINE_SMOOTH_FIXTURE} (fitpack oracle)")
    elif changed:
        print(f"  STALE   {SPLINE_SMOOTH_FIXTURE} (fitpack oracle)")
    else:
        print(f"  ok      {SPLINE_SMOOTH_FIXTURE} (fitpack oracle)")
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify stored output_hex matches libn4m; do not write")
    args = ap.parse_args(argv)

    stale = 0
    for fname, prefix, needs_wl, build in SPECS:
        path = FIXTURE_DIR / fname
        fx = json.loads(path.read_text())
        changed = False
        for case in fx["cases"]:
            out = _compute(fx, prefix, needs_wl, build, case.get("params", {}))
            new_hex = _encode(out)
            old_hex = case.get("output_hex", [])
            if new_hex != old_hex:
                changed = True
                case["output_hex"] = new_hex
                case["output_rows"] = int(out.shape[0])
                case["output_cols"] = int(out.shape[1])
        if changed:
            stale += 1
            if args.check:
                print(f"  STALE   {fname}")
            else:
                path.write_text(json.dumps(fx, indent=2) + "\n")
                print(f"  rewrote {fname}")
        else:
            print(f"  ok      {fname}")

    if _regen_phase17(args.check):
        stale += 1

    if _regen_spline_smooth(args.check):
        stale += 1

    if args.check and stale:
        print(f"\n{stale} fixture(s) do not match libn4m — run without --check to regenerate.")
        return 1
    n_total = len(SPECS) + 1 + (1 if _have_fitpack() else 0)
    print(f"\n{'checked' if args.check else 'regenerated'} {n_total} augmenter fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
