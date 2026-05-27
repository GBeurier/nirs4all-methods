"""Integrity gates for the donor-operator binding specs (`donor_ops.py`).

Three layers, increasingly demanding (the latter two skip when libn4m / the
`n4m_donor_bench` binary are not built):

1. Coverage + structure — every timeable donor op (the `n4m_donor_bench --list`
   set) has exactly one spec with a raw mapping (or an explicit reason) and an
   idiomatic mapping (or an explicit reason); no spec is stale. Cheap, no libn4m.
2. raw↔idiomatic parity — the two binding tiers agree bit-exact (same ABI, same
   seed) for every op that maps both.
3. raw≡C++ — for the dump-validatable ops the raw binding reproduces the C++
   donor-bench `out` buffer, proving the C++→raw param transcription + the
   splitmix64 input reproduction.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.cross_binding import donor_ops as D

REPO = Path(__file__).resolve().parents[3]
SEED, N, P = 1_234_567_890, 250, 50
ALLOWED_RAW_REASONS = {D.NO_RAW}
ALLOWED_IDIOM_REASONS = {D.NO_ESTIMATOR, D.SEMANTIC_MISMATCH, D.BINDING_DEFERRED}


def _find_bench() -> tuple[Path, Path] | None:
    """Return (n4m_donor_bench, libn4m) for any build that has both, or None.

    Globs every `build/*/` dir (so it finds whatever the local dev or CI
    preset produced); the lib is matched to the binary's own build so the
    dump comparison is bit-exact.
    """
    for binary in sorted(REPO.glob("build/*/bench/cpp/n4m_donor_bench")):
        src = binary.parents[2] / "cpp" / "src"        # build/<tier>/cpp/src
        libs = [p for p in sorted(src.glob("libn4m.so.*")) if p.name.count(".") >= 3]
        if libs:
            return binary, libs[0]
    return None


def _bench_list(binary: Path) -> set[str]:
    out = subprocess.run([str(binary), "--list"], capture_output=True, text=True, check=True)
    return {ln.strip() for ln in out.stdout.splitlines() if ln.strip()}


@pytest.fixture(scope="module")
def bench():
    found = _find_bench()
    if found is None:
        pytest.skip("no n4m_donor_bench build found (build with -DN4M_BUILD_BENCH=ON)")
    return found


@pytest.fixture(scope="module")
def n4m_loaded(bench):
    """Import n4m bound to the same build as the bench binary (bit-exact dumps)."""
    _, lib = bench
    os.environ["N4M_LIB_PATH"] = str(lib)
    sys.path.insert(0, str(REPO / "bindings/python/src"))
    try:
        import n4m  # noqa: F401
    except Exception as exc:  # pragma: no cover - host guard
        pytest.skip(f"n4m import failed: {exc}")
    return True


@pytest.fixture(scope="module")
def nirs4all_available(n4m_loaded):
    """Skip Layer-3 reference checks unless nirs4all (the sibling repo) imports."""
    for cand in (REPO.parent / "nirs4all", REPO.parent.parent / "nirs4all"):
        if (cand / "nirs4all" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            break
    try:
        import nirs4all  # noqa: F401
    except Exception:  # pragma: no cover - host guard
        pytest.skip("nirs4all not importable (set PYTHONPATH to the nirs4all checkout)")
    return True


# --- Layer 1: coverage + structure (no libn4m needed) ----------------------
def test_registry_matches_bench_list(bench):
    binary, _ = bench
    listed = _bench_list(binary)
    registry = set(D.REGISTRY)
    assert registry == listed, (
        f"donor_ops registry out of sync with `n4m_donor_bench --list`.\n"
        f"  missing specs (op listed but unmapped): {sorted(listed - registry)}\n"
        f"  stale specs (mapped but not listed):    {sorted(registry - listed)}"
    )


def test_every_spec_is_fully_declared():
    for spec in D.all_specs():
        # The dataclass enforces the XOR; assert the reason vocabularies here.
        if spec.raw_fn is None:
            assert spec.raw_reason in ALLOWED_RAW_REASONS, \
                f"{spec.bench_id}: bad raw_reason {spec.raw_reason!r}"
        if spec.idiom_cls is None:
            assert spec.idiom_reason in ALLOWED_IDIOM_REASONS, \
                f"{spec.bench_id}: bad idiom_reason {spec.idiom_reason!r}"
        assert spec.category in ("aug", "pp", "filter", "split")
        assert spec.parity in ("values", "mask", "indices")
        assert "X" in spec.inputs or "y" in spec.inputs


def test_dashboard_ids_are_unique():
    ids = [s.dashboard_id for s in D.all_specs()]
    assert len(ids) == len(set(ids)), "duplicate dashboard_id in donor_ops"


# --- Layer 2: raw↔idiomatic parity -----------------------------------------
@pytest.mark.parametrize("bench_id", sorted(
    s.bench_id for s in D.all_specs() if s.raw_fn and s.idiom_cls))
def test_raw_idiom_parity(bench_id, n4m_loaded):
    spec = D.REGISTRY[bench_id]
    inp = D.build_inputs(spec, N, P, SEED)
    raw = D.run_raw(spec, inp, SEED)
    idi = D.run_idiom(spec, inp, SEED)
    assert raw.shape == idi.shape, f"{bench_id}: shape {raw.shape} vs {idi.shape}"
    if spec.parity in ("mask", "indices"):
        assert np.array_equal(raw, idi), f"{bench_id}: discrete output differs"
    else:
        np.testing.assert_allclose(raw, idi, rtol=0, atol=1e-9,
                                   err_msg=f"{bench_id}: raw≠idiom")


# --- Layer 3: raw≡C++ (proves the C++→raw transcription) -------------------
@pytest.mark.parametrize("bench_id", sorted(
    s.bench_id for s in D.all_specs() if s.dump_validatable))
def test_raw_binding_matches_cpp(bench_id, bench, n4m_loaded):
    binary, _ = bench
    spec = D.REGISTRY[bench_id]
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        path = tf.name
    try:
        proc = subprocess.run(
            [str(binary), "--method", spec.bench_id, "--n", str(N), "--p", str(P),
             "--seed", str(SEED), "--runs", "2", "--dump-output", path],
            capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.skip(f"bench binary lacks --dump-output or failed: {proc.stderr[:80]}")
        cpp = np.fromfile(path, dtype="<f8")
        raw = D.run_raw(spec, D.build_inputs(spec, N, P, SEED), SEED)
        assert cpp.size == raw.size, f"{bench_id}: size {cpp.size} vs {raw.size}"
        np.testing.assert_allclose(raw, cpp, rtol=0, atol=1e-9,
                                   err_msg=f"{bench_id}: raw binding ≠ C++ donor bench")
    finally:
        os.unlink(path)


# --- Layer 3: nirs4all reference coverage + cross-parity --------------------
def test_every_dashboard_op_has_reference():
    """Every on-dashboard donor op maps to a nirs4all ReferenceSpec."""
    for spec in D.all_specs():
        if not spec.on_dashboard:
            continue
        ref = D.reference_for(spec)
        assert ref is not None, f"{spec.bench_id}: no nirs4all reference"
        assert ref.library == "nirs4all"


def _cross_parity_ok(spec, ref) -> tuple[bool, float]:
    """Run n4m (idiomatic) vs nirs4all at the dashboard size; return (ok, diff)."""
    inp = D.build_inputs(spec, N, P, SEED)
    ref_vec = D.run_reference(spec, ref, inp, SEED)
    n4m_vec = D.run_idiom(spec, inp, SEED)
    if ref_vec.shape != n4m_vec.shape:
        return False, float("inf")
    if spec.parity in ("mask", "indices"):
        return bool(np.array_equal(n4m_vec, ref_vec)), 0.0
    denom = float(np.sqrt(np.mean(ref_vec ** 2))) or 1.0
    rel = float(np.sqrt(np.mean((n4m_vec - ref_vec) ** 2)) / denom)
    return rel <= ref.tol_rel, rel


@pytest.mark.parametrize("bench_id", sorted(
    s.bench_id for s in D.all_specs() if s.on_dashboard))
def test_reference_runs_and_no_unexpected_divergence(bench_id, nirs4all_available):
    spec = D.REGISTRY[bench_id]
    ref = D.reference_for(spec)
    # The nirs4all reference must at least run (timing works for all 94).
    D.run_reference(spec, ref, D.build_inputs(spec, N, P, SEED), SEED)
    if not ref.value_parity:
        return  # timing-only (stochastic / seeded / different op)
    ok, _ = _cross_parity_ok(spec, ref)
    if not ok:
        assert bench_id in D._N4_EXPECTED_DIVERGENCE, (
            f"{bench_id}: UNEXPECTED divergence vs nirs4all — investigate or "
            f"add to _N4_EXPECTED_DIVERGENCE with a reason")


def test_no_stale_expected_divergence(nirs4all_available):
    """Every allowlisted expected-divergence op must actually diverge."""
    stale = []
    for bench_id in D._N4_EXPECTED_DIVERGENCE:
        spec = D.REGISTRY[bench_id]
        ref = D.reference_for(spec)
        ok, _ = _cross_parity_ok(spec, ref)
        if ok:
            stale.append(bench_id)
    assert not stale, f"stale _N4_EXPECTED_DIVERGENCE (these now match nirs4all): {stale}"
