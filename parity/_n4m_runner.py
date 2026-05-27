"""Shared n4m-self runner for the Phase-C-min parity tools.

Both the self-consistency comparator (`parity/comparator/self_consistency.py`)
and the golden-snapshot generator (`parity/generators/run.py`) need the same
thing: run an n4m method on a deterministic input and get its output array
back. Rather than reimplement FFI/env plumbing, this reuses the cross-binding
orchestrator's `gen_dataset_csv` (deterministic CSV input) + `run_backend`
(per-cell env, libn4m discovery, registry param marshalling) through the
`bench_cpp.py` driver — i.e. the canonical C ABI path, no high-level wrapper.

This is intentionally tiny and side-effect-light; it does NOT own numerical
logic (the C++ core does) and introduces no parallel runner.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Preference order for an existing libn4m build when --lib-dir is not given.
_BUILD_PREFERENCE = ("blas-omp", "blas-on", "omp-on", "dev-release", "dev-debug")


def discover_lib_dir() -> str | None:
    """First existing `build/<preset>/cpp/src` holding a libn4m.so, or None."""
    for preset in _BUILD_PREFERENCE:
        cand = REPO / "build" / preset / "cpp" / "src"
        if (cand / "libn4m.so").exists():
            return str(cand)
    # Fall back to any build tree that has the shared object.
    for so in sorted((REPO / "build").glob("*/cpp/src/libn4m.so")):
        return str(so.parent)
    return None


def lib_identity(lib_dir: str) -> dict:
    """Binary identity of the libn4m under `lib_dir`: sha256 + size."""
    so = Path(lib_dir) / "libn4m.so"
    if not so.exists():
        return {"lib_sha256": None, "lib_bytes": None}
    h = hashlib.sha256(so.read_bytes()).hexdigest()
    return {"lib_sha256": h, "lib_bytes": so.stat().st_size}


def abi_version() -> str:
    """ABI version triple from the canonical header (matches render_api)."""
    import re

    hdr = REPO / "cpp" / "include" / "n4m" / "n4m_version.h"
    if not hdr.exists():
        return "unknown"
    txt = hdr.read_text(encoding="utf-8")
    parts = []
    for key in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"N4M_ABI_VERSION_{key}\s+(\d+)", txt)
        if not m:
            return "unknown"
        parts.append(m.group(1))
    return ".".join(parts)


def run_n4m(algo: str, n: int, p: int, nc: int, seed: int, *, lib_dir: str,
            name: str = "n4m", build_key: str = "parity",
            timeout: int = 300) -> tuple[bool, "np.ndarray | None", str]:
    """Run one n4m method on the deterministic (n, p, seed) dataset.

    Returns (ok, predictions_or_None, reason). Reuses the orchestrator's
    `bench_cpp.py` cell — the C ABI via ctypes, routed through the registry's
    canonical `MethodSpec.pls4all_fn` (covers PLS family + AOM/POP).
    """
    import sys

    sys.path.insert(0, str(REPO))
    from benchmarks.cross_binding import orchestrator

    orchestrator.LIBP4A_BUILDS[build_key] = lib_dir
    orchestrator.gen_dataset_csv(n, p, seed)
    rec = orchestrator.run_backend(
        name, "bench_cpp.py", "C++", "core", "n4m_core", algo, n, p, nc,
        1, 1, build_key, seed, timeout)
    if not rec.get("ok") or not rec.get("predictions_path"):
        return False, None, str(rec.get("reason", "unknown error"))
    return True, np.asarray(np.load(rec["predictions_path"]), dtype=np.float64), ""
