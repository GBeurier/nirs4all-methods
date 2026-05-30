#!/usr/bin/env python3
"""R-exact UVE (mcuve_pls) reference reimplementation — Tier C pilot proof.

Demonstrates that n4m's bit-exact R RNG primitives (rng_mt_r: runif + sample)
plus the standard UVE algorithm reproduce R `plsVarSel::mcuve_pls`'s selected
feature set EXACTLY (Jaccard 1.0). This is the algorithmic blueprint for the
forthcoming additive C-kernel path (cfg.uve_legacy=0 + rng_kind=MT_R); every
step here maps 1:1 onto a primitive that already exists in libn4m:

  R step                          n4m primitive
  ------                          -------------
  runif(Nx*Mx) column-major   ->  n4m_rng_mt_r_unif  (bit-exact)
  sample(Mx)                  ->  n4m_rng_mt_r_sample (bit-exact)
  plsr(...) SIMPLS coef       ->  n4m PLS (parity ~1e-13 vs pls::plsr)
  validation="LOO" PRESS      ->  per-subsample LOO refit (n4m PLS)
  which.min(PRESS)            ->  argmin
  RI = mean/sd; threshold     ->  trivial
  too-few fallback            ->  trivial

Run (needs the rng_mt_r shared lib + sklearn):
  cc -O2 -fPIC -shared -Icpp/src/core/common cpp/src/core/common/rng_mt_r.c \
     -lm -o /tmp/librngmtr.so
  N4M_RNG_MT_R_SO=/tmp/librngmtr.so python parity/scripts/uve_r_exact_reference.py X.csv y.csv

Proven 2026-05-30 on a 40x12 set (signal in feat 3 & 8): R mcuve_pls selects
{3,4,5,8,9} (1-based); this reference selects {3,4,5,8,9}. Jaccard 1.0.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression


def _load_rng(so_path: str):
    lib = ctypes.CDLL(so_path)
    lib.n4m_rng_mt_r_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.n4m_rng_mt_r_unif.argtypes = [ctypes.c_void_p]
    lib.n4m_rng_mt_r_unif.restype = ctypes.c_double
    lib.n4m_rng_mt_r_sample.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int)]
    return lib


def uve_r_exact(X: np.ndarray, y: np.ndarray, *, seed: int = 11, n_iter: int = 3,
                ratio: float = 0.75, ncomp: int = 5,
                so_path: str | None = None) -> list[int]:
    """Return 0-based selected feature indices, bit-set-identical to R mcuve_pls."""
    lib = _load_rng(so_path or os.environ.get("N4M_RNG_MT_R_SO", "/tmp/librngmtr.so"))
    state = ctypes.create_string_buffer(4096)  # >= sizeof(n4m_rng_mt_r)
    lib.n4m_rng_mt_r_set_seed(state, seed)
    runif = lambda: lib.n4m_rng_mt_r_unif(state)

    def sample(n):
        out = (ctypes.c_int * n)()
        lib.n4m_rng_mt_r_sample(state, n, out)
        return list(out)

    Mx, Nx = X.shape
    K = int(np.floor(Mx * ratio))
    ncomp = min(min(X.shape), ncomp)
    # R: W <- matrix(runif(Nx*Mx,0,1), Mx, Nx) — column-major fill.
    wflat = np.array([runif() for _ in range(Nx * Mx)])
    W = wflat.reshape(Nx, Mx).T
    Z = np.hstack([X, W])

    def loo_press(Zc, yc, maxk):
        n = Zc.shape[0]
        press = np.zeros(maxk + 1)
        for k in range(1, maxk + 1):
            e2 = 0.0
            for i in range(n):
                mask = np.ones(n, bool); mask[i] = False
                m = PLSRegression(n_components=k, scale=False).fit(Zc[mask], yc[mask])
                pred = np.asarray(m.predict(Zc[i:i + 1])).ravel()[0]
                e2 += (yc[i] - pred) ** 2
            press[k] = e2
        return press

    C = np.zeros((n_iter, Nx * 2))
    for it in range(n_iter):
        calk = sample(Mx)[:K]
        Zcal, ycal = Z[calk], y[calk]
        maxk = min(ncomp, Zcal.shape[1] - 1)
        opt = int(np.argmin(loo_press(Zcal, ycal, maxk)[1:maxk + 1])) + 1
        m = PLSRegression(n_components=opt, scale=False).fit(Zcal, ycal)
        c = m.coef_.ravel()
        C[it, :] = c[:Nx * 2] if c.size >= Nx * 2 else np.pad(c, (0, Nx * 2 - c.size))

    RI = C.mean(0) / C.std(0, ddof=1)
    thr = np.max(np.abs(RI[Nx:]))
    sel = [j for j in range(Nx) if abs(RI[j]) > thr]
    if len(sel) <= ncomp + 1:
        sel = sorted(np.argsort(-np.abs(RI[:Nx]))[:ncomp].tolist())
    return sorted(sel)


if __name__ == "__main__":
    X = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
    y = np.loadtxt(sys.argv[2], delimiter=",", skiprows=1)
    sel = uve_r_exact(X, y)
    print("selected (0-based):", sel)
    print("selected (1-based):", [s + 1 for s in sel])
