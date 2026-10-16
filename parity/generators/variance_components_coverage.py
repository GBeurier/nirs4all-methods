# SPDX-License-Identifier: CECILL-2.1
"""Coverage study of the cluster-bootstrap percentile intervals (test-only).

Runs n4m_stats_variance_components_intervals from a built libn4m on simulated
Gaussian one-way designs and reports the empirical coverage of the nominal
95 % intervals of the within SD and of the REML ICC(1). Recorded with
Python 3.14.3 and NumPy 2.5.4:

    python parity/generators/variance_components_coverage.py LIBN4M out.json [nsim]

The recorded run (nsim = 1000) is parity/fixtures/variance_components_coverage_v1.json.
"""
import ctypes as C
import json
import sys

import numpy as np


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("rows", C.c_int64), ("cols", C.c_int64),
                ("rs", C.c_int64), ("cs", C.c_int64), ("dtype", C.c_int32), ("res", C.c_int32)]


lib = C.CDLL(sys.argv[1])
lib.n4m_context_create.argtypes = [C.c_void_p]
lib.n4m_stats_variance_components_intervals.argtypes = [
    C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_double, C.c_int32,
    C.c_uint64, C.c_void_p, C.c_void_p, C.c_void_p]
ctx = C.c_void_p()
lib.n4m_context_create(C.byref(ctx))


def intervals(y, groups, clusters, seed):
    x = np.ascontiguousarray(np.asarray(y, float)[:, None])
    g = np.asarray(groups, np.int64)
    c = np.asarray(clusters, np.int64) if clusters is not None else None
    out = np.zeros(8); valid = np.zeros(1, np.int64); status = np.zeros(1, np.int32)
    view = View(x.ctypes.data, len(y), 1, 1, 1, 1, 0)
    rc = lib.n4m_stats_variance_components_intervals(
        ctx, C.byref(view), g.ctypes.data, c.ctypes.data if c is not None else None, None,
        0.95, 999, seed, out.ctypes.data, valid.ctypes.data, status.ctypes.data)
    assert rc == 0
    return out, int(valid[0])


def scenario(name, sizes, icc, nsim, rng, unit_size=1, unit_share=0.0):
    """Groups of the given sizes; with unit_size > 1, consecutive groups share
    a unit effect carrying unit_share of the between variance."""
    hits = {"sd": 0, "icc": 0}
    evaluated = 0
    g_idx = np.repeat(np.arange(len(sizes)), sizes)
    units = g_idx // unit_size
    for s in range(nsim):
        n_units = units.max() + 1
        unit_effect = rng.normal(0, np.sqrt(icc * unit_share), n_units)
        group_effect = rng.normal(0, np.sqrt(icc * (1 - unit_share)), len(sizes))
        y = unit_effect[units] + group_effect[g_idx] + rng.normal(0, np.sqrt(1 - icc), len(g_idx))
        out, valid = intervals(y, g_idx, units if unit_size > 1 else None, s + 1)
        if valid < 999:
            continue
        evaluated += 1
        hits["sd"] += out[4] <= np.sqrt(1 - icc) <= out[5]
        hits["icc"] += out[6] <= icc <= out[7]
    return {"scenario": name, "groups": len(sizes), "observations": int(sum(sizes)), "icc": icc,
            "unit_size": unit_size, "evaluated": evaluated,
            "coverage_sd": hits["sd"] / evaluated, "coverage_icc": hits["icc"] / evaluated}


nsim = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rng = np.random.default_rng(20261013)
cycle = lambda g: [1 + i % 4 for i in range(g)]
rows = []
for icc in (0.0, 0.1, 0.5, 0.9):
    rows.append(scenario("balanced_20x2", [2] * 20, icc, nsim, rng))
for icc in (0.1, 0.5, 0.9):
    rows.append(scenario("unbalanced_24_sizes_1to4_singletons", cycle(24), icc, nsim, rng))
    rows.append(scenario("unbalanced_60_sizes_1to4_singletons", cycle(60), icc, nsim, rng))
    rows.append(scenario("balanced_50x3", [3] * 50, icc, nsim, rng))
rows.append(scenario("dependent_pairs_24_units_of_2_groups_x3", [3] * 48, 0.5, nsim, rng,
                     unit_size=2, unit_share=0.5))


# BCa check (balanced 20 x 2, closed-form balanced REML ICC = max(0, ANOVA)):
# the bias correction z0 breaks on the boundary mass at ICC = 0, so BCa
# under-covers badly at small ICC and is not adopted.
def balanced_icc(y):
    g, r = y.shape[-2:]
    m = y.mean(-1)
    msw = ((y - m[..., None]) ** 2).sum((-1, -2)) / (g * (r - 1))
    msb = r * ((m - m.mean(-1, keepdims=True)) ** 2).sum(-1) / (g - 1)
    return np.maximum(0, (msb - msw) / (msb + (r - 1) * msw))


def bca_study(icc, nsim, rng, g=20, r=2, b=999):
    from scipy.stats import norm
    hits = {"percentile": 0, "bca": 0}
    for _ in range(nsim):
        y = rng.normal(0, np.sqrt(icc), g)[:, None] + rng.normal(0, np.sqrt(1 - icc), (g, r))
        theta = balanced_icc(y)
        boot = balanced_icc(y[rng.integers(0, g, (b, g))])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        hits["percentile"] += lo <= icc <= hi
        z0 = norm.ppf(np.clip(((boot < theta).sum() + 0.5 * (boot == theta).sum()) / b, 1e-10, 1 - 1e-10))
        jack = np.array([balanced_icc(np.delete(y, i, 0)) for i in range(g)])
        d = jack.mean() - jack
        a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5) if (d ** 2).sum() > 0 else 0.0
        q = [norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in norm.ppf([0.025, 0.975])]
        lo, hi = np.quantile(boot, q)
        hits["bca"] += lo <= icc <= hi
    return {"icc": icc, "groups": g, "repeats": r,
            "coverage_percentile": hits["percentile"] / nsim, "coverage_bca": hits["bca"] / nsim}


bca = [bca_study(icc, nsim, rng) for icc in (0.1, 0.5, 0.9)]
for r in rows + bca:
    print(r, flush=True)
json.dump({"nominal": 0.95, "replicates": 999, "nsim": nsim,
           "monte_carlo_se": float(np.sqrt(0.95 * 0.05 / nsim)),
           "numpy": np.__version__, "python": sys.version.split()[0], "rows": rows,
           "bca_check": bca},
          open(sys.argv[2], "w"), indent=1)
