# SPDX-License-Identifier: CECILL-2.1
"""Generate parity/fixtures/permanova_v1.json, the PERMANOVA / PERMDISP gate.

Test-only. Requires exactly numpy==2.2.6, scipy==1.15.3 and scikit-bio==0.7.4
(all BSD-3-Clause); none of them enters libn4m.

The reference is deliberately computed by a different algorithm from the
native kernels. Native code uses the reduced normal equations of the two-way
categorical design and the connected components of its level graph; this
generator fits full one-hot designs with ``numpy.linalg.lstsq`` and takes the
rank from ``numpy.linalg.matrix_rank``:

* SS(model) = ||fitted - grand mean||^2 of the multivariate least-squares fit;
  ss_strata = SS(strata), ss_term = SS(strata + term) - SS(strata), which is
  the vegan ``adonis2(by = "margin")`` sum of squares of the term in the
  additive model term + strata (for one other factor). With Euclidean
  distances PERMANOVA (Anderson 2001) equals this trace decomposition.
* pseudo-F = (ss_term / df_term) / (ss_residual / df_residual), with
  df_term = rank(strata + term) - rank(strata) and
  df_residual = n - rank(strata + term).
* For free (unstratified) cases the pseudo-F is cross-checked against
  scikit-bio ``permanova`` on the Euclidean distance matrix.
* PERMDISP is vegan ``betadisper(type = "centroid")`` + ``permutest``:
  Euclidean distances to group centroids, one-way ANOVA F (cross-checked with
  scipy ``f_oneway`` and scikit-bio ``permdisp(test="centroid")``), and a
  permutation test transcribed from vegan ``permutest.betadisper``: the
  residuals of ``lm(distances ~ group)`` are permuted, refitted by least
  squares (``qr.fitted``/``qr.resid`` here ``lstsq``), and
  ``F = (mss/(p-1)) / (rss/(n-p))`` is counted when ``F >= F0 - EPS`` with
  ``EPS = sqrt(.Machine$double.eps)``.

Monte Carlo draws reproduce the native convention with
``rng = numpy.random.default_rng(seed)``. PERMANOVA: for each draw, each
stratum in ascending code order (all rows when unstratified) takes
``q = rng.permutation(n_s)`` over its rows in ascending order, and row
``idx[j]`` receives the data of row ``idx[q[j]]``. As vegan ``anova.cca``
(adonis2), a draw counts when its pseudo-F ``>= F_obs - sqrt(eps)`` (+inf
when its residual SS is zero); no test runs when the observed pseudo-F is
undefined. PERMDISP: ``perm = rng.permutation(n)``; row i receives residual
``r[perm[i]]``. Both comparisons are on the F scale, never on a monotone
transform, because an additive tolerance is not invariant under it (case
``permdisp_boundary``).

Freedman-Lane (Freedman & Lane 1983; Anderson 2001, Can. J. Fish. Aquat. Sci.
58:626-639): the partial test of the term given the additive main effects of
the condition factors permutes the residuals of the reduced model. The oracle
fits both designs with ``lstsq`` and, for each ``perm = rng.permutation(n)``,
refits ``Y* = fitted_reduced + E[perm]`` on both designs; pseudo-F counts when
``>= F0 - sqrt(eps)``. vegan is NOT this algorithm, so it is not the oracle;
two vegan schemes are transcribed only as informational counts (vegan master
964a8de, 2026-10-12):
* ``vegan_margin_raw``: ``adonis2(by = "margin")`` = ``anovaCCAbymargin``
  (R/anova.ccabyterm.R lines 54-110) permutes the raw response for the full
  model and for each model without the term, and compares
  ``(SS_full - SS_without_term)/Df / (RSS_full/df)`` with ``Fstat - EPS``
  (lines 76-79 note that conditioning was dropped after vegan 2.0).
* ``vegan_partial_permutest``: ``permutest.cca(model = "reduced")`` of a partial
  model term + Condition(others) (R/permutest.cca.R lines 154-157, src/getF.c
  lines 383-447) permutes the reduced-model residuals AND the condition rows
  together, residualizes both on the permuted conditions and keeps the term
  fixed; ``r = n - rank - 1``.

Usage: python parity/generators/permanova_v1.py [output.json]
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway
from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp

PINNED = {"numpy": "2.2.6", "scipy": "1.15.3", "scikit-bio": "0.7.4"}
SEED = 20261013
PERMUTATIONS = 999
TOL = float(np.sqrt(np.finfo(np.float64).eps))


def hexd(values) -> list[str]:
    return [struct.pack(">d", float(v)).hex() for v in np.ravel(values)]


def one_hot(codes: np.ndarray) -> np.ndarray:
    levels = np.unique(codes)
    return (codes[:, None] == levels[None, :]).astype(np.float64)


def model_ss(X: np.ndarray, design: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(design, X, rcond=None)
    fitted = design @ beta
    return float(np.sum((fitted - X.mean(axis=0)) ** 2)), int(np.linalg.matrix_rank(design))


def decompose(X: np.ndarray, term: np.ndarray, strata: np.ndarray) -> dict:
    n = X.shape[0]
    ss_total = float(np.sum((X - X.mean(axis=0)) ** 2))
    S = one_hot(strata)
    ss_strata, rank_s = model_ss(X, S)
    ss_full, rank_f = model_ss(X, np.hstack([S, one_hot(term)]))
    ss_term = max(ss_full - ss_strata, 0.0) if rank_f > rank_s else 0.0
    within = ss_total - ss_strata
    ss_res = max(within - ss_term, 0.0)
    df_term = rank_f - rank_s
    df_res = n - rank_f
    defined = df_term > 0 and within > 0
    partial = ss_term / within if within > 0 else float("nan")
    f = (ss_term / df_term) / (ss_res / df_res) if defined and df_res > 0 and ss_res > 0 else float("nan")
    return {
        "ss_total": ss_total, "ss_strata": ss_strata, "ss_term": ss_term, "ss_residual": ss_res,
        "r2": ss_term / ss_total if ss_total > 0 and df_term > 0 else float("nan"),
        "partial_r2": partial if defined else float("nan"),
        "pseudo_f": f, "df_term": df_term, "df_residual": df_res, "defined": defined,
    }


def log10_distinct(term: np.ndarray, strata: np.ndarray) -> float:
    from math import lgamma, log

    total = 0.0
    for s in np.unique(strata):
        t = term[strata == s]
        total += lgamma(t.size + 1) - sum(lgamma(int(c) + 1) for c in np.unique(t, return_counts=True)[1])
    return total / log(10.0)


def permanova_counts(X, term, strata, seed) -> int:
    obs = decompose(X, term, strata)
    if not np.isfinite(obs["pseudo_f"]):
        return 0
    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    b = 0
    for _ in range(PERMUTATIONS):
        perm = np.arange(X.shape[0])
        for idx in blocks:
            perm[idx] = idx[rng.permutation(idx.size)]
        d = decompose(X[perm], term, strata)
        f = (d["ss_term"] / d["df_term"]) / (d["ss_residual"] / d["df_residual"]) if d["ss_residual"] > 0 else np.inf
        if f >= obs["pseudo_f"] - TOL:
            b += 1
    return b


def permdisp_reference(X: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    levels = np.unique(groups)
    z = np.empty(X.shape[0])
    for g in levels:
        rows = groups == g
        z[rows] = np.linalg.norm(X[rows] - X[rows].mean(axis=0), axis=1)
    means = np.array([z[groups == g].mean() for g in levels])
    fitted = means[np.searchsorted(levels, groups)]
    resid = z - fitted
    between = float(np.sum((fitted - z.mean()) ** 2))
    within = float(np.sum(resid**2))
    n, k = z.size, levels.size
    f = (between / (k - 1)) / (within / (n - k))
    scipy_f = float(f_oneway(*[z[groups == g] for g in levels]).statistic)
    # SciPy forms SS_within by subtraction; on the near-singular boundary case
    # it loses ~1e-5 relative, so the cross-check tolerance is 1e-4. The
    # recorded F uses the residuals directly.
    assert abs(scipy_f - f) <= 1e-4 * f, (scipy_f, f)
    # vegan permutest.betadisper, transcribed: permute residuals, refit.
    design = (groups[:, None] == levels[None, :]).astype(np.float64)
    rank = int(np.linalg.matrix_rank(design))
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(PERMUTATIONS):
        perm_resid = resid[rng.permutation(n)]
        beta, *_ = np.linalg.lstsq(design, perm_resid, rcond=None)
        fit = design @ beta
        mss = float(np.sum((fit - fit.mean()) ** 2))
        rss = float(np.sum((perm_resid - fit) ** 2))
        f_perm = (mss / (rank - 1)) / (rss / (n - rank)) if rss > 0 else np.inf
        if f_perm >= f - TOL:
            b += 1
    # Informational: the rejected rule, mss/total >= B/(B+W) - TOL on U.
    rng = np.random.default_rng(seed)
    total = float(np.sum((resid - resid.mean()) ** 2))
    b_u = 0
    for _ in range(PERMUTATIONS):
        r = resid[rng.permutation(n)]
        bp = sum(r[groups == g].sum() ** 2 / np.sum(groups == g) for g in levels) - r.sum() ** 2 / n
        b_u += bp / total >= between / (between + within) - TOL
    from math import lgamma, log

    distinct = (lgamma(n + 1) - sum(lgamma(int(c) + 1) for c in np.unique(groups, return_counts=True)[1])) / log(10.0)
    return {"distances": z, "group_mean_distances": means, "f": f, "df1": k - 1, "df2": n - k,
            "n_groups": k, "log10_distinct": distinct, "b": b, "b_u_rule": int(b_u)}


def _rank(D):
    return int(np.linalg.matrix_rank(D))


def _rss(D, Y):
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    return float(np.sum((Y - D @ beta) ** 2))


def freedman_lane(X, term, conditions, seed) -> dict:
    n = X.shape[0]
    Z = np.hstack([np.ones((n, 1))] + [one_hot(c) for c in conditions])
    W = np.hstack([Z, one_hot(term)])
    rz, rw = _rank(Z), _rank(W)
    df_term, df_res = rw - rz, n - rw
    beta, *_ = np.linalg.lstsq(Z, X, rcond=None)
    fitted = Z @ beta
    E = X - fitted

    def fstat(Y):
        rr, rf = _rss(Z, Y), _rss(W, Y)
        return ((rr - rf) / df_term) / (rf / df_res) if rf > 0 else np.inf

    ss_total = float(np.sum((X - X.mean(axis=0)) ** 2))
    rr, rf = _rss(Z, X), _rss(W, X)
    ss_term = max(rr - rf, 0.0) if df_term > 0 else 0.0
    defined = df_term > 0 and df_res > 0 and rf > 0
    f0 = fstat(X) if defined else float("nan")
    b = 0
    if defined:
        rng = np.random.default_rng(seed)
        for _ in range(PERMUTATIONS):
            if fstat(fitted + E[rng.permutation(n)]) >= f0 - TOL:
                b += 1
    out = {"ss_total": ss_total, "ss_reduced": max(ss_total - rr, 0.0), "ss_term": ss_term,
           "ss_residual": rf, "pseudo_f": f0, "df_term": df_term, "df_residual": df_res,
           "rank_reduced": rz, "b": b, "executed": PERMUTATIONS if defined else 0}
    if defined:
        out["vegan_margin_raw_b"] = vegan_margin_raw(X, Z, W, df_term, df_res, f0, seed)
        out["vegan_partial_permutest_b"] = vegan_partial_permutest(X, term, conditions, seed)
    return out


def vegan_margin_raw(X, Z, W, df_term, df_res, f0, seed) -> int:
    # anovaCCAbymargin: same raw permutation for the full model and the model
    # without the term; F of change against the permuted full residual.
    rng = np.random.default_rng(seed)
    b = 0
    Xc = X - X.mean(axis=0)
    for _ in range(PERMUTATIONS):
        Y = Xc[rng.permutation(X.shape[0])]
        rf = _rss(W, Y)
        f = ((_rss(Z, Y) - rf) / df_term) / (rf / df_res) if rf > 0 else np.inf
        b += f >= f0 - TOL
    return int(b)


def vegan_partial_permutest(X, term, conditions, seed) -> int:
    # getF.c: rows of the reduced-model residuals E and of the conditions Z
    # are permuted together; Y and the fixed term X are residualized on the
    # permuted Z; F = (cca/q)/(ca/r) with r = n - rank([Z X]) - 1.
    n = X.shape[0]
    Zc = np.hstack([one_hot(c) for c in conditions])
    Zc = Zc - Zc.mean(axis=0)
    Tc = one_hot(term) - one_hot(term).mean(axis=0)
    q = _rank(np.hstack([Zc, Tc])) - _rank(Zc)
    r = n - _rank(np.hstack([Zc, Tc])) - 1

    def resid(D, Y):
        beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
        return Y - D @ beta

    def fstat(E, Zp):
        Y = resid(Zp, E)
        Xr = resid(Zp, Tc)
        beta, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
        fit = Xr @ beta
        return (np.sum(fit**2) / q) / (np.sum((Y - fit) ** 2) / r)

    E = resid(Zc, X - X.mean(axis=0))
    f0 = fstat(E, Zc)
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(PERMUTATIONS):
        p = rng.permutation(n)
        b += fstat(E[p], Zc[p]) >= f0 - TOL
    return int(b)


def skbio_checks(X: np.ndarray, term: np.ndarray, ref: dict, disp: dict | None) -> dict:
    d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    ids = [f"r{i}" for i in range(X.shape[0])]
    dm = DistanceMatrix(d, ids)
    grouping = [str(t) for t in term]
    out = {"skbio_permanova_pseudo_f": float(permanova(dm, grouping, permutations=0)["test statistic"])}
    assert abs(out["skbio_permanova_pseudo_f"] - ref["pseudo_f"]) <= 1e-9 * max(1.0, ref["pseudo_f"]), out
    if disp is not None:
        out["skbio_permdisp_f"] = float(permdisp(dm, grouping, test="centroid", permutations=0)["test statistic"])
        assert abs(out["skbio_permdisp_f"] - disp["f"]) <= 1e-4 * disp["f"], out
    return out


def make_case(name, X, term, strata, seed, with_disp, note, skbio=True, conditions=None):
    has_strata = strata is not None
    s = strata if has_strata else np.zeros(X.shape[0], dtype=np.int64)
    ref = decompose(X, term, s)
    disp = permdisp_reference(X, term, seed + 1) if with_disp else None
    case = {
        "name": name, "note": note, "rows": int(X.shape[0]), "cols": int(X.shape[1]),
        "X": hexd(X), "term": [int(v) for v in term],
        "strata": [int(v) for v in strata] if has_strata else None,
        "seed": seed, "n_permutations": PERMUTATIONS,
        "permanova": {
            **{key: hexd([ref[key]])[0] for key in
               ("ss_total", "ss_strata", "ss_term", "ss_residual", "r2", "partial_r2", "pseudo_f")},
            "df_term": ref["df_term"], "df_residual": ref["df_residual"],
            "n_levels": int(np.unique(term).size), "n_strata": int(np.unique(s).size),
            "log10_distinct": hexd([log10_distinct(term, s)])[0],
            "b": permanova_counts(X, term, s, seed),
            "executed": PERMUTATIONS if np.isfinite(ref["pseudo_f"]) else 0,
        },
    }
    if disp is not None:
        case["permdisp"] = {
            "seed": seed + 1, "distances": hexd(disp["distances"]),
            "group_mean_distances": hexd(disp["group_mean_distances"]),
            "f": hexd([disp["f"]])[0], "df1": disp["df1"], "df2": disp["df2"],
            "n_groups": disp["n_groups"], "log10_distinct": hexd([disp["log10_distinct"]])[0],
            "b": disp["b"],
            "b_if_compared_on_u_informational": disp["b_u_rule"],
        }
    if skbio and not has_strata and ref["defined"]:
        case["informational"] = skbio_checks(X, term, ref, disp)
    if conditions is not None:
        fl = freedman_lane(X, term, conditions, seed + 2)
        case["freedman_lane"] = {
            "conditions": [[int(v) for v in c] for c in conditions], "seed": seed + 2,
            **{key: hexd([fl[key]])[0] for key in
               ("ss_total", "ss_reduced", "ss_term", "ss_residual", "pseudo_f")},
            **{key: fl[key] for key in ("df_term", "df_residual", "rank_reduced", "b", "executed")},
            "informational": {key: fl[key] for key in ("vegan_margin_raw_b", "vegan_partial_permutest_b")
                              if key in fl},
        }
    return case


def main() -> None:
    for package, version in PINNED.items():
        if metadata.version(package) != version:
            raise SystemExit(f"{package}=={version} required, found {metadata.version(package)}")
    rng = np.random.default_rng(SEED)
    cases = []

    # Three unbalanced groups (8/10/12) in 3-D, arbitrary int64 codes.
    sizes, codes = [8, 10, 12], [-5, 40, 7]
    term = np.repeat(codes, sizes).astype(np.int64)
    centers = {-5: [0.0, 0.0, 0.0], 40: [0.35, 0.2, 0.0], 7: [0.1, -0.3, 0.15]}
    X = np.array([centers[t] for t in term]) + rng.normal(scale=0.7, size=(term.size, 3))
    cases.append(make_case("one_factor_three_groups", X, term, None, 11, True,
                           "free permutation; skbio cross-check"))

    # Two crossed factors, unbalanced cells; the term is tested within strata.
    cells = {(0, 10): 8, (0, 20): 4, (1, 10): 3, (1, 20): 7, (2, 10): 6, (2, 20): 8}
    term = np.concatenate([np.full(c, a) for (a, _), c in cells.items()]).astype(np.int64)
    strata = np.concatenate([np.full(c, s) for (_, s), c in cells.items()]).astype(np.int64)
    X = (np.array([[0.0, 0.15, 0.0, 0.0], [0.25, 0.0, 0.1, 0.0], [0.1, -0.2, 0.0, 0.1]])[term]
         + np.where(strata[:, None] == 20, 0.8, 0.0) + rng.normal(scale=0.6, size=(term.size, 4)))
    cases.append(make_case("two_factor_conditional", X, term, strata, 12, True,
                           "term given strata = adonis2 by='margin' for term + strata", conditions=[strata]))
    cases.append(make_case("two_factor_free", X, term, None, 13, False,
                           "same data, term alone with free permutation"))

    # Four term levels split across two strata: components 2, df_term = 2 of 3.
    term = np.repeat([0, 1, 2, 3, 0, 1, 2, 3], 3).astype(np.int64)
    strata = np.array([0] * 6 + [1] * 6 + [0] * 6 + [1] * 6, dtype=np.int64)
    X = term[:, None] * np.array([0.5, -0.2]) + rng.normal(scale=0.5, size=(term.size, 2))
    cases.append(make_case("partially_aliased", X, term, strata, 14, False,
                           "levels {0,1} only in stratum 0 and {2,3} only in stratum 1", conditions=[strata]))

    # Term identical to strata: aliased, no identifiable effect, no draws.
    term = np.repeat([3, 9, 27], 5).astype(np.int64)
    X = rng.normal(size=(term.size, 2))
    cases.append(make_case("fully_aliased", X, term, term.copy(), 15, False,
                           "term = strata: df_term 0, p undefined", conditions=[term.copy()]))

    # Singleton stratum, a stratum where the term is constant, and mixed ones.
    strata = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int64)
    term = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int64)
    X = term[:, None] * np.array([0.35, 0.05, -0.1]) + rng.normal(scale=0.8, size=(term.size, 3))
    cases.append(make_case("singleton_and_constant_strata", X, term, strata, 16, False,
                           "stratum 0 has one row, stratum 1 a constant term", conditions=[strata]))

    # Astra review: near U = B/(B+W) = 1 an additive tolerance on U merges
    # very different F values; vegan compares F. Expected b = 0 (U rule: 22).
    X = np.array([-1001.0, -999.0, 999.0, 1001.0,
                  -1000001.0001, -999998.9999, 999998.9999, 1000001.0001])[:, None]
    term = np.repeat([0, 1], 4).astype(np.int64)
    cases.append(make_case("permdisp_boundary", X, term, None, 0, True,
                           "PERMDISP seed 1: F comparison (vegan) versus U comparison",
                           skbio=False))  # scikit-bio PCoA rejects this rank-1 matrix

    # Equal centroids, unequal spreads: dispersion differs, location does not.
    term = np.repeat([1, 2, 3], 10).astype(np.int64)
    X = rng.normal(size=(30, 2)) * np.array([0.85, 1.0, 1.25])[term - 1][:, None]
    cases.append(make_case("dispersion_only", X, term, None, 17, True,
                           "heteroscedastic groups with a common centre"))

    # Three factors: term (3 levels) given additive conditions B (2) and C (3),
    # unbalanced and incompletely crossed; the cells of B x C are the strata.
    term = rng.integers(0, 3, size=40).astype(np.int64)
    b_cond = rng.integers(0, 2, size=40).astype(np.int64)
    c_cond = rng.integers(0, 3, size=40).astype(np.int64)
    X = (np.array([[0.0, 0.2], [0.25, 0.0], [0.1, -0.15]])[term]
         + np.where(b_cond[:, None] == 1, 0.5, 0.0) + 0.3 * c_cond[:, None]
         + rng.normal(scale=0.5, size=(40, 2)))
    cases.append(make_case("three_factor_additive", X, term, b_cond * 3 + c_cond, 18, False,
                           "Freedman-Lane given additive B + C; strata are the B x C cells",
                           conditions=[b_cond, c_cond]))

    # Reduced model saturated: every unit is its own condition level.
    term = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    X = rng.normal(size=(6, 2))
    cases.append(make_case("reduced_saturated", X, term, np.arange(6, dtype=np.int64), 19, False,
                           "rank(reduced) = n: df_term 0, no test",
                           conditions=[np.arange(6, dtype=np.int64)]))

    # Full model saturated: rank(full) = n, df_residual 0, F undefined.
    term = np.array([0, 1, 1], dtype=np.int64)
    cond = np.array([0, 0, 1], dtype=np.int64)
    X = rng.normal(size=(3, 2))
    cases.append(make_case("full_saturated", X, term, cond, 20, False,
                           "rank(full) = n: df_residual 0, no test", conditions=[cond]))

    fixture = {
        "format": "n4m_permanova_v1",
        "reference": {**PINNED, "python": sys.version.split()[0], "pandas": metadata.version("pandas"),
                      "seed": SEED, "n_permutations": PERMUTATIONS, "tie_tolerance": "sqrt(DBL_EPSILON)"},
        "encoding": "ieee754_binary64_be_hex",
        "cases": cases,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "fixtures" / "permanova_v1.json")
    out.write_text(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
