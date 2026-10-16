# SPDX-License-Identifier: CECILL-2.1
"""Generate parity/fixtures/domain_shift_v1.json, the energy distance and
PCA monitoring gate (n4m_stats_energy_distance, n4m_stats_pca_monitor).

Test-only. Requires exactly numpy==2.2.6 and scipy==1.15.3 (BSD-3-Clause);
neither enters libn4m. Every input lies in the supported domain (nonzero |x|
in [2^-256, 2^256], max/min nonzero |x| <= 2^32).

The references never reuse the kernels' rounding formulas:

* Energy distance. ``scipy_cdist`` cases: SciPy distances, M[a][b] the mean
  over all ordered pairs (V-statistic), E = 2M_ab - M_aa - M_bb, T =
  sum_{a<b} n_a n_b / (n_a + n_b) E_ab (R ``energy::eqdist.etest``).
  ``rational`` cases (one integer column): M, E and T in exact fractions, and
  every permutation statistic in exact fractions too. Draws:
  ``rng = numpy.random.default_rng(seed)``, ``perm = rng.permutation(n)``,
  row i receives ``codes[perm[i]]``; the declared tie rule counts
  ``T_perm >= T - sqrt(eps) |T|`` (exact comparison for rational cases).
* PCA monitoring. ``numpy`` cases: exact rational centring, then
  ``numpy.linalg.svd``. ``rational`` cases need no decomposition: either the
  covariance is diagonal (components are coordinate axes, T^2 and Q exact
  sums) or k equals the number of columns (T^2 = x' S^-1 x with the exact
  rational inverse covariance, Q = 0 exactly). Limits are type-7 quantiles,
  exact for rational cases. Effective k counts components above the declared
  resolution 16 m eps lambda_1 (m = max(rows, p)); decisions use the declared
  tolerance tau = 16 m eps lambda_1 / lambda_k: a row is beyond when it
  exceeds the limit by more than tau max(value, limit) for T^2 and tau S for
  Q (S the largest squared centred norm among the reference rows and the
  row), and within tolerance otherwise near the limit. Rational cases decide
  with exact values.

Usage: python parity/generators/domain_shift_v1.py [output.json]
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import energy_distance

PINNED = {"numpy": "2.2.6", "scipy": "1.15.3"}
EPS = float(np.finfo(np.float64).eps)
SQRT_EPS = float(np.sqrt(EPS))


def hexd(values) -> list[str]:
    return [struct.pack(">d", float(v)).hex() for v in np.ravel(values)]


def energy_stats(D: np.ndarray, codes: np.ndarray, g: int):
    M = np.zeros((g, g))
    for a in range(g):
        for b in range(g):
            M[a, b] = D[np.ix_(codes == a, codes == b)].mean()
    E = np.zeros((g, g))
    T = 0.0
    sizes = np.bincount(codes, minlength=g)
    for a in range(g):
        for b in range(g):
            if a != b:
                E[a, b] = 2 * M[a, b] - M[a, a] - M[b, b]
            if a < b:
                T += sizes[a] * sizes[b] / (sizes[a] + sizes[b]) * E[a, b]
    return M, E, T


def rational_energy(x: list, codes: np.ndarray, g: int):
    """Exact M, E and T for one integer column (distances |x_i - x_j| exact)."""
    sizes = np.bincount(codes, minlength=g)
    M = [[Fraction(0)] * g for _ in range(g)]
    for a in range(g):
        for b in range(g):
            ia = [x[i] for i in range(len(x)) if codes[i] == a]
            ib = [x[i] for i in range(len(x)) if codes[i] == b]
            M[a][b] = Fraction(sum(abs(u - v) for u in ia for v in ib), len(ia) * len(ib))
    E = [[2 * M[a][b] - M[a][a] - M[b][b] if a != b else Fraction(0) for b in range(g)]
         for a in range(g)]
    T = sum(Fraction(int(sizes[a] * sizes[b]), int(sizes[a] + sizes[b])) * E[a][b]
            for a in range(g) for b in range(a + 1, g))
    return M, E, T


def energy_case(name: str, X: np.ndarray, codes, g: int, B: int, seed: int,
                rational: bool = False) -> dict:
    codes = np.asarray(codes, dtype=np.int64)
    if rational:
        x = [int(v) for v in X[:, 0]]
        Mr, Er, Tr = rational_energy(x, codes, g)
        M = np.array([[float(v) for v in row] for row in Mr])
        E = np.array([[float(v) for v in row] for row in Er])
        T = float(Tr)
        threshold = Tr - Fraction(SQRT_EPS) * abs(Tr)
        statistic = lambda c: rational_energy(x, c, g)[2]
    else:
        D = cdist(X, X)
        M, E, T = energy_stats(D, codes, g)
        threshold = T - SQRT_EPS * abs(T)
        statistic = lambda c: energy_stats(D, c, g)[2]
    rng = np.random.default_rng(seed)
    b = sum(statistic(codes[rng.permutation(len(codes))]) >= threshold for _ in range(B))
    sizes = np.bincount(codes, minlength=g)
    log10 = (np.sum(np.log(np.arange(1, len(codes) + 1))) -
             sum(np.sum(np.log(np.arange(1, s + 1))) for s in sizes)) / np.log(10)
    case = {
        "name": name, "kind": "energy", "rows": X.shape[0], "cols": X.shape[1],
        "n_groups": g, "codes": codes.tolist(), "n_permutations": B, "seed": str(seed),
        "oracle": "rational" if rational else "scipy_cdist",
        "input_hex": hexd(X), "energy_hex": hexd(E), "mean_distances_hex": hexd(M),
        "statistic_hex": hexd([T]), "log10_distinct_hex": hexd([log10]),
        "p_value_hex": hexd([(1 + b) / (1 + B) if B else np.nan]),
        "n_at_least_observed": int(b),
    }
    if X.shape[1] == 1 and g == 2:
        case["scipy_energy_distance_hex"] = hexd(
            [energy_distance(X[codes == 0, 0], X[codes == 1, 0])])
    return case


def quantile7(values: list, q):
    """Type-7 quantile, exact for fractions."""
    v = sorted(values)
    h = (len(v) - 1) * q
    lo = int(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def rational_inverse(S: list) -> list:
    """Gauss-Jordan inverse on fractions."""
    p = len(S)
    A = [list(row) + [Fraction(int(i == j)) for j in range(p)] for i, row in enumerate(S)]
    for col in range(p):
        piv = next(r for r in range(col, p) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        f = A[col][col]
        A[col] = [v / f for v in A[col]]
        for r in range(p):
            if r != col and A[r][col] != 0:
                g = A[r][col]
                A[r] = [a - g * b for a, b in zip(A[r], A[col])]
    return [row[p:] for row in A]


class Fit:
    """Reference fit on rows R: exact rational centring; rational or numpy."""

    def __init__(self, R: np.ndarray, kmax: int, rational: bool):
        n, p = R.shape
        self.rational = rational
        self.mean = [sum(Fraction(float(v)) for v in R[:, j]) / n for j in range(p)]
        C = [[Fraction(float(v)) - m for v, m in zip(row, self.mean)] for row in R]
        if rational:
            S = [[sum(C[i][a] * C[i][b] for i in range(n)) / (n - 1) for b in range(p)]
                 for a in range(p)]
            self.S = S
            diagonal = all(S[a][b] == 0 for a in range(p) for b in range(p) if a != b)
            if diagonal:
                self.axes = sorted(range(p), key=lambda a: -S[a][a])
                self.lam = [float(S[a][a]) for a in self.axes]
            else:
                assert kmax == p, "rational non-diagonal fits need k = p"
                self.axes = None
                self.inverse = rational_inverse(S)
                self.lam = sorted(np.linalg.eigvalsh(np.array(S, float)), reverse=True)
        else:
            _, s, vt = np.linalg.svd(np.array(C, float), full_matrices=False)
            self.vt = vt
            self.lam = list(s ** 2 / (n - 1))
        resolution = 16 * max(n, p) * EPS * max(self.lam[0], 0.0)
        self.resolved = sum(1 for v in self.lam[:kmax] if v > resolution)

    def evaluate(self, x, k: int):
        c = [Fraction(float(v)) - m for v, m in zip(x, self.mean)]
        norm2 = sum(v * v for v in c)
        if self.rational and self.axes is not None:
            t2 = sum(c[a] * c[a] / self.S[a][a] for a in self.axes[:k]) if k else None
            q = sum(c[a] * c[a] for a in self.axes[k:])
            return t2, q, norm2
        if self.rational:
            p = len(c)
            t2 = sum(c[a] * self.inverse[a][b] * c[b] for a in range(p) for b in range(p))
            return t2, Fraction(0), norm2
        cf = np.array([float(v) for v in c])
        V = self.vt[:k]
        t = V @ cf
        r = cf - t @ V
        t2 = float(np.sum(t * t / np.array(self.lam[:k]))) if k else None
        return t2, float(r @ r), float(norm2)


def monitor_case(name: str, R: np.ndarray, Mo: np.ndarray, kmax: int, folds: int,
                 alpha: Fraction, rational: bool = False) -> dict:
    nr, p = R.shape
    fits = []
    for f in range(folds):
        train = np.array([i % folds != f for i in range(nr)])
        fits.append((train, Fit(R[train], kmax, rational)))
    full = Fit(R, kmax, rational)
    k = min([kmax, full.resolved] + [m.resolved for _, m in fits])
    rs = [None] * nr
    for train, fit in fits:
        for i in np.flatnonzero(~train):
            rs[i] = fit.evaluate(R[i], k)
    level = 1 - alpha if rational else float(1 - alpha)
    t2_limit = quantile7([r[0] for r in rs], level) if k else None
    q_limit = quantile7([r[1] for r in rs], level)
    scale = max(r[2] for r in rs)
    cond = full.lam[0] / full.lam[k - 1] if k else 1.0
    tau = 16 * max(nr, p) * EPS * cond
    ms = [full.evaluate(x, k) for x in Mo]

    def decide(value, limit, tol):
        if float(value - limit) > tol:
            return 1
        if float(limit - value) >= tol:
            return 0
        return -1

    counts = [0] * 6
    for t2, q, n2 in ms:
        ot = decide(t2, t2_limit, tau * float(max(t2, t2_limit))) if k else 0
        oq = decide(q, q_limit, tau * float(max(scale, n2)))
        either = 1 if 1 in (ot, oq) else (0 if ot == 0 and oq == 0 else -1)
        for i, flag in enumerate([ot == 1, oq == 1, either == 1, ot == -1, oq == -1, either == -1]):
            counts[i] += int(flag)
    mon_mean = [sum(Fraction(float(v)) for v in Mo[:, j]) / len(Mo) for j in range(p)]
    trace = lambda X, mean: sum((Fraction(float(v)) - m) ** 2 for row in X for v, m in
                                zip(row, mean)) / (len(X) - 1)
    out = [
        float(t2_limit) if k else np.nan,
        float(q_limit),
        float(trace(R, full.mean)),
        float(trace(Mo, mon_mean)) if len(Mo) > 1 else np.nan,
        float(np.sqrt(float(sum((a - b) ** 2 for a, b in zip(mon_mean, full.mean))))),
    ]
    eig = np.full(kmax, np.nan)
    eig[:k] = full.lam[:k]
    f = lambda v: np.nan if v is None else float(v)
    return {
        "name": name, "kind": "pca_monitor", "oracle": "rational" if rational else "numpy",
        "rows": nr, "monitored_rows": Mo.shape[0], "cols": p, "k": kmax, "n_folds": folds,
        "alpha_hex": hexd([float(alpha)]), "effective_k": k, "counts": counts,
        "tau_hex": hexd([tau]), "input_hex": hexd(R), "monitored_hex": hexd(Mo),
        "eigenvalues_hex": hexd(eig),
        "reference_t2_hex": hexd([f(r[0]) for r in rs]),
        "reference_q_hex": hexd([f(r[1]) for r in rs]),
        "monitored_t2_hex": hexd([f(m[0]) for m in ms]),
        "monitored_q_hex": hexd([f(m[1]) for m in ms]), "out_hex": hexd(out),
    }


def low_rank(rng, n: int, p: int, rank: int, noise: float, shift: float = 0.0) -> np.ndarray:
    scores = rng.normal(size=(n, rank)) * np.linspace(3.0, 1.0, rank)
    loadings = np.sin(np.outer(np.arange(1, rank + 1), np.linspace(0, np.pi, p)))
    return scores @ loadings + noise * rng.normal(size=(n, p)) + shift


def main() -> None:
    for package, version in PINNED.items():
        if metadata.version(package) != version:
            raise SystemExit(f"{package} {version} required, found {metadata.version(package)}")
    rng = np.random.default_rng(20261013)
    col = lambda *v: np.array([[float(x)] for x in v])
    cases = []
    x1 = np.r_[rng.normal(0, 1, 7), rng.normal(1.5, 2, 5)][:, None]
    cases.append(energy_case("one_column_two_groups", x1, [0] * 7 + [1] * 5, 2, 999, 11))
    x3 = np.vstack([rng.normal(0, 1, (6, 4)), rng.normal(0.8, 1, (4, 4)),
                    rng.normal(0, 2.5, (7, 4))])
    cases.append(energy_case("three_groups_unequal", x3, [0] * 6 + [1] * 4 + [2] * 7, 3, 999,
                             20261013))
    xn = rng.normal(size=(20, 3))
    cases.append(energy_case("null_two_groups", xn, [0, 1] * 10, 2, 999, 7))
    cases.append(energy_case("identical_rows", np.ones((6, 2)), [0, 0, 1, 1, 1, 0], 2, 50, 3))
    cases.append(energy_case("exact_ties", col(0, 0, 1, 1), [0, 0, 1, 1], 2, 200, 5,
                             rational=True))
    cases.append(energy_case("identical_distributions", col(0, 1, 1, 0), [0, 0, 1, 1], 2, 0, 1,
                             rational=True))
    cases.append(energy_case("large_constant_column",
                             np.array([[2.0 ** 31, 0.0], [2.0 ** 31, 0.0], [2.0 ** 31, 1.0],
                                       [2.0 ** 31, 1.0]]), [0, 0, 1, 1], 2, 0, 1))
    cases.append(energy_case("singleton_offset", col(2 ** 31, 0, 1), [0, 1, 2], 3, 0, 1,
                             rational=True))
    cases.append(energy_case("cancellation", col(0, -4, -7, 2, 2, 2 ** 31, 1, -4, -7, 2, 2, 2 ** 31),
                             [0] * 6 + [1] * 6, 2, 0, 1, rational=True))
    cases.append(energy_case("symmetric_half", col(0, 2 ** 31, 1, 2 ** 31), [0, 0, 1, 1], 2, 0, 1,
                             rational=True))
    cases.append(energy_case("mixed_exponents_energy", col(
        2 ** 30, 3, -5, 2 ** 30 + 1, 7, 2 ** 20, -2 ** 30, 1), [0, 1, 0, 1, 2, 2, 0, 1], 3, 199, 9,
        rational=True))
    # Declared relative tie rule with large distances (Astra b10fix2 P0-2,
    # inside the 2^32 domain): exact rational permutation statistics.
    cases.append(energy_case("relative_tie_rule", col(0, 1, 2, 2 ** 31, 100, 101, 102, 2 ** 31),
                             [0] * 4 + [1] * 4, 2, 999, 42, rational=True))
    a = Fraction(1, 10)
    cases.append(monitor_case("wide_low_rank", low_rank(rng, 23, 30, 4, 0.05),
                              low_rank(rng, 7, 30, 4, 0.05, 0.3), 3, 5, a))
    cases.append(monitor_case("leave_one_out", low_rank(rng, 6, 8, 5, 0.1),
                              low_rank(rng, 4, 8, 5, 0.1), 4, 6, Fraction(1, 5)))
    cases.append(monitor_case("tall", low_rank(rng, 40, 6, 3, 0.2),
                              low_rank(rng, 9, 6, 3, 0.2, 1.0), 2, 10, Fraction(1, 20)))
    cases.append(monitor_case("one_monitored_row", low_rank(rng, 12, 10, 3, 0.1),
                              low_rank(rng, 1, 10, 3, 0.1), 3, 4, a))
    # Rational cases: axis-aligned covariance (T^2 and Q exact) or k = p.
    line = np.column_stack([np.arange(-5.0, 6.0), np.zeros(11)])
    cases.append(monitor_case("line_reference", line,
                              np.array([[0.0, 100.0], [1.0, 2.0 ** -18], [2.0, 0.0], [9.0, 0.0]]),
                              1, 5, Fraction(1, 20), rational=True))
    cases.append(monitor_case("level_reference_unit_targets",
                              np.array([[2.0 ** 31 + 256 * i] for i in range(-5, 6)]),
                              np.array([[2.0 ** 31 - 2.0 ** 20], [2.0 ** 31 - 2.0 ** 20 + 1]]),
                              1, 5, Fraction(1, 20), rational=True))
    av = [0, 2, 4, -1, -5, 0, -3, 5]
    bv = [-3, 5, -2, -1, 5, 5, -1, -1]
    near = np.array([[a_, a_ + b_ * 2.0 ** -20] for a_, b_ in zip(av, bv)])
    # Astra b10fix2 P0-1: exact limit 48245/7384; the first target is within
    # the tolerance of it, the second clearly beyond.
    cases.append(monitor_case("ill_conditioned_full_rank", near,
                              np.array([[-8.364198114193899, -8.364194618854127], [20.0, 20.0]]),
                              2, 4, Fraction(1, 8), rational=True))
    cases.append(monitor_case("rank_zero_reference", np.ones((6, 3)),
                              np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]), 1, 3, Fraction(1, 5),
                              rational=True))
    doc = {
        "fixture": "n4m_domain_shift_v1",
        "generator": "parity/generators/domain_shift_v1.py",
        "references": {k: v for k, v in PINNED.items()},
        "tie_rule": "T_perm >= T - sqrt(eps) |T|",
        "cases": cases,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[1] / "fixtures" / "domain_shift_v1.json")
    out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
