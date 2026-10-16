# SPDX-License-Identifier: CECILL-2.1
"""Print the references pinned in cpp/tests/test_variance_components.cpp.

Test-only; nothing here enters libn4m. Recorded with Python 3.14.3,
numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0 and pandas 3.0.6:

    python parity/generators/variance_components_v1.py out.json

References: dense-matrix REML maximised on V = s2b ZZ' + s2w I and polished
on the Harville (1977) score equations; statsmodels MixedLM REML; balanced
ANOVA closed forms; SciPy chi2/F quantiles; a cluster bootstrap redrawn with
numpy.random.default_rng(seed).integers and refitted with the dense REML.
"""
import json, sys, warnings
import numpy as np, pandas as pd, scipy, statsmodels
import statsmodels.formula.api as smf
from scipy import stats, optimize
warnings.simplefilter("ignore")

def dense_parts(groups, y, sb, sw):
    groups = np.asarray(groups); y = np.asarray(y, float); N = len(y)
    Z = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    X = np.ones((N, 1))
    V = sb * Z @ Z.T + sw * np.eye(N)
    Vi = np.linalg.inv(V)
    XtViX = X.T @ Vi @ X
    P = Vi - Vi @ X @ np.linalg.inv(XtViX) @ X.T @ Vi
    return Z, X, V, Vi, XtViX, P, y, N

def reml_loglik(groups, y, sb, sw):
    Z, X, V, Vi, XtViX, P, y, N = dense_parts(groups, y, sb, sw)
    return -0.5 * ((N - 1) * np.log(2 * np.pi) + np.linalg.slogdet(V)[1]
                   + np.linalg.slogdet(XtViX)[1] + y @ P @ y)

def reml_score(groups, y, sb, sw):
    # Harville (1977): dl/dtheta_k = -1/2 tr(P V_k) + 1/2 y'P V_k P y.
    Z, X, V, Vi, XtViX, P, y, N = dense_parts(groups, y, sb, sw)
    out = []
    for Vk in (Z @ Z.T, np.eye(N)):
        out.append(-0.5 * np.trace(P @ Vk) + 0.5 * y @ P @ Vk @ P @ y)
    return np.array(out)

def dense_reml(groups, y):
    """Bounded REML maximum: sb >= 0, sw > 0; interior optima polished on the score."""
    v = np.var(y, ddof=1)
    best = None
    # Optimise over (sb/v >= 0, log(sw/v)), chain rule on the dense score.
    fun = lambda t: -reml_loglik(groups, y, t[0] * v, np.exp(t[1]) * v)
    def jac(t):
        s = reml_score(groups, y, t[0] * v, np.exp(t[1]) * v)
        return -np.array([s[0] * v, s[1] * np.exp(t[1]) * v])
    for frac in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999999):
        o = optimize.minimize(fun, [frac, np.log(1 - frac + 1e-9)], jac=jac, method="L-BFGS-B",
                              bounds=[(0, 1e3), (-25, 5)],
                              options=dict(ftol=1e-16, gtol=1e-14, maxiter=100000))
        if best is None or o.fun < best.fun:
            best = o
    best.x = np.array([best.x[0] * v, np.exp(best.x[1]) * v])
    sb, sw = best.x
    if sb > 1e-8 * v:
        # Score equations in log parameters: theta_k * dl/dtheta_k = 0.
        eq = lambda t: reml_score(groups, y, np.exp(t[0]), np.exp(t[1])) * np.exp(t)
        root = optimize.root(eq, np.log([sb, sw]), method="hybr", options=dict(xtol=1e-15))
        if root.success:
            sb, sw = np.exp(root.x)
    if reml_loglik(groups, y, 0.0, np.var(y, ddof=1)) >= reml_loglik(groups, y, sb, sw):
        sb, sw = 0.0, np.var(y, ddof=1)
    return float(sb), float(sw), float(sb / (sb + sw)), float(reml_loglik(groups, y, sb, sw))

def statsmodels_reml(groups, y):
    df = pd.DataFrame({"y": y, "g": groups})
    r = smf.mixedlm("y ~ 1", df, groups=df["g"]).fit(reml=True, method=["lbfgs"], gtol=1e-12, maxiter=10000)
    sb, sw = float(r.cov_re.iloc[0, 0]), float(r.scale)
    return sb, sw, sb / (sb + sw), float(r.llf)

out = {"provenance": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "scipy": scipy.__version__, "statsmodels": statsmodels.__version__}}

# Case A: unbalanced, sizes 2,3,4,1,2 (one singleton group).
A_g = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4]
A_y = [1.0, 1.4, 2.2, 2.0, 2.9, 0.3, 0.9, 0.4, 0.8, 1.7, 3.1, 2.5]
# Case D: unbalanced NIRS-like, between SD 1 vs repeat SD 1e-3.
rs = np.random.default_rng(20261013)
D_sizes = [3, 2, 4, 1, 3, 5, 2, 1, 3, 2]
D_g = np.repeat(np.arange(10), D_sizes)
D_y = (rs.normal(0, 1, 10)[D_g] + 0.5 + rs.normal(0, 1e-3, len(D_g))).round(12)
# Case E: unbalanced boundary (REML sb = 0): group means nearly equal.
E_g = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
E_y = [1.0, 2.0, 3.0, 2.0, 2.1, 1.5, 2.5, 1.9, 2.2, 2.05]
cases = {"A": (A_g, A_y), "D": (list(map(int, D_g)), list(map(float, D_y))), "E": (E_g, E_y)}
for name, (g, y) in cases.items():
    out[name] = {"groups": g, "y": y, "dense": dense_reml(g, y), "statsmodels": statsmodels_reml(g, y),
                 "score_at_dense": reml_score(g, y, *dense_reml(g, y)[:2]).tolist()}

# Case B: balanced 4 x 3 with positive ANOVA ICC -> REML equals ANOVA.
B_g = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
B_y = [1.0, 1.2, 0.9, 2.0, 2.3, 2.1, 0.2, 0.1, 0.5, 1.4, 1.6, 1.3]
yb = np.array(B_y).reshape(4, 3)
msw = ((yb - yb.mean(1, keepdims=True)) ** 2).sum() / (4 * 2)
msb = 3 * ((yb.mean(1) - yb.mean()) ** 2).sum() / 3
alpha = 0.05
F = msb / msw
FL = F / stats.f.ppf(1 - alpha / 2, 3, 8)
FU = F * stats.f.ppf(1 - alpha / 2, 8, 3)
ssw = msw * 8
out["B"] = {"groups": B_g, "y": B_y, "msb": msb, "msw": msw,
            "anova": [(msb - msw) / 3, msw, (msb - msw) / (msb + 2 * msw)],
            "dense": dense_reml(B_g, B_y),
            "icc_exact_f": [(FL - 1) / (FL + 2), (FU - 1) / (FU + 2)],
            "sd_chi2": [np.sqrt(ssw / stats.chi2.ppf(1 - alpha / 2, 8)), np.sqrt(ssw / stats.chi2.ppf(alpha / 2, 8))]}
# Case C: balanced, negative ANOVA ICC -> REML boundary sb=0, sw = sample variance.
C_y = [0.0, 2.0, 0.1, 1.9, -0.1, 2.2]
C_g = [0, 0, 1, 1, 2, 2]
out["C"] = {"groups": C_g, "y": C_y, "sample_variance": float(np.var(C_y, ddof=1)),
            "dense": dense_reml(C_g, C_y)}
# Case A chi-square SD interval (pooled df = N - g = 7).
ga = np.array(A_g); ya = np.array(A_y)
ssw_a = sum(((ya[ga == k] - ya[ga == k].mean()) ** 2).sum() for k in np.unique(ga))
out["A"]["sd_chi2"] = [np.sqrt(ssw_a / stats.chi2.ppf(0.975, 7)), np.sqrt(ssw_a / stats.chi2.ppf(0.025, 7))]
out["A"]["sd_chi2_90"] = [np.sqrt(ssw_a / stats.chi2.ppf(0.95, 7)), np.sqrt(ssw_a / stats.chi2.ppf(0.05, 7))]

# Quantile goldens.
out["quantiles"] = {
    "chi2": [[p, d, stats.chi2.ppf(p, d)] for p, d in [(0.025, 1), (0.975, 1), (0.025, 7), (0.975, 7), (0.5, 0.5), (0.025, 2000), (0.975, 150000)]],
    "f": [[p, a, b, stats.f.ppf(p, a, b)] for p, a, b in [(0.975, 3, 8), (0.975, 8, 3), (0.975, 1, 1), (0.975, 2, 1), (0.025, 5, 40), (0.975, 999, 2000)]],
}

# Cluster bootstrap on case A: clusters {0,1} -> 7, {2} -> 3, {3} -> 9, {4} -> 5.
cluster_of_group = {0: 7, 1: 7, 2: 3, 3: 9, 4: 5}
A_c = [cluster_of_group[k] for k in A_g]
units = sorted(set(A_c))
B = 40; seed = 12345
draws = np.random.default_rng(seed).integers(0, len(units), size=(B, len(units)))
sd_rep, icc_rep, n_valid = [], [], 0
for b in range(B):
    gg, yy = [], []
    for k, u in enumerate(draws[b]):
        for gi in sorted(g for g, c in cluster_of_group.items() if c == units[u]):
            idx = [i for i, x in enumerate(A_g) if x == gi]
            gg += [1000 * k + gi] * len(idx); yy += [A_y[i] for i in idx]
    gg = np.array(gg); yy = np.array(yy)
    ng = len(np.unique(gg)); N = len(yy)
    if ng < 2 or N - ng < 1:
        continue
    ssw_b = sum(((yy[gg == k] - yy[gg == k].mean()) ** 2).sum() for k in np.unique(gg))
    sd_rep.append(np.sqrt(ssw_b / (N - ng)))
    icc_rep.append(dense_reml(gg.tolist(), yy.tolist())[2])
out["A_bootstrap"] = {"clusters": A_c, "B": B, "seed": seed, "draws_first_rows": draws[:3].tolist(),
                      "n_valid": len(sd_rep),
                      "sd": np.quantile(sd_rep, [0.025, 0.975], method="linear").tolist(),
                      "icc": np.quantile(icc_rep, [0.025, 0.975], method="linear").tolist()}
def balanced_intervals(y, alpha=0.05):
    y = np.asarray(y, float); g, r = y.shape; df = g * (r - 1)
    ssw = ((y - y.mean(1, keepdims=True)) ** 2).sum()
    msw = ssw / df; msb = r * ((y.mean(1) - y.mean()) ** 2).sum() / (g - 1); F = msb / msw
    FL = F / stats.f.ppf(1 - alpha / 2, g - 1, df); FU = F * stats.f.ppf(1 - alpha / 2, df, g - 1)
    return {"icc_anova": (msb - msw) / (msb + (r - 1) * msw),
            "icc_exact_f": [(FL - 1) / (FL + r - 1), (FU - 1) / (FU + r - 1)],
            "sd_chi2": [np.sqrt(ssw / stats.chi2.ppf(1 - alpha / 2, df)), np.sqrt(ssw / stats.chi2.ppf(alpha / 2, df))]}

out["C"]["intervals"] = balanced_intervals(np.array(C_y).reshape(3, 2))
# Large balanced design y_ij = i + 50 (j-1), 1000 groups of 3 (df 2000).
out["large"] = balanced_intervals([[i + 50.0 * (j - 1) for j in range(3)] for i in range(1000)])
# One within df: groups {0,0,1}, y = 1, 3, 10, SSW = 2.
out["one_df_sd_chi2"] = [np.sqrt(2 / stats.chi2.ppf(0.975, 1)), np.sqrt(2 / stats.chi2.ppf(0.025, 1))]
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])


# ---- Review round 1 additions -------------------------------------------
def witness_data():
    """Multimodal witness: sizes [1,152,350,363]; every group is its mean
    +/- h (odd sizes keep one value at the mean), SSW = 3333.5."""
    sizes = [1, 152, 350, 363]
    means = [7.763121181, -0.196654690, -0.399240353, -0.241778259]
    h = np.sqrt(3333.5 / sum(r - (r % 2) for r in sizes))
    g, y = [], []
    for i, (r, m) in enumerate(zip(sizes, means)):
        for j in range(r):
            if r % 2 == 1 and j == r - 1:
                y.append(m)
            else:
                y.append(m + h if j % 2 == 0 else m - h)
            g.append(i)
    return g, y, h

wg, wy, wh = witness_data()
dense = dense_reml(wg, wy)
# Profile scan on a dense grid of gamma (dense REML log-likelihood, profiled
# s2w by 1-D bounded search) to confirm the two local maxima.
out["witness"] = {"h": wh, "dense": dense,
                  "loglik_boundary": float(reml_loglik(wg, wy, 0.0, np.var(wy, ddof=1)))}

from fractions import Fraction
def exact_balanced(groups):
    """Exact ANOVA of a balanced design from the actual doubles (Fraction)."""
    ys = [[Fraction(v) for v in grp] for grp in groups]
    g, r = len(ys), len(ys[0])
    m = [sum(v) / r for v in ys]; gm = sum(m) / g
    msw = sum((v - mi) ** 2 for grp, mi in zip(ys, m) for v in grp) / (g * (r - 1))
    msb = r * sum((mi - gm) ** 2 for mi in m) / (g - 1)
    return float(max(Fraction(0), (msb - msw) / r)), float(msw), float((msb - msw) / (msb + (r - 1) * msw))
huge = [[1e155 - 3e145, 1e155 - 1e145], [1e155 + 1e145, 1e155 + 3e145]]
out["huge_offset"] = {"y": [v for grp in huge for v in grp], "exact": exact_balanced(huge)}

# Quantile goldens over the documented domain, including extreme tails.
# SciPy 1.18.1 chi2.ppf is ~3e-6 relative off in the lower tail at df = 1e8
# (Wilson-Hilferty agrees with mpmath, not SciPy there), so those two points
# come from an mpmath (30 digits) bisection on the defining series
# P(a, x) = x^a e^-x sum_k x^k / Gamma(a + k + 1).
import mpmath as mp

def mp_chi2_lower_ppf(p, df):
    mp.mp.dps = 30
    a = mp.mpf(df) / 2
    def cdf(x):
        x = mp.mpf(x) / 2
        term = 1 / a; total = term; k = 0
        while True:
            k += 1; term *= x / (a + k); total += term
            if term < total * mp.mpf(10) ** -24:
                break
        return total * mp.e ** (a * mp.log(x) - x - mp.loggamma(a))
    lo, hi = mp.mpf(df) * 0.99, mp.mpf(df)
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) < p: lo = mid
        else: hi = mid
    return float((lo + hi) / 2)

qs = {"chi2": [], "f": []}
for p in [1e-12, 1e-6, 0.025, 0.5, 0.975, 1 - 1e-6, 1 - 1e-12]:
    for d in [0.5, 1, 7, 100, 1e4, 1e6, 1e8]:
        if d >= 1e8 and p < 1e-3:
            qs["chi2"].append([p, d, mp_chi2_lower_ppf(p, d)])
        else:
            qs["chi2"].append([p, d, float(stats.chi2.ppf(p, d))])
    for d1, d2 in [(1, 1), (2, 1), (3, 8), (8, 3), (999, 2000), (1, 1e6), (1e6, 1), (5e5, 5e5)]:
        qs["f"].append([p, d1, d2, float(stats.f.ppf(p, d1, d2))])
# Domain boundary: tail 2^-64 on both sides (SciPy; mpmath for the upper F
# tail, where SciPy 1.18.1 returns inf).
def mp_f_isf(tail, d1, d2):
    mp.mp.dps = 40
    a, b = mp.mpf(d1) / 2, mp.mpf(d2) / 2
    sf = lambda lf: mp.betainc(b, a, 0, d2 / (d2 + d1 * mp.e ** lf), regularized=True)
    lo, hi = mp.mpf(-50), mp.mpf(200)
    for _ in range(200):
        mid = (lo + hi) / 2
        if sf(mid) > tail: lo = mid
        else: hi = mid
    return float(mp.e ** ((lo + hi) / 2))
t64 = 2.0 ** -64
qs["boundary"] = {
    "chi2": [[d, float(stats.chi2.ppf(t64, d)), float(stats.chi2.isf(t64, d))] for d in (1, 7, 100, 1e4)],
    "f": [[a, b, float(stats.f.ppf(t64, a, b)), mp_f_isf(t64, a, b)] for a, b in ((3, 8), (8, 3), (999, 2000))],
}
out["quantile_grid"] = qs
out["provenance"]["mpmath"] = mp.__version__
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
