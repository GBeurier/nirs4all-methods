# Tier C pilot — uve_select R-parity: findings & stop decision

Goal: drive the uve_select pilot to the point of a clear go/no-go on bit-exact
Jaccard-1.0 parity with R `plsVarSel::mcuve_pls`, additively, no regression.

## What was built and PROVEN (committed, branch feat/rng-r-parity)

All RNG primitives `mcuve_pls` consumes are now bit-exact vs base R (R 4.3.3):
- `runif` (R-MT, scale 1/2^32) — bit-exact.
- `rnorm` (Inversion + Wichura qnorm5) — bit-exact.
- **`sample(n)`** (R 3.6+ "Rejection" `R_unif_index` + R's Fisher-Yates) —
  bit-exact: `set.seed(11); sample(10)` → `10 2 8 1 7 5 4 9 3 6`, matched.
  (`R_unif_index` = reject-resample of `ceil(log2(dn))` bits from 16-bit `unif`
  chunks; the permutation swaps `x[j]=x[--n]`.)

So the foundational risk — "can we reproduce R's RNG bit-for-bit in C?" — is
**solved** for the whole R-selector family, not just uve.

## The decisive experiment

R's exact algorithm (decoded from `deparse(mcuve_pls)`):
1. `W <- matrix(runif(Nx*Mx,0,1), Mx, Nx)` — Nx U(0,1) noise cols, **not**
   standardized (n4m kernel: `noise_features` *signed* uniform cols, **standardized**).
2. per iter `temp <- sample(Mx); calk <- temp[1:floor(Mx*ratio)]`.
3. `plsr(ycal ~ Zcal, validation="LOO")`; `opt.comp <- which.min(PRESS)`;
   coefficients at `opt.comp`.
4. `RI = mean/sd` over iters; threshold `max(|RI[noise]|)`; selection
   `which(|RI[real]| > threshold)` + a too-few-selected fallback.

On a fixed 40×12 dataset (signal in features 3 & 8), MEASURED (not assumed):
- R `mcuve_pls` (set.seed(11), N=3, ratio=0.75, ncomp=5) selects **{3,4,5,8,9}**
  (1-based) = **{2,3,4,7,8}** 0-based. R's RI for the real features is
  `[-1.02, 0.34, 34.2, 1.74, -2.48, 1.03, 0.58, -36.6, -1.03, -0.61, -0.79, -0.28]`
  with noise threshold `max|RI_noise| = 4.99`: features 3 & 8 dominate, and
  4, 5, 9 are the borderline ones near the threshold.
- n4m's C kernel (different noise model + splitmix RNG) selects **{0,2,4,7,8,11}**
  (0-based) = {1,3,5,8,9,12} 1-based.
- Overlap {2,4,7,8} / union {0,2,3,4,7,8,11} → **Jaccard 4/7 ≈ 0.57.**

So the honest result is NOT "already matches": the strong features (3, 8) agree,
but the BORDERLINE features near the noise threshold (R's 4,5,9 vs n4m's 1,12)
diverge — exactly because n4m's noise model (standardized signed-uniform,
splitmix) and LOO-PRESS path differ from R's (unstandardized U(0,1), R-MT).
This is the dashboard's Jaccard-0.75-class gap, reproduced.

## What it takes to close it — and the building blocks all exist

To reach Jaccard 1.0, n4m's UVE C kernel must replicate R exactly. Every piece
is now available in n4m:
- Nx unstandardized `runif(0,1)` noise cols — **RNG have** (`rng_mt_r_unif`, bit-exact).
- `sample()` subsampling — **have** (`rng_mt_r_sample`, golden-tested bit-exact).
- `pls::plsr` SIMPLS coefficients — **have** (n4m PLS ~1e-13 vs pls::plsr).
- per-subsample **LOO-PRESS `which.min(opt.comp)`** — **have**
  (`n4m_approximate_press_compute`, legacy=0, documented to match
  `pls::plsr(validation='LOO', method='simpls', scale=FALSE)` bit-for-bit,
  cpp/src/core/extra_pls.cpp:3097). The fragile bit is the discrete argmin
  (can flip on ~1e-12 PRESS ties), but the PRESS values themselves are available.
- RI mean/sd + threshold + too-few fallback — trivial.

So a `uve_legacy=0` R-exact path is **feasible additively** (new code path behind
a config flag; the current splitmix kernel stays as the default), and the only
real risk is the LOO-PRESS argmin tie-sensitivity. Estimated ~1 day for the uve
path itself + a fixture parity test vs R.

## Stop decision (per maintainer: "selon les difficultés rencontrées, on arrête")

This is the difficulty checkpoint. Status:
- **Foundation: DONE + proven** (3 RNG engines bit-exact, unified dispatch with
  frozen splitmix golden, additive ABI `n4m_rng_kind_t` 1.10.0). No regression
  (default splitmix unchanged).
- **uve already matches R on clear signal.** The residual gap is only
  borderline features.
- **The full R-exact uve kernel port** is a multi-day, additive-but-substantial
  rewrite whose success hinges on the fragile LOO-PRESS argmin, and whose payoff
  is only borderline-feature agreement.

**Recommendation: STOP the pilot here.** The infrastructure the maintainer asked
for ("RNG as an optional model parameter + additional RNG engines + tests seeded
with the right RNG") is built, proven, and additive. The per-method R-exact
algorithm ports (Tier C) are a separate, large, opt-in effort to schedule
deliberately — for the methods where the reference IS the target algorithm
(uve/mcuve), with the borderline-only payoff understood; analog-only refs
(cars vs enpls.fs) should stay documented cross_check.
