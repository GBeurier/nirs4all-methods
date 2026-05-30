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

On a fixed 40×12 dataset (signal in features 3 & 8):
- R `mcuve_pls` selects **{3, 8}** (1-based). The RI of the real features is
  `[0.18, -0.10, 11.36, 0.13, 0.08, 0.18, 0.08, -7.43, -0.71, 0.47, 0.01, 0.16]`
  — features 3 and 8 dominate by ~10× over everything else (all |RI| < 0.7).
- n4m's C kernel (different noise model + splitmix RNG) selects **{2, 7}**
  (0-based) = **the same features. Jaccard 1.0.**

## Conclusion (the load-bearing finding)

**n4m's UVE is not wrong, and on signal-dominated data it ALREADY matches R
bit-for-set (Jaccard 1.0) despite a different noise model and RNG** — because
strong features dominate the noise threshold by an order of magnitude. The
dashboard's Jaccard 0.75 (e.g. uve at 200×40) is a **borderline-feature**
effect: one feature sits near the noise threshold and n4m vs R classify it
differently because their noise realizations differ. It is *intrinsic to a
stochastic noise-threshold selector*, not a bug.

To force Jaccard 1.0 **even on borderline features**, n4m's C kernel must be
rewritten to replicate R exactly: Nx unstandardized `runif` noise cols (RNG:
have) + `sample()` subsampling (have) + `pls::plsr` SIMPLS coefficients (have,
~1e-13) + **per-subsample LOO-PRESS `which.min(opt.comp)`** (the fragile part:
a discrete argmin that can flip on 1e-12 PRESS ties) + RI/threshold/fallback.

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
