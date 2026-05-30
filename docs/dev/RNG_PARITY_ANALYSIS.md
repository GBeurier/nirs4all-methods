# RNG parity for stochastic methods — feasibility analysis

**Question (from maintainer):** make the stochastic methods reproduce their
external references *exactly* (Jaccard 1.0 for selectors). Do it *properly*:
make the RNG an optional model parameter, implement additional RNG algorithms
beyond splitmix64 to cover the tests, and initialise the tests with the right
RNG + seed.

This document analyses impact, feasibility, and effort, and proposes a tiered
plan. It is the basis for a Codex review.

## Status update (2026-05-30)

This analysis is retained for rationale, but it is no longer the current task
list. The additive RNG infrastructure has shipped (`n4m_rng_kind_t` with
splitmix64, PCG64, R-MT, and numpy RandomState-MT), the UVE R-exact pilot is
available behind `rng_kind=MT_R`, and the dashboard now renders documented
selector RNG/noise/model mismatches as `cross_check`/`BD J` instead of red
numeric divergences. See
[`RNG_TIER0_INVENTORY.md`](RNG_TIER0_INVENTORY.md) for the authoritative
post-investigation verdict table.

## 1. What we have today (verified in source)

- **Two RNGs already exist in the core:**
  - `splitmix64` — drives binary-mask / index generators in the selectors
    (hardcoded inside each method).
  - `n4m_rng_pcg64_*` (public ABI, `cpp/include/n4m/n4m.h:367+`) — documented to
    reproduce `numpy.random.default_rng(seed)` bit-for-bit
    (`uint64_fill ≡ default_rng(s).integers`, `standard_normal_fill ≡ ...`).
- **The config has NO rng-kind selector.** `cfg` exposes `solver`, `deflation`,
  scaling, etc., but no `n4m_config_set_rng_kind`. RNG choice is currently
  baked into each method's C++.
- **~14 seed-taking parameters** across the public PLS/selector ABI
  (`uint64_t seed | noise_seed | randomization_seed` in `pls.h`): the
  stochastic surface = the variable selectors (uve, emcuve, cars, random_frog,
  spa-ish, ga, pso, vissa, irf, randomization, stability, scars, …) plus the
  ensemble PLS (bagging, boosting, random_subspace) and some augmenters.

## 2. The real root cause (uve_select, worked example)

The dashboard "Jaccard 0.75" for `uve_select` is NOT a binding bug:

| Tier | What it actually runs | Selected indices |
|------|----------------------|------------------|
| `registry` / `cpp` / `python_tier1` | **call R `plsVarSel::mcuve_pls`** (the canonical path is `legacy=False` → R) | `[2,5,8,13]` |
| `ref_r_plsvarsel` | R `mcuve_pls` (oracle) | `[2,5,8,13]` |
| `r_tier1` / `matlab_tier1` / `python_tier2` | **the n4m C kernel `n4m_uve_select`** | `[2,5,8]` |

So the comparison is **n4m's own C kernel vs R `mcuve_pls`**, and they differ by
one boundary variable (index 13). Per the registry docstring
(`_uve_select_pls4all`), the C kernel deliberately differs from R in **three**
independent ways:

1. **Noise model** — n4m adds a fixed `noise_features` count of signed-uniform
   noise standardised per column; R adds `ncol(X)` uniform-noise columns.
2. **RNG** — n4m splitmix64 vs R Mersenne-Twister (`set.seed(11)`).
3. **Fold / CV semantics** — contiguous-fold coefficient sampling vs R's.

**Consequence:** matching R bit-for-bit needs ALL THREE aligned, not just the
RNG. With a different noise model or draw order, even an identical RNG stream
diverges. "Make the RNG a parameter" is *necessary but far from sufficient*.

This generalises: each stochastic method's reference is one of
- **R** (Mersenne-Twister, R's `set.seed` scrambler, R's `rnorm` inversion) —
  most selectors (`plsVarSel`, `mdatools`),
- **numpy/sklearn** (PCG64 — which n4m ALREADY reproduces) — the
  python-composite methods,
- **auswahl** (numpy under the hood) — random_frog / vissa / irf / vip_spa,
- **deterministic** (no RNG) — spa, vip, coefficient, interval.

## 3. Why exact repro is hard (the honest part)

Bit-exact match to R requires, *per method*:

1. **R-compatible Mersenne-Twister.** R's RNG is MT19937 but with R's own
   `set.seed` scrambling (Init_R_seed) and R's `unif_rand`/`norm_rand`
   (inversion via `qnorm`). This is NOT the same byte stream as numpy's
   `RandomState` (also MT but seeded differently) nor generic MT. It is
   well-documented (R `src/main/RNG.c`) but fiddly — ~a few hundred LOC + an
   exact-stream test against R.
2. **Exact noise model** — replicate how R builds its noise columns (count,
   distribution, placement).
3. **Exact draw order** — the sequence of RNG consumption must match R's loop
   structure draw-for-draw; one extra/missing draw desynchronises everything.
4. **Exact CV / fold semantics** and the **exact thresholding rule**.

(2)–(4) amount to **porting each R function's algorithm into C++**. That is the
bulk of the effort and the risky part — some references (auswahl, mdatools)
have undocumented internals.

There is also a **design tension**: n4m's C kernels were deliberately built with
their OWN deterministic, documented noise model (so they are reproducible and
not hostage to R's RNG). Making them mimic R means *abandoning n4m's own design
to copy R bit-for-bit*. That is a legitimate choice for "dashboard shows green",
but it is "n4m becomes R", not "n4m is independently correct".

## 3b. Concrete classification (verified by grep over registry source)

- **13** seed-taking methods in the public ABI (`uint64_t seed|noise_seed|
  randomization_seed` in `pls.h`).
- **15** registry helper functions route a selector to R (`_*_indices_via_r`):
  bve, cars(enpls), emcuve, ga, ipls/interval, ipw, randomization, rep, shaving,
  spa, sr, stability/mcuve, st, uve, wvc_threshold.
- **3** route to auswahl (numpy): random_frog, vip_spa, irf.
- **12** methods have their **DEFAULT** path (`legacy=False`) routing to R —
  i.e. for these 12 the "canonical n4m" value shown on the dashboard *is R's
  output*, and the n4m C kernel is the `legacy=True` opt-in that the binding
  tiers exercise. This is the set where C-kernel-vs-R parity is measured.
- The **numpy/sklearn composites** (`bagging/boosting/random_subspace/n_pls/
  pls_qda/pls_glm/pls_cox/pls_logistic/group_sparse`) BYPASS the C kernel
  entirely — NOT an RNG problem; correctly `not_available` for C-kernel bindings.

**Net (corrected):** the RNG-exact-repro effort is **~12–15 R-referenced
selectors**, not the 6 I first estimated — i.e. Tier C is *larger* than my first
pass. The "cheap PCG64 win" (Tier A) is genuinely small, because the
numpy-referenced stochastic surface is mostly composites the C kernel does not
run. Tier B (RNG abstraction) is the enabler for the 12–15 R cases + 3 auswahl
cases.

## 3c. Codex review corrections (verified vs source — authoritative over §3b)

A read-only Codex review (high reasoning) checked the analysis against the code
and corrected three factual points + added two risks:

1. **R-referenced set ≥ 12 and `iriv_select` is NOT R-backed.** R-default
   selectors include stability, uve, spa, cars, ga, shaving, bve, emcuve,
   randomization, rep, ipw, st, wvc_threshold (registry.py:3588/3845/3957/4867/
   5001/5119/5244/5381). `iriv_select` routes through a **NumPy port using
   `default_rng`** (registry.py:4485, 7659), not R.
2. **auswahl is NOT numpy-PCG64.** It passes `random_state=int(seed)`
   (registry.py:1562/4014/7543); auswahl 0.9.0 uses sklearn
   `check_random_state` → legacy **`numpy.random.RandomState` (MT19937)**, not
   `default_rng`/PCG64. The auswahl cases need a THIRD engine (numpy
   RandomState-MT); the "cheap PCG64 Tier A" is smaller/different than hoped.
3. **`n4m.h:344` overclaims PCG64 adoption** — UVE/EMCUVE public APIs take only
   a seed (pls.h:1404/1605) and UVE uses file-local splitmix64
   (uve_selection.cpp:144/215). The header comment needs correcting.
4. **RISK (Tier B): symbol-additive ≠ behaviour-preserving.** splitmix is
   duplicated file-local across ~10 selectors with method-specific seed
   perturbations (EMCUVE `noise_seed + ensemble*golden`
   emcuve_selection.cpp:87; randomization per-permutation seed :281;
   randomized-SVD context-seed offset model.cpp:766). **Freeze current splitmix
   streams with golden tests BEFORE the vtable refactor**; default `rng_kind`
   must = exact-current-splitmix.
5. **RISK (Tier C): R-MT is bounded to `set.seed`+`unif_rand`(+`rnorm`).** Methods
   that use R's `sample()`, CV-segment generation or package internals also need
   `sample.kind` semantics + package draw order — NOT "a few hundred LOC". The
   R-MT engine just landed covers the bounded case only.

Also: some references are "closest installable analog", not the method proper
(CARS vs `enpls.fs`, registry.py:6061) → keep those as documented
`cross_check`/Jaccard, never a forced bit-match.

**Revised verdict (Codex-aligned), supersedes §5:**
- **Tier 0 first** — fix the method inventory; per method decide canonical /
  compat-mode / documented-cross_check (§3b was unreliable).
- **Tier B only after golden stream tests** freeze current splitmix; `rng_kind`
  additive, default = current splitmix.
- **Implement only the RNGs actually needed:** PCG64 exists; add numpy
  `RandomState`-MT for auswahl; R-MT (DONE) for true R-compat methods.
- **Tier C only for high-value methods** where the reference IS the target
  algorithm (UVE/MCUVE); cross_check/Jaccard for the analog-only ones.

## 4. Proposed tiered plan

### Tier B (infrastructure — do this regardless; it is the right abstraction)
Make RNG a first-class, pluggable, configurable concept:
- `typedef enum n4m_rng_kind_t { N4M_RNG_SPLITMIX64=0, N4M_RNG_PCG64=1,
  N4M_RNG_MT19937_R=2, ... }` in a public header.
- A core RNG interface (vtable: `next_u64`, `next_double`, `next_normal`,
  `reseed`) so methods draw from an abstract RNG, not a hardcoded one.
- `n4m_config_set_rng_kind(cfg, kind)` + reuse the existing seed.
- Wire each of the ~14 stochastic methods to draw from the configured RNG.
- **ABI impact:** new public symbols → regenerate `expected_symbols_*` on all 3
  platforms + bump `N4M_ABI_VERSION_*` + `docs/abi/changes_log.md`. The enum and
  setter are additive (backward-compatible), default = current behaviour
  (splitmix64) so nothing silently changes.
- **Effort:** ~1–2 weeks C++ + ABI gates + per-binding plumbing (R/MATLAB/Py/JS
  must expose the new setter). Self-contained, testable, valuable on its own.

### Tier A (cheap wins — do first)
For methods whose reference is **numpy/sklearn**, n4m ALREADY has PCG64. If a
method currently uses splitmix64 but its reference uses numpy, switching it to
the existing PCG64 RNG (via Tier B's config, or directly) can yield exact repro
*cheaply* — no new RNG needed. Audit which of the ~14 fall in this bucket.

### Tier C (expensive — per-method R alignment; decide method-by-method)
Implement `N4M_RNG_MT19937_R` (R-compatible MT + `rnorm` inversion), then for
each R-referenced selector replicate R's noise model + draw order + folds +
threshold. **Effort: days per method × ~10–12 methods = weeks→months**, with no
guarantee for the undocumented ones (auswahl/mdatools internals).

## 5. Recommendation (to be reviewed by Codex)

1. **Tier B is worth doing on its own merits** — "RNG as a configurable,
   pluggable model parameter" is the correct architecture and unblocks
   everything else. Ship it with `default = splitmix64` (no behaviour change)
   and PCG64 already wired.
2. **Tier A** likely converts the numpy/sklearn-referenced stochastic methods to
   exact repro quickly once Tier B exists.
3. **Tier C is a per-method decision, not a blanket commitment.** For each
   R-referenced selector, weigh "port R's algorithm into the C kernel" vs
   "accept it as a documented `cross_check` (n4m has its own valid algorithm)".
   Bit-exact-to-R is achievable but is real reverse-engineering and partly
   makes n4m a re-implementation of R.
4. **Independent of all the above**, the dashboard must stop showing a Jaccard
   overlap as if it were an RMSE divergence (separate task #3, in progress) — a
   Jaccard 0.75 is "75% feature overlap", not "0.75 numeric drift".

## 6. Open questions for the maintainer

- For the R-referenced selectors, is the goal "n4m reproduces R exactly"
  (Tier C, n4m mimics R) or "n4m has a documented, correct selection that we
  display honestly as a cross-check"? This is the cost driver.
- Is the new RNG abstraction (Tier B) acceptable as an additive ABI change with
  an ABI-version bump and snapshot regen on all platforms?
