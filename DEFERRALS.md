# Deferrals — deliberate, documented non-implementations

This file is the canonical record of features that are **intentionally not
implemented** in the current ABI, with a stable public surface reserved for
them and a concrete trigger for when they land. A deferral here is **not
release debt** — it is a decided scope boundary. Code comments that say
"see DEFERRALS.md" point here.

A deferral is legitimate only when all of these hold:

- the public surface is **reserved** (ABI symbols exist; `create`/`destroy`
  work; the operative call returns a clear, tested error);
- there is a **documented reason** the work is gated;
- there is a **concrete trigger** that unblocks it;
- tests **assert the deferred behaviour** so it cannot regress silently.

---

## Dashboard — interactive SPA (D-SPA)

The interactive Svelte/Vite dashboard (Phase D, D1–D15) is **deferred, not
cancelled**. D-min shipped the load-bearing half: a schema-validated
`dashboard.json` **contract** + automated per-method score cards, which the SPA
will consume **unchanged**.

- **Reserved surface:** the `make dashboard-data` / `dashboard-serve` /
  `dashboard-build` targets are stubs that print "not yet bootstrapped (Phase D)"
  and exit 0; the `dashboard/` app does not exist.
- **Reason:** the static Sphinx/landing dashboard + the `method_scores` cards
  already make the parity/timing signal visible and maintainable. The SPA is UX
  polish, made cheap by the fixed contract.
- **Trigger:** when the static dashboard becomes the limiting factor for
  consuming the signal (rich filtering, drift history, multi-host).
- **Full scope:** [`docs/dashboard_contract.md`](docs/dashboard_contract.md) → "D-SPA (deferred): scope".

## Binding-scaling infrastructure (Phase F-prep)

The 10+-language binding-scaling infra is **reduced and deferred** by the
4-language target + the hand-written-idiomatic decision in
[`bindings/SPEC.md`](bindings/SPEC.md).

- **Cancelled (over-engineering at 4-language scale):** the framework-profile
  schema (F-prep-3) and the per-`(method × profile)` template/codegen engine
  (F-prep-4).
- **Deferred, low value:** the skeleton generator (`make new-binding LANG=`,
  a stub) and the unified `bindings.yml` / `release-bindings.yml` matrices
  (a CI/release consolidation, not new capability).
- **Trigger:** a request for a 5th+ target language.
- **Full scope:** [`bindings/SPEC.md`](bindings/SPEC.md) → "F-prep scope".

## GPU — batched execution path (single-fit cuBLAS shipped; batch deferred)

The cuBLAS backend (`cuda-on` preset, `cpp/src/core/cuda_dispatch.cpp`) **ships
and is verified** (builds + bit-identical to the CPU reference on GPU). What is
**deferred** is the *batched* GPU execution path, which is where PLS on a GPU
actually pays off for NIRS-scale data. There are two distinct acceleration axes,
and only the first exists today:

1. **Single-fit GEMM offload (shipped).** `linalg::gemv/gemm/ger` route to cuBLAS
   when `N4M_USE_CUDA` is compiled. This is a *latency* play whose benefit is
   gated by problem size: at routine NIRS sizes the matrices are too small for
   the host↔device copy + kernel-launch overhead to amortise. Measured on
   RTX 4090/5090: `gpr_pls` 4000×800 = **1.01×**, `mb_pls` 3000×1500 = **0.99×**
   (parity). The per-method diagnostic
   ([`docs/benchmarks/cuda_diagnostic.md`](docs/benchmarks/cuda_diagnostic.md))
   buckets **168 / 177** methods as "CUDA would be slower" at dashboard sizes —
   only large-`n×p` Gram/kernel/multi-block variants (IKPLS, GPR-PLS, MB-PLS,
   PCR/SVD, AOM-selection) are candidates, and only well above NIRS-typical sizes.

2. **Batched-throughput execution (deferred).** The reliable PLS-on-GPU win is
   batching *many* fits into one GPU job — cross-validation folds × component
   counts × hyperparameter grid, ensembles, or the AOM/POP operator bank. This is
   the pattern of `ikpls` (`jax_ikpls_alg_1/2`, `cross_validate`, JAX `jit`/`vmap`),
   which nirs4all already wires as `IKPLS(backend='jax')` in
   `nirs4all/operators/models/sklearn/ikpls.py` — though nirs4all currently calls
   only the single `fit`, not `cross_validate`, so even it does not yet exploit
   ikpls's main GPU strength.

- **Why deferred:** axis 1 is a niche win (correct but ≈parity at NIRS sizes);
  axis 2 is genuinely valuable but is a **new capability** (a batched fit/CV
  execution path + its C ABI surface), not a refinement of the existing backend.
  It does not block CPU wheel/CRAN packaging (CUDA is an opt-in build).
- **Trigger:** demand for high-throughput CV / component-sweep / ensemble fitting
  at scale, where batching K folds × C components into one GPU job dominates the
  single-fit latency question.
- **Explicitly NOT chosen:** widening the single-fit cuBLAS path further (more
  ops on GPU) as a perf feature — the measurements show that does not move the
  needle at NIRS sizes.
