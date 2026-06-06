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

## GPU — fused cartesian sweep path (cuBLAS + bounded PLS CV routes shipped; full grinder deferred)

The cuBLAS backend (`cuda-on` preset, `cpp/src/core/cuda_dispatch.cpp`) **ships
and is verified** (builds + bit-identical to the CPU reference on GPU). What is
**deferred** is the *fully fused cartesian sweep* execution path, which is where
PLS on a GPU should pay off for very large preprocessing screens. There are
three distinct acceleration axes:

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

2. **Bounded exact PLS CV scheduling (shipped for the moment path).**
   Compatible single-target PLS1 moment sweeps now expose
   `cuda_pls_parallel_folds=True` for bounded stream/cuBLAS fold scheduling and
   `cuda_pls_many_batched=True` for the optional tiled/strided-batched
   many-design route. The CUDA smoke artifacts pin both paths with explicit
   route counters, but these paths still launch over already-formed moment
   jobs; they are not a full many-chain/many-fold/many-candidate fused IKPLS
   engine.

3. **Fused cartesian-throughput execution (deferred).** The reliable large-screen
   win is batching *many* fits into one fused GPU job — preprocessing chains ×
   cross-validation folds × component counts × hyperparameter grid, ensembles,
   or the AOM/POP operator bank. This is the broader pattern of `ikpls`
   (`jax_ikpls_alg_1/2`, `cross_validate`, JAX `jit`/`vmap`). The nirs4all
   Python-side `IKPLS(backend='jax')` wrapper is outside this C++/moment port,
   and nirs4all historically called only the single `fit`, not
   `cross_validate`, so it still does not provide the target 200k-chain AOM
   grinder inside `libn4m`.

- **Why deferred:** axis 1 is a niche win (correct but ≈parity at NIRS sizes);
  axis 2 is useful observability/control for exact PLS CV on one GPU, but the
  target 200k-chain grinder is axis 3: a **new capability** with grouped
  operator kernels, fused batched fit/CV execution and likely new C ABI surface.
  It does not block CPU wheel/CRAN packaging (CUDA is an opt-in build).
- **Trigger:** demand for high-throughput CV / component-sweep / ensemble fitting
  at scale, where batching K folds × C components into one GPU job dominates the
  single-fit latency question.
- **Explicitly NOT chosen:** widening the single-fit cuBLAS path further (more
  ops on GPU) as a perf feature — the measurements show that does not move the
  needle at NIRS sizes.

### Known limitations of the shipped single-fit cuBLAS path (documented, not bugs)

- **Single-thread-only in a cuda-on build.** All GEMM/GEMV/GER share one
  process-wide cuBLAS handle (`cpp/src/core/cuda_dispatch.cpp`), which cuBLAS
  does not allow to be used concurrently from multiple host threads. So the
  `n4m.h` "across contexts thread-safe" guarantee holds for the **CPU** build but
  **not** for a cuda-on build — treat cuda-on as single-threaded until the
  per-stream handle pool (the batched path below) lands. This is now stated in
  the `n4m.h` Threading section.
- **Contiguous-buffer precondition.** `cuda_dispatch` assumes row-major
  contiguous matrices (`lda == cols`); the H2D copy uses `rows*cols`, so a
  strided/transposed view would be silently mis-copied. The model/PLS call sites
  only ever pass contiguous buffers, so this is safe today, but it is a
  precondition, not full Rule-4 stride-awareness, for the GPU path.

### Reserving the batched fit/CV ABI surface (when scheduled)

The batched path (axis 2) is the genuine GPU win but is a **new capability**, not
a refinement — so per this file's legitimacy criteria it needs a *reserved*
public surface. When it is scheduled, reserve one additive symbol (e.g.
`n4m_pls_cross_validate(ctx, cfg, X, Y, fold_assignment, n_folds,
component_grid, n_components_grid, out_result)`) returning
`N4M_ERR_NOT_IMPLEMENTED` until the device-resident batched executor exists, with
a test asserting that status. The exact signature is intentionally **not yet
committed** (it is a design decision that benefits from the consuming
cross-validation API in nirs4all); reserving it is a one-line additive ABI minor
bump + a 3-platform snapshot regen when the work starts.
