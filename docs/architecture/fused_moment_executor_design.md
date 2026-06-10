# Fused moment-screen executor — design / scoping (2026-06-08, Codex-reviewed)

Phase 5a of the AOM/moment accuracy-then-throughput work. **Scoping doc for review
BEFORE any kernel code.** Revised after an adversarial Codex review (raw review:
`fused_moment_executor_design.codex_review_2026-06-08.md`) that corrected several
overclaims in the first draft — see "Codex review incorporated" at the bottom.

## TL;DR — direction confirmed, diagnosis must be measured first

The handoff's deferred "big task" was a *fused many-chain/fold/candidate IKPLS
**scoring** executor*. That targets the wrong layer: the PLS moment scorer is already
on CUDA and is not where large-p wall-clock goes. **Codex concurs with redirecting to
the upstream moment / data-movement layer, but flags that the first draft overclaimed
*which* sublayer dominates and mislabeled the cost as "host CPU GEMM."**

Corrected thesis: on large-p datasets the staged campaign spends its time on the
**host side of the per-candidate moment pipeline** — but the GEMMs themselves already
route to cuBLAS, so the cost is most plausibly **per-candidate data movement
(H2D/D2H), device allocations, and launch/sync overhead, plus genuinely host-side
Ridge eigensolves** — *not* CPU matmul. **This must be confirmed by a phase-timing
profile (5b-0) before any kernel is written.** 5b-0 is a hard gate, not a warm-up.

## 5b-0 MEASURED (2026-06-08, nsys + before/after)

Profiled one BERRY brix screen (n=1434, p=2101, compact, 71.3 s) with `nsys
--sample=cpu` + a correct-but-neutral first change, and the measurement **overturned
the draft hypothesis**:

- **#1 host hotspot: `n4m_householder_qr` = 37.1% of CPU samples** (`libn4m.so`) — the
  **Ridge GCV solve** (`ridge.cpp` / `sweep.cpp` `solve_square_qr`, single-threaded
  O(p³/n³) QR). Note: this runs even though `n_ridge_moment_*` counters are 0 — Ridge
  candidates are scored through a QR path the moment counters don't track.
- **#2: synchronous `cudaMemcpy` = 4.59 s** (98% of CUDA API time, 888 blocking copies,
  one 174 ms) — host-blocking H2D/D2H, ~6% of wall-clock, ~20% of samples land in the
  libcuda copy machinery.
- CPU OpenBLAS `blas_thread_server` ~6%; `cuda_dispatch::gemm` only **2.6%**;
  `cudaMalloc`+`cudaFree` = **14 ms total** (negligible).
- **First change (neutral, kept):** gave `cuda_dispatch::gemm` a thread-local reusable
  device scratch arena (kills per-call cudaMalloc/Free). Result: RMSEP **bit-identical**
  on BERRY/COLZA/LUCAS, speedup **1.00×** — because alloc/GEMM was never the cost. The
  change is correct and foundational for residency, just not the bottleneck.

**Revised priority (data-driven):** the moment-transform GEMM (draft's #1 target) is
minor. The real targets, in order: **(1) move the Ridge GCV solve off the CPU
householder-QR onto the GPU** (cuSOLVER SPD Cholesky/eigh — the ridge normal system is
SPD); **(2) cut the 4.59 s of synchronous cudaMemcpy** (device residency / pinned). The
moment-transform batching matters only after these.

## What is established vs hypothesized

**Established (direct evidence):**
- BERRY brix (n=1434, p=2101) took **1212 s with GPU0 at ~0% util**, one CPU core
  pinned. Across 51 datasets `corr(log fit_time, log n·p²)=0.86`; median fit
  **40.6 s (p≥1500) vs 3.9 s (p<800)** — 10×. → The slow phase is **not active CUDA
  kernels**, and cost grows steeply with size.
- The PLS moment scorer ran fully on device for BERRY
  (`n_screen_pls_moment_cuda_device_cv_fits=120` host 0; refit 20 host 0). → The
  scorer is not the bottleneck.

**Hypothesized (NOT yet proven — 5b-0 must apportion):**
- *Which* host phase dominates: base-moment build, per-candidate operator→moment
  transform, host Ridge eigensolves, or the data-movement/alloc/launch overhead
  around the cuBLAS calls. `corr=0.86` with `n·p²` is **confounded** — large p
  correlates with large n, chain count, refit load, and stage mix, and at BERRY scale
  `p³` and `n·p²` are same-order, so the correlation cannot separate transform from
  build from eigensolve.
- The BERRY `n_ridge_moment_*=0` reading does **not** prove "wide Ridge materialized";
  it could equally mean the Ridge moment route did not run. Needs the
  dual/materialized counters + active-head evidence.

## Corrected code facts (verified by Codex against source)

These replace the first draft's claims:

1. **Base moment build is NOT host-CPU GEMM in CUDA builds.** `compute_moments`
   (`moments.cpp:190-201,264-270`) builds `xtx/xty` via `linalg::gemm`, which in CUDA
   builds dispatches to `cuda_dispatch::gemm` (`common/linalg.hpp:173-178`) — cuBLAS
   with per-call device alloc + H2D + D2H (`cuda_dispatch.cpp:1439-1472`). Buffers are
   **host-owned**; the matmul is on-device but wrapped in per-call copies. So the
   suspect is **copy/alloc/launch overhead**, not CPU matmul.
2. **The dense O(p³) operator→moment transform is only one route.**
   `transform_moments_by_operator` does `opᵀ·xty` (O(p²q)) + `xtx·op` + `opᵀ·temp`
   (2×O(p³)) at `aom_sweep.cpp:674-687`, but the code tries **structured/banded**
   transforms first (`aom_sweep.cpp:1323-1344,1425-1468`) and disables dense for
   `p>min_train` except a tiny cap (`aom_sweep.cpp:2222-2225`). "Every candidate = two
   O(p³) GEMMs" is false in general.
3. **Wide-Ridge does NOT host-materialize via `should_materialize_cpu_wide_ridge` in
   CUDA builds** — it returns false immediately when `uses_cuda_linalg_build()`
   (`aom_sweep.cpp:555-560`). Materialized fallback can still occur via unsupported
   moment routes, but not from this predicate.
4. **Ridge GCV is host-only (confirmed) but amortizes lambdas.** `fit_ridge_from_moments`
   uses host vectors + host `solve_square_qr` (`sweep.cpp:840-868`, `ridge.cpp:104-120`);
   the eigen path is host (`sweep.cpp:956-958`, `svd.c:263-283`). But multiple λ share
   one host eigen prep per fold (`sweep.cpp:2094-2110`, `fit_ridge_from_eigen_path`
   `sweep.cpp:2157-2172`) — not QR/eigh per λ. **cuSOLVER is not linked** (only
   `CUDA::cudart CUDA::cublas`, `n4m_targets.cmake:144-148`).
5. **The PLS scorer is device-resident only *inside* the scorer.** It takes host
   `C/s` pointers and copies them in per tile (`cuda_dispatch.cpp:834-843`), runs
   cuBLAS strided-batched (`854-1072`), copies W/P/Q back (`1104-1113`). Feeding it
   device-resident moments is **not a drop-in** — the scorer would need an internal
   device-pointer entry variant.

## Current batching layout

The scorer `pls1_moment_components_many_batched_tiled` (`cuda_dispatch.cpp:784`)
batches **folds** into device tiles (one cuBLAS strided-batched call per component over
all folds of one candidate), `dcur_yy`/`dQ` device-resident, W/P/Q read back per tile.
It is fp64, called **once per (chain×candidate)** from `sweep.cpp:1636`, fed
host-built moments. **Host-looped today:** chain, candidate, and the entire
moment-production pipeline that precedes the scorer.

## Revised proposed design — kill per-candidate data movement, keep moments on device

Keep `libn4m` single-runtime. The intent is **not** "move CPU GEMM to GPU" (it's
already cuBLAS) but **"stop round-tripping moments through host buffers per
candidate"** and **batch the per-candidate launches** so the device stays busy.
Invariants preserved: fold-clean stateful heads, P-dependent detrend projector Q,
`MomentStats` raw+centered fields, fp64-for-scores, train-CV-only selection.

Candidate components (each gated on 5b-0 evidence):
1. **Device-resident base moments per fold-config** — compute `XtX/XtY/yy` once and
   keep on device, instead of host-owned buffers re-copied per candidate.
2. **Batched operator→moment transform on device, route-aware** — for the routes that
   actually run (dense / banded / structured), apply the operator transform to
   device-resident base moments in **batched** cuBLAS calls over candidates. Do **not**
   force dense p×p batches for large p where the code already uses banded/structured.
3. **Device-pointer scorer entry** — add an internal variant of the PLS scorer that
   accepts device-resident `C/s/yy`, eliminating the per-tile H2D of `hC/hs`.
4. **(Conditional) device Ridge GCV** — only if 5b-0 shows host Ridge is a material
   fraction; would add a cuSOLVER dependency (currently unlinked).
5. **Batch chain×candidate into the tile dimension** — so the executor is one call per
   stage-chunk, not one per candidate, eliminating the host launch/copy loop.

## Internal device-state contract (new — per Codex)

Any device-resident state must be defined before code:
- **Memory budget.** One p×p fp64 base moment at p=4200 (LUCAS) = **141 MB**; raw +
  centered + transformed + temp + operator matrices multiply that. Must fit the
  existing PLS tile budget (`choose_pls_batch_tile_jobs`, `cuda_dispatch.cpp:433-469`)
  — chain×candidate×fold batching **collides** with the current per-job p×p budget and
  must share one accounting.
- **No persistent cross-call cache.** Device state stays **strictly internal to one
  public call** (so "no ABI change" holds). Persisting base moments in `Context` or
  across calls changes lifetime/thread-safety/failure semantics even with symbols at
  ABI 1.22.0 (`n4m_version.h:20-22`) — out of scope unless explicitly designed.
- **fp64 + equivalence.** Keep fp64; assert batched-device transform == host transform
  within tol; a degenerate-component batch must preserve the existing per-job
  failure/recovery semantics (don't let one bad candidate poison the batch).
- **Fold-local state.** MSC/EMSC re-fit per fold; the detrend projector/inverse is
  P-dependent (`aom_sweep.cpp:904-928`) — must not be hoisted across folds/P.
- **Device selection.** CUDA state currently hardcodes device 0
  (`cuda_dispatch.cpp:50-56`); the executor must honor `CUDA_VISIBLE_DEVICES` / the
  chosen device, not assume 0.
- **cuSOLVER gating.** Only linked if 5b-3 is greenlit by profiling.

## Increments (5b-0 is a HARD GATE)

- **5b-0 — Phase-timing profile (mandatory, gates everything).** Native phase timers
  (or NVTX ranges) around `compute_moments*`, `transform_chain`/operator-transform,
  Ridge eigen/direct fits, PLS scoring, and the Python chunk/checkpoint overhead, run
  on **BERRY / COLZA / LUCAS** with checkpoints off, scale_x grid fixed, plus Nsight
  Systems CUDA kernel-vs-copy time and the dual/materialized counters. **Deliverable: a
  reproducible artifact apportioning the large-p host time.** No kernel work until this
  names the dominant sublayer.
- **5b-1..5b-5** (device-resident base moments → batched route-aware transform →
  device-pointer scorer entry → conditional cuSOLVER Ridge → chain×candidate batching):
  **scope and order to be set by 5b-0's measured split**, biggest measured win first.
  Each behind the full green gate (`n4m_tests` + `n4m_internal_tests` CUDA &
  dev-release, targeted CUDA pytest, catalog `split/--check-references/--strict-abi` +
  `reconcile_abi --check`, `git diff --check`), with host-equivalence asserted and ABI
  held at 1.22.0.

## Open questions — resolved (Codex + author agree)

1. **Scope:** target the upstream moment / data-movement layer, not the PLS scorer —
   **but narrow only after 5b-0**; do not assume dense-transform+Ridge dominates. ✓
2. **cuSOLVER:** **not yet** — keep Ridge host-side until 5b-0 shows it's material. ✓
3. **Precision:** **fp64 first**; fp32-screen/fp64-verify is a later experiment needing
   a recall/top-k-miss audit (the accuracy track already showed selection is variance-
   sensitive). ✓
4. **5b-0 first:** **mandatory hard gate** with native timers + route/materialized
   counters + CUDA copy/kernel time. ✓

## Codex review incorporated (2026-06-08)

Changed vs first draft: (a) the cost is reframed from "host CPU GEMM" to "per-candidate
data-movement/alloc/launch overhead + host eigensolves" — the GEMMs already route to
cuBLAS; (b) the `should_materialize_cpu_wide_ridge` and "every candidate = two O(p³)
GEMMs" and "Ridge QR/eigh per λ" claims are corrected/refuted with line evidence;
(c) the empirical section now separates *established* from *hypothesized* and flags the
`n·p²` confound and the counter over-reading; (d) 5b-0 is promoted to a hard gate that
must apportion the host time before any kernel; (e) a device-state contract (memory at
p=4200, tiling collision, fold-local state, device selection, no cross-call cache) is
added; (f) feeding the scorer device moments is acknowledged to need a new device-
pointer entry, not a drop-in. Raw review preserved alongside.

## Why this is the right investment

The accuracy track established the *accuracy* ceiling is model expressiveness, not
selection (`RECALL_FIX_RESULTS.md`). The remaining engine work is *throughput*, and the
honest lever is the host-bound large-p moment pipeline (10× slower than small-p) — but
**what exactly to fuse is a measurement question, answered by 5b-0, not assumed.**
