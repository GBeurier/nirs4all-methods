# Codex adversarial review — fused_moment_executor_design.md (2026-06-08)

Raw output of `codex exec` (codex-cli 0.137.0, read-only) reviewing the first draft of
`fused_moment_executor_design.md`. Preserved verbatim for the record; the design doc
was revised to incorporate these findings. See the design doc's "Codex review
incorporated" section for the resolution of each point.

---

**VERDICT**
The reframe away from a fused PLS scorer is directionally plausible, but the document overclaims the diagnosis. The code supports "host-owned moment production and copy/orchestration are suspicious"; it does not yet prove "dense moment transform + Ridge GCV are the dominant BERRY bottleneck."

**CLAIM VERIFICATION**
- **Base moment build is host-bound: overstated.** `compute_moments` copies `X/Y` into host vectors at `cpp/src/core/moments.cpp:264-270`, then builds `xtx/xty/yty` via `linalg::gemm` at `moments.cpp:190-201`. But in CUDA builds, `linalg::gemm` dispatches to `cuda_dispatch::gemm` (`cpp/src/core/common/linalg.hpp:173-178`), which allocates device buffers, copies host inputs, runs cuBLAS, and copies back (`cpp/src/core/cuda_dispatch.cpp:1439-1472`). So the buffers are host-owned, but the GEMM math is not necessarily CPU-side.

- **`transform_moments_by_operator` has two O(p^3) dense GEMMs: confirmed, but scoped too broadly.** It does `op^T * xty` at `cpp/src/core/aom_sweep.cpp:674-677` (`O(p^2 q)`), then `xtx * op` and `op^T * temp` at `aom_sweep.cpp:679-687` (`2 * O(p^3)`). However this is only the dense route. The code first tries structured/banded transforms (`aom_sweep.cpp:1323-1344`, `1425-1468`), and mixed scoring disables dense for `p > min_train` except a tiny cap (`aom_sweep.cpp:2222-2225`). The doc's "per-candidate transform = two O(p^3) GEMMs" is not universally true.

- **Those GEMMs are "host" work: inaccurate in CUDA builds.** The function uses host `std::vector` storage, but its GEMMs go through `linalg::gemm`; CUDA builds route that to cuBLAS (`common/linalg.hpp:173-178`) with per-call H2D/D2H copies (`cuda_dispatch.cpp:1439-1472`). The real current problem may be host-owned buffers, repeated allocations/copies, and launch synchronization, not CPU GEMM.

- **`fit_ridge_from_moments` is host-only: confirmed.** It builds host vectors (`cpp/src/core/sweep.cpp:840-844`), scales/regularizes them on the host (`sweep.cpp:846-865`), then calls `solve_square_qr` (`sweep.cpp:867-868`). That QR is the shipped host Householder implementation (`cpp/src/core/ridge.cpp:104-120`). The eigen path is also host (`sweep.cpp:956-958`, `cpp/src/core/common/svd.c:263-283`). Current CUDA linkage is only `CUDA::cudart CUDA::cublas`, not cuSOLVER (`cpp/src/n4m_targets.cmake:144-148`).

- **Ridge GCV is host QR/eigh per candidate: overstated.** Multiple lambdas can share one host eigen preparation per fold (`cpp/src/core/sweep.cpp:2094-2110`), and lambda fits use `fit_ridge_from_eigen_path` when available (`sweep.cpp:2157-2172`). So Ridge is host-only, but not necessarily repeated QR/eigh per lambda.

- **`should_materialize_cpu_wide_ridge` forces host materialization for p>>n: refuted as written.** It returns false immediately in CUDA linalg builds: `if (uses_cuda_linalg_build()) return false;` at `cpp/src/core/aom_sweep.cpp:555-560`. Only non-CUDA builds return `p > min_train_rows(...)` (`aom_sweep.cpp:561-563`). Wide materialized fallback can still happen when the moment route is unsupported, but that is not this predicate alone.

- **PLS moment scorer is CUDA and fed host-built moments: mostly confirmed.** `pls1_design_moments` consumes host `MomentStats` and builds host `C/s/yy` (`cpp/src/core/sweep.cpp:1220-1240`, `1617-1629`). The many-batched CUDA path copies `C/s` into host tile buffers (`cpp/src/core/cuda_dispatch.cpp:834-840`), H2D copies them (`cuda_dispatch.cpp:841-843`), runs cuBLAS strided-batched work (`cuda_dispatch.cpp:854-1072`), then copies W/P/Q back (`cuda_dispatch.cpp:1104-1113`). It is device-resident only inside the scorer, not across the moment pipeline.

- **BERRY counter claims: not independently verifiable from source.** The code has the counters and increments CUDA-device PLS fits when the device path is used (`cpp/src/core/sweep.cpp:1815-1827`, `2486-2496`). But "all `n_ridge_moment_*` counters were 0 therefore wide Ridge materialized" needs the corresponding dual/materialized counters and active-head evidence; zero moment counters alone could also mean Ridge was not run.

**EMPIRICAL CRITIQUE**
0% GPU plus one pinned CPU core strongly suggests the slow phase is not active CUDA kernels, but it does not identify the phase. It could be staged Python orchestration, checkpoint JSON writes, multiple stages, `scale_x_values` doubling, split head scoring, sequential chunking, materialized `transform_chain`, host eigensolves, or repeated H2D/D2H copies around CUDA BLAS.

`corr(log fit_time, log(n*p^2)) = 0.86` is suggestive but confounded. Large `p` may correlate with `n`, number of chains retained, route fallback, refit workload, and stage mix. It also does not distinguish `n*p^2` base moments from `p^3` dense transforms or host Ridge eigensolves; for BERRY-scale `p^3` and `n*p^2` are same-order enough that correlation alone cannot apportion blame.

The single clean measurement: one BERRY/COLZA chunk with checkpoints off, native phase timers or NVTX ranges around `compute_moments*`, `transform_chain`, operator-moment transform, Ridge eigen/direct fits, PLS component scoring, and Python chunk/checkpoint overhead, plus CUDA kernel/copy time from Nsight Systems. Without that split, 5b-1 is speculative.

**DESIGN & ORDERING CRITIQUE**
5b-0 must be first and should be a hard gate. 5b-1 is not justified as "biggest first" until the timers prove dense operator-moment transform dominates. For wide `p`, current code may materialize raw transformed `X` or use structured/banded paths rather than the cited dense transform.

5b-2 only pays off after moments are actually produced on device. Today the scorer API takes host pointers and copies `C/s` internally; feeding it device moments is not a small drop-in unless the scorer grows an internal device-pointer variant.

5b-3 cuSOLVER Ridge should not precede evidence that Ridge dominates. Existing host eigen-path amortizes lambdas, and cuSOLVER adds a real dependency and packaging surface.

Correctness traps: fold-local fitted preprocessing state must not leak across train/test folds; detrend uses a p-dependent projection basis and inverse (`cpp/src/core/aom_sweep.cpp:904-928`); `MomentStats` consumers require raw and centered fields (`cpp/src/core/sweep.cpp:1237-1255`); fp64 equivalence matters for ranking; cuSOLVER eigh/QR tolerances may perturb candidate ordering; and batching chain×candidate×fold can silently change error/recovery behavior for degenerate PLS components.

"No ABI change" is true only if all device state is strictly internal to one public call. Persisting device-resident base moments across public calls or in `Context` changes resource lifetime, thread-safety, and failure semantics even if exported symbols stay at ABI 1.22.0 (`cpp/include/n4m/n4m_version.h:20-22`).

**ANSWERS TO Q1-Q4**
- **Q1:** Target the upstream moment/data-movement layer, not a PLS scorer megakernel. But narrow it after 5b-0; do not assume dense transform + Ridge is the dominant layer.
- **Q2:** Do not add cuSOLVER yet. Keep Ridge host-side until profiling shows Ridge is a material fraction and the exact Ridge route is active.
- **Q3:** Keep fp64 for the first implementation. fp32 screen plus fp64 verify is a later experiment requiring recall/top-k miss audits.
- **Q4:** Yes, 5b-0 first is mandatory. It should include native phase timers, route counters, dual/materialized counters, and CUDA copy/kernel time.

**MISSING / RISKS**
Device memory is under-specified. One p=4200 double p×p matrix is ~141 MB; raw + centered + transformed + temp + operator matrices quickly exceed practical tile budgets. Existing PLS tiling already accounts for p×p per job (`cpp/src/core/cuda_dispatch.cpp:433-469`); chain×candidate×fold batching will collide with that.

The design also misses multi-GPU/device selection: current CUDA state hardcodes device 0 (`cpp/src/core/cuda_dispatch.cpp:50-56`). It misses the risk that per-candidate device transforms without batching just replace a host loop with a launch/copy loop. It also does not separate screen time from exact refit/impact/rank-diagnostic time in the staged Python workflow.

**TOP 3 CHANGES**
1. Replace the empirical section with a reproducible phase-timing artifact and route-counter table for BERRY/COLZA/LUCAS.
2. Redesign around actual routes: dense, banded, structured, and materialized fallback; avoid dense p×p operator batches for large p unless measured necessary.
3. Define the internal device-state contract: memory budget, tiling, fp64 tolerances, fold-local state handling, cuSOLVER gating, and no persistent public-call cache unless explicitly designed.
