I've verified every item in the checklist. Let me confirm the one remaining detail — that the GEMM argument mapping in the new functions is character-for-character what the existing wrapper would emit — by comparing the derived calls (already done in analysis). I have everything I need.

## VERDICT: **ship**

B3 is correct, in-scope, ABI-neutral, fold-safe, and bit-equivalent to the existing CUDA GEMM path. The non-CUDA build compiles it out. Findings below; all the substantive ones are `[OK]`, with two non-blocking `[SCOPE]`/`[STYLE]` notes.

---

### 1. `build_moments_device` — GEMM mapping & bit-equivalence — `[OK]`
`cpp/src/core/cuda_dispatch.cpp:1222-1338`. I derived each cuBLAS call and compared it to what the existing wrapper (`cuda_dispatch.cpp:1963-2021`, which emits `cublasDgemm_v2(op_b, op_a, N, M, K, …, dB, ldb, dA, lda, …, dC, ldc)`) would issue for the host-fallback `linalg::gemm` calls:

| Product | New code (1283-1318) | What the wrapper emits for the fallback call | Match |
|---|---|---|---|
| XᵀX | `OP_N,OP_T, p,p,n, X,p, X,p, C,p` | `gemm(T,N,p,p,n,X,p,X,p)` → `OP_N,OP_T,p,p,n,X,p,X,p,C,p` | ✓ identical |
| XᵀY | `OP_N,OP_T, q,p,n, Y,q, X,p, C,q` | `gemm(T,N,p,q,n,X,p,Y,q)` → `OP_N,OP_T,q,p,n,Y,q,X,p,C,q` | ✓ identical |
| YᵀY | `OP_N,OP_T, q,q,n, Y,q, Y,q, C,q` | `gemm(T,N,q,q,n,Y,q,Y,q)` → `OP_N,OP_T,q,q,n,Y,q,Y,q,C,q` | ✓ identical |

X and Y are each uploaded exactly once (`copy_h2d` at 1279-1280); XᵀX reuses `ws.dX()` as both operands, YᵀY reuses `ws.dY()`, no second upload. YᵀY (and its D2H and the `q*q` overflow check) is gated on `YtY != nullptr` (1252, 1260, 1306-1318, 1322). Results written row-major (symmetric products are transpose-invariant; XᵀY's q×p column-major → p×q row-major reading is the correct `(YᵀX)ᵀ = XᵀY`). Guards present: no-GPU→2 (1235), null X/Y/XtX/XtY→2 (1239), `>INT_MAX`→2 (1244), element + byte overflow→2 (1250-1268). `dgemm` only, fp64 `alpha/beta`, no `dsyrk`, no mixed precision.

### 2. `build_gram_device` — `[OK]`
`cpp/src/core/cuda_dispatch.cpp:1340-1411`. `K`: `OP_T,OP_N, n,n,p, Xw,p, Xw,p, C,n` (1386-1395). The fallback `gemm(N,T,n,n,p,Xw,p,Xw,p)` → wrapper emits `OP_T,OP_N,n,n,p,Xw,p,Xw,p,C,n` — identical. Single upload (1383), same `ws.dX()` pointer as both operands, row-major symmetric K out. Same guard set.

### 3. Integration / scope — `[OK]`
- `moments.cpp:193-217` — guarded `build_moments_device` under `#if N4M_USE_CUDA` + `cuda_runtime_available()`; on non-zero `built_on_device` stays false and the **verbatim** three `linalg::gemm` calls run (beta=0, full overwrite, so no partial-D2H leakage). `recompute_centered_moments` still runs (218). Covers both `compute_moments` and `compute_moments_subset` since both funnel here.
- `sweep.cpp:389-406` — identical guarded pattern for the gram in `prepare_ridge_dual_design`.
- Left **unchanged** as required: cross-gram `prepare_ridge_dual_cross` (`sweep.cpp:448-452`, still host `gemm`), the PLS scorer (`pls1_moment_components_*`), `prepare_dual_ridge`'s host-K upload, and `aom_sweep.cpp` (not in the changeset). `build_*_device` is referenced only in the 5 expected files.

### 4. Fold isolation — `[OK]`
Thread-local `ReusableMomentBuildWorkspace`/`ReusableGramBuildWorkspace` are grow-only (`cuda_dispatch.cpp:162-178`). Each call `copy_h2d`s the current fold's contiguous `n×p`/`n×q` into the leading region and cuBLAS reads only the leading region (lda = current `p`, m/n/k from current dims); D2H reads only current-fold output size; `beta=0` overwrites. A larger prior fold leaves capacity but no readable stale rows — no accumulation, nothing hoisted across folds. Builds run strictly after materialization (MSC/EMSC heads applied upstream in `transform_chain`); B3 moves no build above that boundary. Structured/banded/detrend routes untouched.

### 5. Equivalence tests — `[OK]`
`cpp/tests/test_internal_linalg.cpp:537-636`. Deterministic wide problem (n=96, p=160 → p>n; q=3 → q>1). Independent host triple-loop references (`reference_moments`/`reference_gram`, 105-164) are correct row-major; tol `1e-8` (values ~O(4), fp64 error ~1e-14 abs → huge margin, not flaky). Covers the `yty=nullptr` path (593-600) and the gram (603-636). Self-skips when `!cuda_runtime_available()` (538, 604); registered under `#if N4M_USE_CUDA` (648-653).

### 6. ABI / non-CUDA build — `[OK]`
Both functions live in `namespace n4m::cuda_dispatch`, **not** `N4M_API`, in the internal `cpp/src/core/cuda_dispatch.hpp` (no public header / catalog / Python change). With `CXX_VISIBILITY_PRESET hidden` (`n4m_targets.cmake:110-112`) + the `n4m_linux.map` `n4m_*`-only export script, they stay out of the dynamic symtab — same treatment as `spd_solve`/`prepare_dual_ridge`. `cuda_dispatch.cpp` is compiled only under `if(N4M_WITH_CUDA)` (`cpp/src/CMakeLists.txt:85-88`); the header include in `moments.cpp:14-16` and the call sites are all `#if defined(N4M_USE_CUDA)`-guarded — the dev-release build compiles the device branch out. ABI 1.22.0 unaffected.

### 7. Risk: single-pointer aliasing — `[OK]`
Passing the same device pointer as A and B to `cublasDgemm_v2` is well-defined: A and B are read-only inputs, and C (`dXtX`/`dYtY`/`dK`) is a distinct buffer (no in/out overlap). cuBLAS does not special-case `dgemm` for `A==B` (that's `dsyrk`), so the kernel/accumulation order is identical to the wrapper's two-distinct-buffers-with-identical-contents case → bit-equivalent. Synchronization is correct (all default-stream; blocking `cudaMemcpy` + same-stream `dgemm` order H2D→compute→D2H). Memory footprint per call is *lower* than the old per-GEMM wrapper, and `bad_alloc→return 2→host fallback` is the budget backstop.

---

### Non-blocking notes

- **[SCOPE]** `cpp/src/core/sweep.cpp` — the working-tree diff is **not pure B3**. Besides the gram drop-in (389-406), it also carries the B2 per-fold dual-ridge prep (`PreparedDualGpu` struct lines 85-113, `solve_dual_alpha` 544-579, `prepare_dual_ridge` wiring in `run_moment_sweep` ~2080-2117, and the `dual_gpu` threading through the predict/score/fit helpers). That code uses the **host-K** `prepare_dual_ridge` upload (consistent with design §6.2 staying deferred — not the deferred device-K handoff), so it's legitimate B2, not a B3 violation. Just confirm B2 and B3 are attributed/committed as intended; the B3-specific sweep change is only the 389-406 hunk. (Likewise the `cuda_kernels.{cu,hpp}` and CMake edits in the tree are from prior PLS-many-batched batches, not B3.)

- **[STYLE]** Pre-existing, not introduced here: `build_*_device` use the process-wide singleton `state().handle`, which is not thread-safe across concurrent fits — same contract as `gemm`/`spd_solve`, and the sweep is serial, so no new hazard. Worth keeping in mind for the Phase-45b per-stream pool.

The §3.5 unit equivalence gate is in place; the §5 GPU end-to-end candidate-score parity + bench run separately by the orchestrator as you noted.
