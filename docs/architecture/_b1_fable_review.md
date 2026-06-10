I have completed my review. Here is my assessment.

---

# Adversarial Review: Ridge GCV → cuSOLVER SPD solve (Phase 1)

## VERDICT: **fix-then-ship**

The actual Phase‑1 deliverable — `spd_solve`, the cuSOLVER handle, the gating helper, the 3 call‑site swaps, the CMake link/compile‑def edits, and the equivalence test — is **correct and faithful to the approved design**. I could not find a correctness, ABI, or fold‑isolation bug in any of it.

There is **one real problem**: an unrelated, out‑of‑scope rewrite of the hot `gemm()` dispatch path is bundled into the same diff. It isn't in the approved design, it touches a far broader blast radius than the isolated solve, and it's not covered by the equivalence test. Split it out (or justify+test it separately) before merging; the Ridge work itself is shippable.

---

## Findings

**1. [SCOPE] `gemm()` was rewritten to use `static thread_local ReusableDeviceBuffer` scratch — not part of this design.** `cpp/src/core/cuda_dispatch.cpp:1559-1572` (and the `dA/dB/dC` deref changes at `:1597-1603`). The approved design (§3) scopes Phase 1 to `spd_solve` + the cuSOLVER handle + CMake + the 3 dual call‑site swaps + the test. This `gemm` change is a separate performance optimization (replacing per‑call `DevicePtr` with grow‑only per‑thread scratch to avoid `cudaMalloc/cudaFree` device‑sync stalls). It is *plausibly correct* — `dA`/`dB` are fully overwritten by `copy_h2d` each call, `dC` is only read when `beta != 0` (same contract as the original uninitialized `DevicePtr`), grow‑only `ensure()` is sound, and `thread_local` gives per‑thread isolation — but:
   - It modifies the primitive used by **every** CUDA GEMM in libn4m, including the `K = Xc·Xcᵀ` build for the dual design and all PLS moment builds — much wider than the isolated `spd_solve`.
   - It is **not exercised by the equivalence test** (which only calls `spd_solve` directly), and the design's optional moment‑sweep parity sanity (§5) wasn't added.
   - §4.2 of the design explicitly cautioned against introducing reused/persistent device buffers in Phase 1.
   - **Fix:** extract this into its own PR with its own benchmark + a `run_moment_sweep` parity check (`N4M_CUDA_RIDGE_DISABLE=1` vs not, scores agree ≤1e‑8). If it must ship together, call it out explicitly and add the parity gate.

**2. [STYLE] `static thread_local` lifetime footnote (only if finding #1 stays).** `cuda_dispatch.cpp:1564-1566`. The scratch destructors run `cudaFree` at thread/process exit, where the CUDA runtime may already be torn down (`cudaErrorCudartUnloading`). The return is ignored, so it's benign, and it mirrors the existing `CublasState` dtor pattern — but it's a new lifetime that the per‑call `DevicePtr` didn't have, and the buffers stay resident at high‑water mark for the process lifetime. Acceptable as a documented tradeoff; noting for completeness.

**3. [OK] `spd_solve` correctness** (`cuda_dispatch.cpp:1156-1260`). Verified against §3.1 and the §4 traps:
   - A uploaded as‑is, no transpose — correct (A symmetric ⇒ row‑major image == column‑major image; `uplo` immaterial, documented at `:1207`).
   - B packed column‑major `Bcm[target*n+i]=B[i*q+target]` (`:1200-1204`) and X unpacked inverse `X[i*q+target]=Xcm[target*n+i]` (`:1242-1245`) — both correct.
   - `devInfo` checked after **both** `potrf` (`:1219-1228`) and `potrs` (`:1234-1238`).
   - Return codes exactly per spec: `potrf info>0 → 1` (not‑PD), `info<0 → 2`, `potrs info!=0 → 2`, and `check_cusolver` throws → caught by the `bad_alloc/exception/...` envelope → `2` (`:1248-1259`).
   - Workspace sized via `cusolverDnDpotrf_bufferSize` (`:1210-1213`), no assumed size.
   - Overflow/validity guards are thorough: null/zero, `n,q ≤ INT_MAX`, `mul_overflows(n,n)`, `mul_overflows(n,q)`, and byte‑size overflow (`:1170-1189`).
   - No leaks (RAII `DevicePtr` for `dA/dB/dInfo/dWork`, `std::vector` staging). Default‑stream ordering is correct — `copy_d2h` of `info` is a synchronous `cudaMemcpy` on the default stream, so it waits for `potrf`/`potrs` (no explicit sync needed).

**4. [OK] Scope of the swaps.** Exactly the 3 dual sites route to `solve_dual_spd`: `predict_ridge_from_dual_design` (`sweep.cpp:529`), `ridge_dual_cross_heldout_sse` (`:577`), `fit_ridge_from_dual_design` (`:623`). The k×k `PᵀW` inverse (`sweep.cpp:721`), the primal moment solve (`:913`), and the PLS‑prefix inverse (`:1403`) all remain on host `solve_square_qr`. `ridge.cpp` is untouched (confirmed via diff‑stat). No Phase‑2 eigh code was added.

**5. [OK] Fallback + gating** (`sweep.cpp:435-469`). `solve_dual_spd` falls through to host QR on `s==1` **and** `s==2` (the `if (s==0) return; ` leaves both other codes falling out of the `#if` block to `solve_square_qr`). `ridge_cuda_dual_enabled`: `n<256 → false`, `N4M_CUDA_RIDGE_DISABLE` honored, `cuda_runtime_available()` checked, `#else → false`. The λ=0 generic case is handled correctly: `potrf` reports not‑PD → `1` → host QR, which itself rejects the singular `R` with `N4M_ERR_NUMERICAL_FAILURE` — identical to the pre‑change behavior. The `[[maybe_unused]]` on `ridge_cuda_dual_enabled` correctly prevents an unused‑function `-Werror` failure in the non‑CUDA build.

**6. [OK] Fold‑clean.** `spd_solve` uses only per‑call `DevicePtr`s and the passed‑in `A`/`B`; no persistent or cross‑fold device buffer. `K`/`Y_work` are fold‑local. No fold‑to‑fold state in the solve path. (Caveat: finding #1's `gemm` scratch *is* reused, but it's fully overwritten each call so it carries no numerical state — not a leakage vector, but it's the kind of reuse §4.2 flagged.)

**7. [OK] ABI.** `spd_solve` is declared only in the internal `cuda_dispatch.hpp`, has no `N4M_API`, and the project compiles with `-fvisibility=hidden` (`CMakeLists.txt:77-79`, `n4m_targets.cmake:111-112`) plus the `n4m_linux.map` version script — so it stays out of the dynamic symbol table, exactly like the existing `gemm`/`pls1_*` internals. No `cpp/abi/`, `cpp/include/`, `bindings/`, or `catalog/` changes (confirmed via diff‑stat). cuSOLVER handle lifecycle is correct: created **after** cuBLAS, `available` gated on **both** succeeding, cuBLAS rolled back (`cublasDestroy_v2` + `handle=nullptr`) if `cusolverDnCreate` fails, and destroyed before cuBLAS in the dtor (`cuda_dispatch.cpp:47-76`).

**8. [OK] The test** (`test_internal_linalg.cpp:304-371`). It's a real test: builds a deterministic SPD `A = (GᵀG)/n + I` (eigenvalues ≥ 1, well‑conditioned), solves via `spd_solve` **and** a host QR reference (`qr_square_multi`, byte‑for‑byte the same algorithm as the production `solve_square_qr`), and asserts max abs diff ≤ `1e-8`. It self‑skips with a printed message when no GPU is present. The not‑PD path is covered with a `diag(1,…,1,0)` matrix (a clean zero pivot → `potrf info=n>0`), asserting `spd_solve(...) == 1`. The `#if defined(N4M_USE_CUDA)` guards (helper, test, and invocation) plus the `cpp/tests/CMakeLists.txt:94-95` generator‑expression compile‑def correctly make the TU see the macro that `n4m_c_static` keeps `PRIVATE`. Note: it tests `n=64 < 256` but calls `spd_solve` **directly**, bypassing the `n<256` gate — correct, so the device path is actually exercised.

**9. [STYLE] `N4M_CUDA_RIDGE_DISABLE` semantics** (`sweep.cpp:439-442`). Any non‑empty value other than a leading `'0'` disables — so `N4M_CUDA_RIDGE_DISABLE=false` would *also* disable CUDA, diverging from the `truthy_env` idiom used elsewhere (`cuda_dispatch.cpp:386`). This matches the design's §3.4 spec verbatim and is defensible for a kill‑switch, so it's not a bug — just a minor inconsistency worth a one‑line comment if you care about env‑var uniformity.

---

**Bottom line:** the Ridge spd_solve change is clean, correct, ABI‑safe, fold‑isolated, and well‑tested. Pull the `gemm()` `thread_local`‑scratch optimization out of this diff (finding #1) and the Phase‑1 work ships as designed. The orchestrator's GPU run should confirm the `≤1e-8` equivalence and the not‑PD `==1` assertions, which my static read indicates will pass.
