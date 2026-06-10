I've read all the load-bearing code. Here's the determination, then the roadmap.

---

# Roadmap: Route the Ridge GCV solve to the GPU (`libn4m` / `n4m`)

## 0. Executive summary

The measured 37%-CPU hotspot (`n4m_householder_qr`, `cpp/src/core/common/linalg.c:17`) is reached, for the wide‑`p` BERRY screen, through **`solve_square_qr` on the dual `n×n` Gram system `(K + λI) α = Y_work`**, re‑factorized from scratch once per `(λ, fold)`. The fix is a cuSOLVER **SPD Cholesky** drop‑in (`cusolverDnDpotrf`+`cusolverDnDpotrs`) behind a single host helper, with the host QR kept as fallback. No public symbol, ABI snapshot, catalog, or Python surface changes — everything lives in `n4m::cuda_dispatch` (hidden visibility) and `sweep.cpp`'s anonymous namespace.

---

## 1. What is actually solved on the hot screen path (source determination)

**For wide `p` (`p > n`), every CV fold uses the DUAL `n×n` path.** In `run_moment_sweep` (`cpp/src/core/sweep.cpp:1832`), the materialized‑fold decision at `sweep.cpp:1929-1938` sets `use_materialized_fold[fold]=1` whenever `p > train_rows[fold].size()`. For BERRY (`n=1434`, `p=2101`, 5‑fold → `n_train≈1147`), `2101 > 1147` holds for **all** folds, so the primal `p×p` moment path (`fit_ridge_from_moments`, `sweep.cpp:868`) and its eigen‑path amortization (`prepare_ridge_moment_eigen_path`, `sweep.cpp:898`) are **never entered**; the dual design is built instead (`prepare_ridge_dual_design`, `sweep.cpp:300`, building `K = Xc·Xcᵀ` at `sweep.cpp:358-363`).

**The system solved is the dual `n×n` SPD system**, identical at all three call sites — each forms `K = design.K; K[i·n+i] += λ` then calls `solve_square_qr(K, n, design.Y_work, q, α)`:
- `predict_ridge_from_dual_design` → `sweep.cpp:480-484`
- `ridge_dual_cross_heldout_sse` → `sweep.cpp:528-532`
- `fit_ridge_from_dual_design` → `sweep.cpp:574-578`

Properties:
- **SPD:** `K = Xc·Xcᵀ` is a Gram matrix → PSD; `+λI` with `λ>0` → strictly SPD. AOM screens enforce `λ>0` (`aom_sweep.cpp:435`, `ridge_grid_is_strictly_positive`), so on the hot path the matrix is *always* SPD. `λ=0` is only reachable through the generic `run_moment_sweep` validation (`sweep.cpp:1881` allows `λ≥0`).
- **RHS count `q`:** number of targets — `q=1` for the typical single‑reference NIRS screen; small (`≤ a few`) otherwise. RHS layout is **row‑major `n×q`** (`design.Y_work`, `solve_square_qr` loops targets at `sweep.cpp:441-456`).
- **λ‑grid sweep:** **one full re‑factorization per λ** on the dual path. `K` is shared across the whole grid (only the diagonal shift changes), but the dual path does **not** amortize — each λ rebuilds `K=design.K` and runs a fresh `O(n³)` Householder QR. (Contrast: the *primal* path already amortizes via `prepare_ridge_moment_eigen_path`/`fit_ridge_from_eigen_path`, `sweep.cpp:898`/`982`, but that path is dead for `p>n`.) **This non‑amortized per‑λ `O(n³)` refactor of an `n=1434` matrix is the hotspot.**
- **Dominant exact call site:** all three reduce to `solve_square_qr` on the `n×n` `K` (`sweep.cpp:426`). For a score‑only screen (`cfg.aom_score_only`, `sweep.cpp:2069` / `aom_sweep.cpp:539`), the branch is `ridge_dual_cross_heldout_sse` (**`sweep.cpp:532`**, via `sweep.cpp:2133`) when `should_use_ridge_dual_cross` is true (`sweep.cpp:413-424`; true only for very large λ‑grids), otherwise `fit_ridge_from_dual_design` (**`sweep.cpp:578`**, via `sweep.cpp:2144`). Both burn the same kernel, so a single fix at the shared solve covers every branch.
- **`ridge.cpp:233`** (`solve_square_qr` inside `fit_ridge`'s dual path, `cpp/src/core/ridge.cpp:214-233`) is the **standalone single‑shot** `fit_ridge` API — one solve per call, not in a λ‑loop. It is **not** the screen hotspot (the screen runs through `run_moment_sweep`), though it can reuse the same device helper for consistency (secondary).

**Conclusion:** target the dual `n×n` SPD solve shared by `sweep.cpp:484/532/578`.

---

## 2. Device approach: SPD Cholesky vs symmetric eigh

**Recommendation: ship cuSOLVER SPD Cholesky (`potrf`+`potrs`) first.** The λ‑sweep structure is the deciding factor, and it points to Cholesky for the first implementation for two concrete reasons:

1. **The sweep guarantees SPD‑ness on the hot path.** `K + λI` with the screen's `λ>0` (`aom_sweep.cpp:435`) is strictly SPD for *every* candidate, so `potrf` is the canonical, fastest, and numerically tightest solver and is guaranteed to succeed. Cholesky is `~⅓n³` flops (half of QR) and maps to one well‑optimized cuSOLVER pair. On `n=1434`, GPU `potrf` is sub‑millisecond; even re‑factoring per λ, the 37% CPU hotspot is eliminated entirely.
2. **Minimal surface, trivial parity, ABI‑safe.** It is a near drop‑in for `solve_square_qr` (same system, same row‑major RHS), so it lands behind a one‑line host fallback with a tight, directly‑testable fp64 equivalence vs the host QR — satisfying the "keep host fallback" and "ABI frozen" constraints with the least risk.

**Why not eigh first.** `cusolverDnDsyevd` once per fold (mirroring the host primal eigen‑path `sweep.cpp:898`/`982`) would amortize the `O(n³)` decomposition across the λ‑grid, cutting per‑λ cost to `O(n²q)`. That is the asymptotically‑superior endpoint, but: (a) the per‑λ Cholesky is already off the critical path after Phase 1 (saving the residual `~0.2–0.5 s` of GPU refactor vs ~26 s of CPU QR removed); (b) it requires a new dual eigen‑path data structure + prepare/solve functions + orchestration wiring (a much larger change); and (c) it is numerically more delicate near `μ+λ≈0`. **Eigh is a clearly‑scoped Phase‑2 follow‑up (§6), to be implemented only if post‑Phase‑1 profiling shows the per‑λ device Cholesky still material** — which it will not be for the measured case.

---

## 3. Precise specification for Codex

### 3.1 New internal device function (`n4m::cuda_dispatch`)

**Declaration** — add to `cpp/src/core/cuda_dispatch.hpp` after the PLS block (after line 107), **no `N4M_API` decoration** (stays hidden‑visibility / internal):

```cpp
// Device SPD linear solve via cuSOLVER Cholesky. Solves A X = B where
//   A : n x n row-major, SYMMETRIC POSITIVE DEFINITE (= K + lambda*I, lambda>0)
//   B : n x q row-major (q right-hand sides / targets)
//   X : n x q row-major output (caller-allocated, n*q doubles)
// A is symmetric so its row-major and column-major images coincide; B and X are
// repacked to/from column-major internally. The caller's A/B are not modified.
// Return codes (match the pls1_moment_components convention):
//   0 : success
//   1 : not positive definite (potrf leading-minor failure) -> caller falls
//       back to the host QR solve. This is the lambda==0 / rank-deficient case.
//   2 : CUDA / cuSOLVER / allocation failure, or no GPU available.
int spd_solve(std::size_t n, std::size_t q,
              const double* A, const double* B, double* X,
              std::string* error);
```

**Definition** — add to `cpp/src/core/cuda_dispatch.cpp` (new public function near `pls1_moment_components`, `cuda_dispatch.cpp:1142`), wrapped in the same `try/catch` envelope (`cuda_dispatch.cpp:1153-1188`: `bad_alloc`→2, `std::exception`→2, `...`→2). Body:

1. Guard: `if (!cuda_runtime_available()) { set_error(...); return 2; }`; validate `n>0`, `q>0`, non‑null pointers, and `n,q ≤ INT_MAX` (cuSOLVER takes `int`), else `return 2` (mirror `cuda_dispatch.cpp:1160-1169`).
2. `DevicePtr<double> dA(n*n); DevicePtr<double> dB(n*q); DevicePtr<int> dInfo(1);` (reuse `DevicePtr`, `cuda_dispatch.cpp:81`).
3. `copy_h2d(dA.get(), A, n*n*sizeof(double))` — **A symmetric ⇒ row‑major image == column‑major image, no transpose** (`copy_h2d`, `cuda_dispatch.cpp:295`).
4. **Pack B to column‑major** in a host staging buffer: `std::vector<double> Bcm(n*q); for i,t: Bcm[t*n+i] = B[i*q+t];` then `copy_h2d(dB.get(), Bcm.data(), n*q*sizeof(double))`. (For `q==1` this is a contiguous copy.)
5. `const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;` — both triangles of `A` are materialized by the host (`prepare_ridge_dual_design` fills full `K` via GEMM; the λ‑shift touches only the diagonal), so `uplo` choice is immaterial; document this.
6. Workspace size + factor:
   ```cpp
   int lwork = 0;
   check_cusolver(cusolverDnDpotrf_bufferSize(state().solver, uplo, (int)n, dA.get(), (int)n, &lwork), "potrf_bufferSize");
   DevicePtr<double> dWork(static_cast<std::size_t>(lwork));
   check_cusolver(cusolverDnDpotrf(state().solver, uplo, (int)n, dA.get(), (int)n, dWork.get(), lwork, dInfo.get()), "potrf");
   int info = 0; copy_d2h(&info, dInfo.get(), sizeof(int));
   if (info > 0) { set_error(error, "ridge SPD solve: matrix not positive definite"); return 1; }  // lambda==0 / rank-deficient
   if (info < 0) { set_error(error, "ridge SPD solve: potrf invalid argument"); return 2; }
   ```
7. Solve (in‑place in `dB`):
   ```cpp
   check_cusolver(cusolverDnDpotrs(state().solver, uplo, (int)n, (int)q, dA.get(), (int)n, dB.get(), (int)n, dInfo.get()), "potrs");
   copy_d2h(&info, dInfo.get(), sizeof(int));
   if (info != 0) { set_error(error, "ridge SPD solve: potrs failed"); return 2; }
   ```
8. **Unpack X from column‑major:** `std::vector<double> Xcm(n*q); copy_d2h(Xcm.data(), dB.get(), n*q*sizeof(double)); for i,t: X[i*q+t] = Xcm[t*n+i];`
9. `return 0;`

Add a `check_cusolver(cusolverStatus_t, const char*)` helper next to `check_cublas` (`cuda_dispatch.cpp:338`) that throws `std::runtime_error` on `!= CUSOLVER_STATUS_SUCCESS` (so the `try/catch` maps it to return code 2). Add `#include <cusolverDn.h>` next to the cuBLAS include (`cuda_dispatch.cpp:28`).

> Note: `state().handle`/`state().solver` use the **default stream** — `cudaMemcpy` (`copy_d2h`) is synchronizing, so reading `dInfo` after `potrf`/`potrs` is correctly ordered without an explicit `cudaDeviceSynchronize`.

### 3.2 cuSOLVER handle on the `CublasState` singleton

In `cpp/src/core/cuda_dispatch.cpp`, extend `struct CublasState` (`cuda_dispatch.cpp:46-72`):
- Add member: `cusolverDnHandle_t solver{};`
- In the constructor (after the successful `cublasCreate_v2`, `cuda_dispatch.cpp:58-61`), create the solver handle; **only set `available=true` if both succeed**, and roll back cuBLAS on solver failure:
  ```cpp
  if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) { return; }
  if (cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS) { cublasDestroy_v2(handle); return; }
  available = true;
  ```
- In the destructor (`cuda_dispatch.cpp:64-68`), destroy it first: `if (available) { cusolverDnDestroy(solver); cublasDestroy_v2(handle); }`.

Lifetime/lazy‑init/memoization are inherited from the existing `state()` Meyers singleton (`cuda_dispatch.cpp:74-77`); no other lifetime work.

### 3.3 CMake — link `CUDA::cusolver`

`find_package(CUDAToolkit REQUIRED)` (`CMakeLists.txt:114`) already exposes the `CUDA::cusolver` imported target. Add it at **both** existing CUDA link sites (keep them in lockstep):
- **`cpp/src/n4m_targets.cmake:148`** (shared `n4m_c`): `CUDA::cudart CUDA::cublas` → `CUDA::cudart CUDA::cublas CUDA::cusolver`
- **`cpp/src/CMakeLists.txt:285`** (static `n4m_c_static`, consumed by `n4m_internal_tests`): same edit.

The internal‑test executable inherits cuSOLVER transitively through the static archive's link deps (same mechanism that currently delivers cuBLAS). If the test link fails to resolve `cusolverDn*`, also add `CUDA::cusolver` to `n4m_internal_tests` in `cpp/tests/CMakeLists.txt:89`.

### 3.4 Host integration point + gating heuristic

Add one helper in `sweep.cpp`'s anonymous namespace, just **above** `solve_square_qr` (`sweep.cpp:426`):

```cpp
// CUDA dual-Ridge gating: only route the n x n SPD solve to the GPU when a
// device is present and the system is large enough to amortize the H2D/D2H
// round-trip. Conservative crossover; N4M_CUDA_RIDGE_DISABLE forces host QR.
[[nodiscard]] bool ridge_cuda_dual_enabled(std::size_t n) noexcept {
#if defined(N4M_USE_CUDA)
    constexpr std::size_t kRidgeCudaMinDualSamples = 256;
    if (n < kRidgeCudaMinDualSamples) return false;
    if (const char* e = std::getenv("N4M_CUDA_RIDGE_DISABLE");
        e != nullptr && e[0] != '\0' && e[0] != '0') return false;
    return ::n4m::cuda_dispatch::cuda_runtime_available();
#else
    (void)n; return false;
#endif
}

// Dual SPD solve: GPU Cholesky when enabled, else the host Householder QR.
[[nodiscard]] n4m_status_t solve_dual_spd(const std::vector<double>& A,
                                          std::size_t n,
                                          const std::vector<double>& B,
                                          std::size_t q,
                                          std::vector<double>& X) {
#if defined(N4M_USE_CUDA)
    if (ridge_cuda_dual_enabled(n)) {
        X.assign(n * q, 0.0);
        std::string err;
        const int s = ::n4m::cuda_dispatch::spd_solve(n, q, A.data(), B.data(),
                                                      X.data(), &err);
        if (s == 0) return N4M_OK;
        // s==1 (not PD: lambda==0 / rank-deficient) or s==2 (runtime): fall
        // through to the host QR, which also rejects singular systems cleanly.
    }
#endif
    return solve_square_qr(A, n, B, q, X);
}
```

Then **replace the three dual call sites** `solve_square_qr(K, n, design.Y_work, q, alpha)` → `solve_dual_spd(K, n, design.Y_work, q, alpha)` at:
- `sweep.cpp:484` (`predict_ridge_from_dual_design`)
- `sweep.cpp:532` (`ridge_dual_cross_heldout_sse`)
- `sweep.cpp:578` (`fit_ridge_from_dual_design`)

This mirrors the PLS scorer's automatic gating (`sweep.cpp:1443-1469`: `if (threshold && cuda_runtime_available())`) and requires **no `Config` field and no `n4m_config_*` ABI change** — the threshold is a compile‑time constant plus an env kill‑switch, matching the existing `truthy_env` idiom (`cuda_dispatch.cpp:372`). Leave `sweep.cpp:676` (tiny `k×k` `PᵀW` inverse) and the primal `sweep.cpp:868` on the host (the host eigen‑path already covers the multi‑λ primal case; routing `868` is an optional extra, not required).

Optionally apply the same swap at `ridge.cpp:233` for the standalone `fit_ridge` dual path (secondary; not the screen hotspot).

### 3.5 Host fallback path

`solve_square_qr` (`sweep.cpp:426`) is retained verbatim and reached whenever: CUDA is not compiled (`#else`), no GPU (`cuda_runtime_available()==false`), `n < 256`, `N4M_CUDA_RIDGE_DISABLE` set, `potrf` reports not‑PD (return 1 — e.g. `λ=0` rank‑deficient), or any cuSOLVER/runtime failure (return 2). The host QR independently rejects singular `R` (`n4m_back_solve_R`, `linalg.c:107-135`), so a `λ=0` rank‑deficient system fails cleanly on either branch.

### 3.6 fp64 equivalence guarantee + concrete test

**Guarantee:** Cholesky and Householder QR solve the *same* SPD system; for a well‑conditioned `K+λI` (`λ>0`) both deliver `α` to `~κ(A)·ε` relative error. Identical dtype (fp64) throughout — `spd_solve` does no fp32 narrowing.

**Test** — add `TEST`/checks to `cpp/tests/test_internal_linalg.cpp` (links `n4m_c_static`, which carries `N4M_USE_CUDA` + cuSOLVER). Guard and self‑skip:

```cpp
#if defined(N4M_USE_CUDA)
#include "core/cuda_dispatch.hpp"
#include "core/common/linalg.h"   // n4m_householder_qr / n4m_apply_qt / n4m_back_solve_R
// ... inside a test case:
if (!n4m::cuda_dispatch::cuda_runtime_available()) { /* skip: no GPU */ }
else {
    constexpr std::size_t n = 64, q = 3;
    // Deterministic SPD A = Gᵀ·G + I (well conditioned), full symmetric storage.
    // Deterministic B (n x q, row-major).
    // Reference: per-target solve via n4m_householder_qr + n4m_apply_qt + n4m_back_solve_R
    //            on a copy of A (exactly what solve_square_qr does), into X_ref.
    // Device:    n4m::cuda_dispatch::spd_solve(n, q, A.data(), B.data(), X_dev.data(), &err) == 0
    // Assert: max_{i,t} |X_dev[i*q+t] - X_ref[i*q+t]| <= kSpdTol;   kSpdTol = 1e-8
}
#endif
```

**Tolerance: `kSpdTol = 1e-8` (absolute, max componentwise)** — consistent with the existing internal‑sweep convention (`test_internal_sweep.cpp:31`, `kTol=1e-8`); typical agreement on a well‑conditioned matrix is `~1e-11`. Add a second assertion: a singular case (`A = Gᵀ·G` with rank‑deficient `G`, i.e. `λ=0`) returns `spd_solve(...) == 1` (graceful not‑PD), proving the fallback signal.

**CMake for the test guard:** add to `cpp/tests/CMakeLists.txt` (so the test TU sees the macro — `n4m_c_static`'s `N4M_USE_CUDA` is `PRIVATE` and does not propagate to the test source):
```cmake
target_compile_definitions(n4m_internal_tests PRIVATE
    $<$<BOOL:${N4M_WITH_CUDA}>:N4M_USE_CUDA=1>)
```

---

## 4. Correctness traps for Codex

1. **SPD only for `λ>0`.** `potrf` *fails* (devInfo>0) on `λ=0` rank‑deficient `K`. Do **not** treat that as fatal — `spd_solve` returns `1` and `solve_dual_spd` falls through to the host QR. The hot AOM screen always has `λ>0` (`aom_sweep.cpp:435`), so this is purely the generic‑caller safety net.
2. **Per‑fold isolation / no leakage.** `K`, `K_cross`, `Y_work` are built strictly from fold‑local train rows (`prepare_ridge_dual_design`, `sweep.cpp:300-364`; held‑out rows only enter `K_cross`, never `K`). `spd_solve` operates solely on its passed‑in `A`,`B` and the singleton handle/`DevicePtr`s, which are overwritten each call and carry **no** fold‑to‑fold state. Do not introduce any reused/persistent device buffer keyed across folds in Phase 1 — it would risk exactly the leakage the dual design avoids.
3. **Symmetric row↔col‑major.** `A` symmetric ⇒ row‑major image equals column‑major image; upload as‑is, no transpose, `uplo` immaterial (both triangles present). **`B`/`X` are *not* symmetric** — they MUST be packed to / unpacked from column‑major (`§3.1` steps 4 & 8). Skipping this silently transposes the RHS for `q>1`.
4. **Workspace sizing + devInfo.** Always size via `cusolverDnDpotrf_bufferSize` before allocating `dWork`; never assume a size. Check `dInfo` after **both** `potrf` and `potrs` (copy to host with the synchronous `copy_d2h`). `info>0` from `potrf` = not‑PD (→1); `info<0` = bad argument (→2).
5. **Numerical tolerance.** Equivalence test asserts `≤1e-8` absolute (§3.6). Do not tighten below `~1e-10` — `potrf`/QR pivoting differ in rounding.
6. **Thread‑safety of the singleton handle.** The `cusolverDnHandle_t` (like `cublasHandle_t`) is **not** thread‑safe. The integration point is provably **serial**: the candidate‑λ × fold loop in `run_moment_sweep` (`sweep.cpp:2114-2230`) runs single‑threaded, exactly as `pls1_moment_components_many_sequential` already reuses `state().handle` serially (`cuda_dispatch.cpp:773`). Add a one‑line comment at `solve_dual_spd` documenting this serial contract. **Do not** add a mutex (the existing `gemm`/`gemv`/`ger` singleton users don't, `cuda_dispatch.cpp:1384/1471/1518`); concurrent `run_moment_sweep` calls from multiple threads are a pre‑existing limitation this change *inherits*, not introduces. If a future parallel‑fold ridge path is added, it must use a per‑thread local handle (mirror `LocalCublasHandle`, `cuda_dispatch.cpp:523`).
7. **Buffer truncation contract.** `spd_solve` follows the existing contiguous‑buffer contract (`cuda_dispatch.hpp:35-40`): `A` is `n×n` contiguous (`lda==n`), `B`/`X` are `n×q` contiguous (`ld==q`). The dual `K`/`Y_work` are dense contiguous `std::vector`s, so this holds; assert/document it.

---

## 5. Step‑by‑step roadmap (file‑by‑file) + green gates

**Phase 1 — GPU SPD Cholesky drop‑in**

1. `cpp/src/core/cuda_dispatch.cpp:28` — add `#include <cusolverDn.h>`.
2. `cpp/src/core/cuda_dispatch.cpp:46-72` — add `cusolverDnHandle_t solver{}` to `CublasState`; create in ctor after cuBLAS (gate `available` on both), destroy in dtor (§3.2).
3. `cpp/src/core/cuda_dispatch.cpp:338` — add `check_cusolver(...)` helper.
4. `cpp/src/core/cuda_dispatch.cpp:~1142` — add `int spd_solve(...)` (§3.1), in the `try/catch` envelope.
5. `cpp/src/core/cuda_dispatch.hpp:107` — declare `spd_solve` (no `N4M_API`).
6. `cpp/src/n4m_targets.cmake:148` and `cpp/src/CMakeLists.txt:285` — append `CUDA::cusolver`.
7. `cpp/src/core/sweep.cpp:426` — add `ridge_cuda_dual_enabled` + `solve_dual_spd` above `solve_square_qr` (§3.4).
8. `cpp/src/core/sweep.cpp:484`, `:532`, `:578` — swap `solve_square_qr` → `solve_dual_spd`.
9. `cpp/tests/CMakeLists.txt:89` — add the `N4M_USE_CUDA` generator‑expr compile‑def to `n4m_internal_tests` (§3.6); add `CUDA::cusolver` to it only if the link fails.
10. `cpp/tests/test_internal_linalg.cpp` — add the Cholesky‑vs‑host‑QR equivalence test + the `λ=0`‑returns‑1 case (§3.6).

**Green gates (run in order):**

- **CPU build is byte‑identical (no GPU regression):**
  `cmake --preset dev-release && cmake --build --preset dev-release -j && ctest --preset dev-release --output-on-failure`
  (the `#if defined(N4M_USE_CUDA)` guards keep the host path untouched; `n4m_tests` + `n4m_internal_tests` must stay green).
- **CUDA build + equivalence test:**
  `cmake --preset cuda-on && cmake --build --preset cuda-on -j && ctest --preset cuda-on --output-on-failure`
  (the new equivalence test runs under `N4M_USE_CUDA`; self‑skips cleanly if no GPU is present).
- **ABI gate proves no public‑surface change** (`spd_solve` is internal, hidden visibility, not `N4M_API`):
  `scripts/bump_version.sh --check`, then the symbol diff `nm -D --defined-only build/dev-release/cpp/src/libn4m.so.<ver> | awk '{print $3}' | sort -u | diff -u cpp/abi/expected_symbols_linux.txt -` — must be empty. ABI stays **1.22.0**; do not regenerate snapshots.
- **Smoke + lint:** `build/cuda-on/cpp/cli/n4m_cli --selfcheck`; `clang-format`/`clang-tidy` clean on the touched files.
- **Optional parity sanity (integration):** run `run_moment_sweep` on a `p>n` fixture with and without `N4M_CUDA_RIDGE_DISABLE=1`; assert `SweepResult.candidate_scores` agree within `1e-8`.

**Phase 2 — eigh amortization (OPTIONAL; only if Phase‑1 profiling still shows the dual solve material)**

Do not build unless measured. Mirror the proven primal eigen‑path on the dual side:
- New `struct RidgeDualEigenPath { n_samples, n_targets; eigenvalues(n); eigenvectors(n*n); projected_Y(n*q = Vᵀ·Y_work); + K_cross/x_mean/y_mean/x_scale }` next to `RidgeMomentEigenPath` (`sweep.cpp:74`).
- `prepare_ridge_dual_eigen_path(...)`: one device `cusolverDnDsyevd` of `K` per materialized fold (mirror `prepare_ridge_moment_eigen_path`, `sweep.cpp:898`); precompute `Z = Vᵀ·Y_work`.
- `solve_dual_from_eigen_path(path, λ, α)`: `α = V·diag(1/(μ+λ))·Z`, `O(n²q)` per λ, reusing the **same** PSD clamp/tolerance as `fit_ridge_from_eigen_path` (`sweep.cpp:1004-1030`).
- Wire alongside the existing per‑fold eigen‑path prep (`sweep.cpp:2096-2112`) so the λ‑loop (`sweep.cpp:2114`) consumes it for predict/sse/fit. Same `1e-8` fp64 parity test, plus reuse of the already‑accepted host primal eigen‑path numerics as the model.

---

This keeps the change internal to `n4m::cuda_dispatch` + `sweep.cpp`'s anonymous namespace, preserves the host QR as fallback, holds ABI at 1.22.0, and removes the 37% `n4m_householder_qr` hotspot on the wide‑`p` screen with a single, equivalence‑tested device solve.
