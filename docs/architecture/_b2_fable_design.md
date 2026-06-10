I have everything I need. I traced the full dual path, the `spd_solve`/`ReusableDeviceBuffer`/kernel infrastructure, the build wiring, and the existing test. Here is the implementation roadmap.

---

# B2 Implementation Roadmap — Device-resident dual-Ridge solve (upload-once K + reuse-buffers Cholesky)

**Scope:** internal `cpp/src/core/cuda_dispatch.{hpp,cpp}`, `cpp/src/core/cuda_kernels.{hpp,cu}`, `cpp/src/core/sweep.cpp`, and one test in `cpp/tests/test_internal_linalg.cpp`. **No** public `n4m_*` symbol, catalog, or Python change. ABI stays **1.22.0** (these are internal `n4m::cuda_dispatch` C++ symbols, hidden by visibility — same status as B1's `spd_solve`). fp64 throughout. Host fallback preserved; `N4M_CUDA_RIDGE_DISABLE` still forces the full host path.

---

## 1. Confirmed call structure (with the loop quoted)

**The gram `K` is built once per fold, and the three consumers are then called per‑λ on that same `K`.** Confirmed by reading the source:

- `prepare_ridge_dual_design` (`sweep.cpp:301`) builds `out.K = X_work·X_workᵀ` exactly once via `linalg::gemm` at `sweep.cpp:359-364`.
- All three per‑λ consumers re-form `K` on the host and re-solve, every call, on the **identical** `design.K` and **identical** `design.Y_work` RHS:
  - `predict_ridge_from_dual_design` — `K = design.K; K[i*n+i] += lambda` (`sweep.cpp:525-526`), `solve_dual_spd(K, n, design.Y_work, q, alpha)` (`sweep.cpp:529`).
  - `ridge_dual_cross_heldout_sse` — same shift (`sweep.cpp:573-574`), same solve (`sweep.cpp:577`).
  - `fit_ridge_from_dual_design` — same shift (`sweep.cpp:619-620`), same solve (`sweep.cpp:623`).
- `solve_dual_spd` (`sweep.cpp:451`) routes to `cuda_dispatch::spd_solve` (`sweep.cpp:462`) when `ridge_cuda_dual_enabled(n)` (`sweep.cpp:435`), else host `solve_square_qr` (`sweep.cpp:468`). `spd_solve` (`cuda_dispatch.cpp:1156`) re‑uploads the full n×n `A` (`copy_h2d` at `:1197`) and allocates fresh `DevicePtr`s every call (`:1193-1195`, `:1214`) — this is the measured H2D + malloc churn.

**The prep loop** (built once per fold, `sweep.cpp:2018-2044`):

```cpp
std::vector<RidgeDualDesign> fold_dual_designs(static_cast<std::size_t>(actual_cv));
std::vector<unsigned char> use_dual_cross_fold(static_cast<std::size_t>(actual_cv), 0);
if (do_ridge) {
    for (std::int32_t fold = 0; fold < actual_cv; ++fold) {
        const auto fold_ix = static_cast<std::size_t>(fold);
        if (use_materialized_fold[fold_ix] == 0) continue;
        st = prepare_ridge_dual_design(ctx, cfg, fold_designs[fold_ix].X_train, ...,
                                       fold_dual_designs[fold_ix]);     // builds .K ONCE
        if (st != N4M_OK) return st;
        if (should_use_ridge_dual_cross(...)) {
            ...
            st = prepare_ridge_dual_cross(ctx, cfg, fold_dual_designs[fold_ix], Xc,
                                          fold_rows[fold_ix], fold_dual_designs[fold_ix]);
            use_dual_cross_fold[fold_ix] = 1;
        }
    }
}
```

**The consuming λ‑outer / fold‑inner loop** (`sweep.cpp:2159-2200`, condensed):

```cpp
for (std::size_t candidate = 0; candidate < lambdas.size(); ++candidate) {   // λ OUTER
    const double lambda = lambdas[candidate];
    for (std::int32_t fold = 0; fold < actual_cv; ++fold) {                   // fold INNER
        const auto fold_ix = static_cast<std::size_t>(fold);
        if (use_materialized_fold[fold_ix] != 0) {
            if (use_dual_cross_fold[fold_ix] != 0) {
                if (score_only) ridge_dual_cross_heldout_sse(ctx, fold_dual_designs[fold_ix], ..., lambda, fold_sse); // :2178
                else            predict_ridge_from_dual_design(ctx, fold_dual_designs[fold_ix], lambda, pred);        // :2183
            } else {
                fit_ridge_from_dual_design(ctx, fold_dual_designs[fold_ix], lambda, fit);                             // :2189
            }
        } else { /* moment / eigen-path branch — not dual */ }                // :2201
    }
}
```

**Key structural facts that drive the design:**
- Loop order is **λ‑outer, fold‑inner**, so a single device‑resident K per fold must persist for the *whole* λ loop (all folds coexist). This exactly mirrors the existing host eigen‑path vector `std::vector<RidgeMomentEigenPath> ridge_moment_eigen_paths` prepared per fold at `sweep.cpp:2139-2157` (gated `lambdas.size() > 1U`, `sweep.cpp:2141`) and consumed per (λ,fold) in the `else` branch at `sweep.cpp:2203-2211`.
- For BERRY n=1434, `design.K` is 1434²×8 ≈ **16.4 MB**, re-uploaded once per (λ×fold) ≈ 360 calls = the 1.77 s `cudaMemcpy` and 1232 `cudaMalloc`s the profile flagged.
- The two final‑fit sites (`sweep.cpp:2553`, `sweep.cpp:2657`) are single‑λ fits on the full‑data `all_dual_design`; they are **not** in the grid loop and need no prepared handle.

---

## 2. Recommendation: **Shape (A) — Upload‑once + reuse‑buffers Cholesky** (variant A1: per‑fold device‑resident K)

**Recommended. Reject (B) eigh amortization for B2.**

Justification, anchored to the measured profile:

| | (A) Upload‑once + reuse‑buffers Cholesky | (B) Eigh amortization |
|---|---|---|
| Kills per‑λ K H2D (16 MB × ~360) — **the 78% cost** | ✅ (one H2D per fold) | ✅ (one H2D per fold) |
| Kills per‑call malloc churn (1232) | ✅ (persistent + shared reusable buffers) | ✅ |
| Per‑λ device work | one D2D 16 MB + tiny diag kernel + `potrf`/`potrs` (**sub‑ms per the profile**) | tiny diag‑scale kernel + 1 gemm (`O(n²q)`) — slightly less, but `potrf` was never the bottleneck |
| Per‑fold prep cost | upload K (one H2D) | `cusolverDnDsyevd` of K (`O(n³)`, materially heavier than one H2D) |
| Numerical equivalence to current `spd_solve` | **bit‑identical matrix into the same `potrf`/`potrs`** → ~1e‑12 | different algorithm (eigendecomp vs Cholesky) → needs its own tolerance argument |
| Code size / risk | small, additive; mirrors the eigen‑path vector pattern | larger; new syevd path + diag‑scale + two gemms |

The profile says **copies dominate (78%) and GPU Cholesky is sub‑ms**. (A) removes exactly the dominant cost (per‑λ H2D + mallocs) while keeping the cheap per‑λ `potrf`/`potrs`. (B) additionally removes the per‑λ `potrf` — but that is *already sub‑ms*, so (B) spends extra risk (eigendecomposition equivalence, a heavier per‑fold prep) to optimize a non‑bottleneck. (A) is the higher value/risk ratio. Keep (B) on the shelf for a future block only if a re‑profile ever shows per‑λ `potrf` rising to the top.

**Within (A), use variant A1** (one persistent device K + RHS *per fold*, all coexisting for the sweep), because the loop is λ‑outer/fold‑inner. A single shared device‑K reused across folds would force a re‑upload on every inner fold step (i.e., per λ) — that is the current behavior and would not remove the H2D. A1 mirrors `ridge_moment_eigen_paths` (a per‑fold prepared state held in a vector across the λ loop) and changes nothing about candidate scores.

---

## 3. Precise specification for Codex

### 3.1 New internal device API (`cuda_dispatch`)

Add to **`cpp/src/core/cuda_dispatch.hpp`** (after the `spd_solve` decl at `:115-120`):

```cpp
// Per-fold prepared dual-Ridge solver (B2). Holds the device-resident gram
// K (n x n, row-major symmetric) and the column-major RHS B (n x q) uploaded
// ONCE, so the whole lambda grid reuses one H2D. Opaque; defined in the .cpp.
struct PreparedDualRidge;

// Upload K and B once. Returns nullptr if no GPU is available or any device
// allocation/upload fails — the caller MUST treat nullptr as "use the host /
// per-call path" (no error is fatal here). K is n x n row-major symmetric
// (either triangle valid); B is n x q row-major.
PreparedDualRidge* prepare_dual_ridge(std::size_t n, std::size_t q,
                                      const double* K, const double* B,
                                      std::string* error);

// Solve (K + lambda*I) X = B for the stored K/B, writing X (n x q row-major).
// Forms A = K on device (D2D), adds lambda to A's diagonal, runs cuSOLVER
// Cholesky, reusing one thread-local work buffer. Return codes match spd_solve:
//   0: success
//   1: not positive definite (caller falls back to host QR)
//   2: CUDA / cuSOLVER / allocation failure
int dual_ridge_solve(PreparedDualRidge* handle, double lambda,
                     double* X, std::string* error);

// Release the device buffers. Safe on nullptr.
void destroy_dual_ridge(PreparedDualRidge* handle) noexcept;
```

Implementation in **`cpp/src/core/cuda_dispatch.cpp`** (place next to `spd_solve`, after `:1260`):

- **`struct PreparedDualRidge`** (named namespace, file scope): owns `ReusableDeviceBuffer<double> dK` (sized `n*n`) and `ReusableDeviceBuffer<double> dB0` (sized `n*q`, column‑major *pristine* RHS), plus `std::size_t n, q; int lwork;`. Reuse the existing `ReusableDeviceBuffer<T>` (`cuda_dispatch.cpp:148`) for RAII `cudaFree`.

- **`prepare_dual_ridge`**: mirror `spd_solve`'s guards (`:1166-1189` — `cuda_runtime_available`, null/zero checks, int‑range, `mul_overflows`). Then:
  1. `h->dK.ensure(n*n); copy_h2d(h->dK.get(), K, n*n*sizeof(double));` (full symmetric K, same as `spd_solve:1197`).
  2. Build column‑major `Bcm` exactly as `spd_solve:1199-1204` (`Bcm[target*n + i] = B[i*q + target]`), `h->dB0.ensure(n*q); copy_h2d(h->dB0.get(), Bcm.data(), ...)`.
  3. `cusolverDnDpotrf_bufferSize(state().solver, CUBLAS_FILL_MODE_LOWER, ni, h->dK.get(), ni, &lwork)` (depends only on n/uplo, so computing once in prepare is valid); store `h->lwork`.
  4. Return `h.release()`. **Any exception (incl. `std::bad_alloc`) → set `*error`, return `nullptr`** (no throw across the call; nullptr means host fallback).

- **`dual_ridge_solve`**: 
  1. Guards (`cuda_runtime_available`, `handle != nullptr`, `X != nullptr`), `ni=(int)h->n`, `qi=(int)h->q`.
  2. `thread_local ReusableDualWorkspace ws; ws.ensure(h->n, h->q, h->lwork);` — a new small class (mirror `ReusablePls1MomentWorkspace`, `:187`) holding `ReusableDeviceBuffer<double> dA(n*n), dB(n*q), dWork(lwork); ReusableDeviceBuffer<int> dInfo(1)`.
  3. `copy_d2d_contiguous(h->dK.get(), ws.dA(), h->n*h->n, "dual_ridge K->A");` (existing helper, `cuda_dispatch.cpp:495`; default null stream).
  4. `check_cuda(add_scaled_identity(ws.dA(), h->n, lambda, /*stream=*/nullptr), "add_scaled_identity");` (new kernel — §3.2).
  5. `copy_d2d_contiguous(h->dB0.get(), ws.dB(), h->n*h->q, "dual_ridge B0->B");` (must copy into a *work* RHS; `potrs` overwrites it — never touch the pristine `dB0`).
  6. `cusolverDnDpotrf(state().solver, CUBLAS_FILL_MODE_LOWER, ni, ws.dA(), ni, ws.dWork(), h->lwork, ws.dInfo())`; `copy_d2h(&info, ws.dInfo(), sizeof(int))`; `info>0 → return 1` (not PD → host QR), `info<0 → return 2`. (Same semantics/uplo as `spd_solve:1208-1228`.)
  7. `cusolverDnDpotrs(state().solver, CUBLAS_FILL_MODE_LOWER, ni, qi, ws.dA(), ni, ws.dB(), ni, ws.dInfo())`; check info `!=0 → 2`.
  8. `copy_d2h(Xcm, ws.dB(), ...)` then transpose column‑major → row‑major `X` exactly as `spd_solve:1240-1246`.
  9. `return 0`. Same try/catch envelope as `spd_solve:1248-1259`.

- **`destroy_dual_ridge`**: `delete handle;` (ReusableDeviceBuffer destructors free device memory).

### 3.2 Diagonal‑add kernel (`cuda_kernels`)

Recommend the tiny custom kernel over `cublasDaxpy` (no need to materialize a device vector of λ; the diagonal index is layout‑agnostic).

Add to **`cpp/src/core/cuda_kernels.hpp`** (after `:67`):

```cpp
// A += lambda on the diagonal: A[i*n + i] += lambda for i in [0, n).
// A is n x n. The diagonal index i*n+i is identical for row-major and
// column-major, so this matches the host K[i*n+i] += lambda bit-for-bit.
cudaError_t add_scaled_identity(double* A, std::size_t n, double lambda,
                                cudaStream_t stream) noexcept;
```

Add to **`cpp/src/core/cuda_kernels.cu`** (mirror the launcher pattern at `:195-215`, block size 256, grid `ceil(n/256)`):

```cpp
__global__ void add_scaled_identity_kernel(double* A, int n, double lambda) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + i] += lambda;
}

cudaError_t add_scaled_identity(double* A, std::size_t n, double lambda,
                                cudaStream_t stream) noexcept {
    if (n == 0) return cudaSuccess;
    if (A == nullptr ||
        n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }
    constexpr int kBlock = 256;
    const dim3 grid(static_cast<unsigned int>((n + kBlock - 1) / kBlock));
    add_scaled_identity_kernel<<<grid, dim3(kBlock), 0, stream>>>(
        A, static_cast<int>(n), lambda);
    return cudaGetLastError();
}
```

No CMake change: `cuda_kernels.cu` is already compiled into `n4m_core` (`cpp/src/CMakeLists.txt:85-88`).

### 3.3 sweep.cpp prepared‑state mirror + loop integration

Mirror the host eigen‑path vector. All additions guarded by `#if defined(N4M_USE_CUDA)`.

- **RAII wrapper** near the `RidgeMomentEigenPath` struct (`sweep.cpp:75`):

```cpp
#if defined(N4M_USE_CUDA)
struct PreparedDualGpu {
    ::n4m::cuda_dispatch::PreparedDualRidge* handle{nullptr};
    PreparedDualGpu() = default;
    ~PreparedDualGpu() { ::n4m::cuda_dispatch::destroy_dual_ridge(handle); }
    PreparedDualGpu(const PreparedDualGpu&) = delete;
    PreparedDualGpu& operator=(const PreparedDualGpu&) = delete;
    PreparedDualGpu(PreparedDualGpu&& o) noexcept : handle(o.handle) { o.handle = nullptr; }
    PreparedDualGpu& operator=(PreparedDualGpu&& o) noexcept {
        if (this != &o) { ::n4m::cuda_dispatch::destroy_dual_ridge(handle); handle = o.handle; o.handle = nullptr; }
        return *this;
    }
    bool valid() const noexcept { return handle != nullptr; }
};
#endif
```

- **One shared α‑solve helper** that the three consumers call, replacing their inline `K = design.K; K[i*n+i]+=λ; solve_dual_spd(...)` (`sweep.cpp:525-529`, `:573-577`, `:619-623`):

```cpp
[[nodiscard]] n4m_status_t solve_dual_alpha(Context& ctx,
                                            const RidgeDualDesign& design,
                                            double lambda,
                                            const PreparedDualGpu* gpu,
                                            std::vector<double>& alpha) {
    const std::size_t n = design.n_samples, q = design.n_targets;
#if defined(N4M_USE_CUDA)
    if (gpu != nullptr && gpu->valid()) {
        alpha.assign(n * q, 0.0);
        std::string err;
        const int s = ::n4m::cuda_dispatch::dual_ridge_solve(gpu->handle, lambda, alpha.data(), &err);
        if (s == 0) return N4M_OK;          // s==1 (not PD) / s==2 (runtime) → host fallback below
    }
#else
    (void)gpu;
#endif
    std::vector<double> K = design.K;
    for (std::size_t i = 0; i < n; ++i) K[i * n + i] += lambda;
    return solve_dual_spd(K, n, design.Y_work, q, alpha);   // unchanged B1 path (GPU per-call or host QR)
}
```

  Then the three consumers gain a trailing **defaulted** param `const PreparedDualGpu* gpu = nullptr` and replace their inline shift+solve with `solve_dual_alpha(ctx, design, lambda, gpu, alpha)`. (When `N4M_USE_CUDA` is off, `PreparedDualGpu` does not exist — declare the param as `const void* gpu = nullptr` is messy; instead guard the param itself: keep the three signatures CUDA‑agnostic by forward‑declaring `struct PreparedDualGpu;` unconditionally and only *defining* it under CUDA. The `gpu` pointer is only dereferenced inside the `#if defined(N4M_USE_CUDA)` block of `solve_dual_alpha`, so a forward declaration suffices for non‑CUDA builds.)

  Defaulted nullptr means the two final‑fit sites (`sweep.cpp:2553`, `:2657`) compile unchanged and keep today's behavior.

- **Prepared‑handle vector**, declared next to the eigen‑path vector (`sweep.cpp:2139`) and populated in the per‑fold prep loop (insert at the end of each fold iteration, after the optional cross build, ~`sweep.cpp:2042`):

```cpp
std::vector<PreparedDualGpu> fold_prepared_dual(static_cast<std::size_t>(actual_cv));   // entries default-null
...
// inside the prep loop, after fold_dual_designs[fold_ix] (and any cross) is built:
#if defined(N4M_USE_CUDA)
if (lambdas.size() > 1U && ridge_cuda_dual_enabled(fold_dual_designs[fold_ix].n_samples)) {
    std::string err;
    fold_prepared_dual[fold_ix].handle = ::n4m::cuda_dispatch::prepare_dual_ridge(
        fold_dual_designs[fold_ix].n_samples, q,
        fold_dual_designs[fold_ix].K.data(),
        fold_dual_designs[fold_ix].Y_work.data(), &err);   // nullptr on failure → host path
}
#endif
```

  Gating reuses **`ridge_cuda_dual_enabled`** (`sweep.cpp:435`), so the n≥256 threshold, runtime check, and `N4M_CUDA_RIDGE_DISABLE` all apply unchanged. `lambdas.size() > 1U` mirrors the eigen‑path amortization gate (`sweep.cpp:2141`).

- **Three call sites in the grid loop** pass the per‑fold handle:
  - `ridge_dual_cross_heldout_sse(ctx, fold_dual_designs[fold_ix], Yc, q, rows, lambda, fold_sse, &fold_prepared_dual[fold_ix])` (`sweep.cpp:2178`)
  - `predict_ridge_from_dual_design(ctx, fold_dual_designs[fold_ix], lambda, pred, &fold_prepared_dual[fold_ix])` (`sweep.cpp:2183`)
  - `fit_ridge_from_dual_design(ctx, fold_dual_designs[fold_ix], lambda, fit, &fold_prepared_dual[fold_ix])` (`sweep.cpp:2189`)

  For non‑CUDA builds these still compile because the param is a forward‑declared pointer; pass `&fold_prepared_dual[fold_ix]` only under `#if defined(N4M_USE_CUDA)`, else `nullptr` — or keep `fold_prepared_dual` defined only under CUDA and select the argument with a small `#if`. Either is fine; prefer one `#if` at the three call sites for clarity.

---

## 4. Correctness traps (explicit)

1. **No fold leakage by construction.** Each materialized‑dual fold owns its *own* persistent `PreparedDualRidge` (its own `dK`, `dB0`); there is no shared persistent K across folds. The single thread‑local `ReusableDualWorkspace` (`dA`, `dB`, `dWork`, `dInfo`) is *fully overwritten at the start of every solve*: `copy_d2d_contiguous(dK→dA, n*n)` writes all `n_f²` leading elements and `copy_d2d_contiguous(dB0→dB, n*q)` writes all `n_f·q` leading elements. After a larger fold, `ws.ensure` keeps the larger capacity, but `potrf`/`potrs` operate strictly on the leading `ni×ni` / `ni×qi` region with `ld = ni`, so stale tail bytes are never read. State this in a comment at the `ws.ensure` call.

2. **On‑device λ‑shift is bit‑identical to host `K[i*n+i] += λ`.** The off‑diagonal bytes of `dA` are a verbatim D2D copy of the host‑built `dK`; the diagonal element `A[i*n+i]` equals `K[i*n+i]` (same bytes) and the kernel performs a single IEEE‑754 round‑to‑nearest fp64 add of the same `λ` (plain `+`, no FMA contraction) — identical to the host. The matrix fed to `potrf` is therefore **bit‑for‑bit** the same matrix that current `spd_solve` would upload after the host formed `K+λI`. Since the same cuSOLVER `potrf`/`potrs` then run on the same GPU, `dual_ridge_solve` reproduces `spd_solve` to ~1e‑12 — far inside the 1e‑8 gate.

3. **Pristine RHS preserved.** `potrs` overwrites its RHS; never pass `dB0`. Always solve in the work `dB` (copied from `dB0` each call). One mistaken `potrs` on `dB0` would silently corrupt every later λ for that fold.

4. **Serial‑handle contract.** `dual_ridge_solve` uses the process‑wide singleton `state().solver`/`state().handle` (`cuda_dispatch.cpp:82`, `:47-50`) and the default (null) stream, which serializes the D2D copy → diag kernel → `potrf` → `potrs` in issue order; the `copy_d2h(info)` forces a sync. The sweep λ/fold loop is single‑threaded and serial (same contract `solve_dual_spd` documents at `sweep.cpp:458-459`), so the `thread_local` shared workspace is never reentered concurrently. Do **not** introduce streams here.

5. **Fallback paths.** `prepare_dual_ridge` returning `nullptr` (no GPU, disabled, or alloc failure) ⇒ `gpu->valid()` is false ⇒ `solve_dual_alpha` runs the unchanged B1 `solve_dual_spd` path. `dual_ridge_solve` returning `1` (not PD) or `2` (runtime) ⇒ same fallback. `N4M_CUDA_RIDGE_DISABLE=1` ⇒ `ridge_cuda_dual_enabled` false ⇒ no handle prepared ⇒ full host path, *and* the fallback `solve_dual_spd` also sees it disabled ⇒ host QR. End‑to‑end host behavior is byte‑for‑byte the pre‑B2 path.

6. **ABI / surface.** All new symbols live in `n4m::cuda_dispatch` (internal, visibility‑hidden) and `sweep.cpp`'s anonymous namespace — no `n4m_*` export, no header in `cpp/include/n4m/`, no `.def`/`.map` change. ABI stays 1.22.0; `cpp/abi/expected_symbols_*.txt` unchanged.

7. **Device‑memory footprint.** Resident ≈ Σ_folds(`n_f²` + `n_f·q`) + max‑fold work buffers (`n_f²` + `n_f·q` + `lwork`). BERRY ≈ 10 folds × 16 MB ≈ ~180 MB — acceptable. The `bad_alloc`→`nullptr` contract is the safety net: if a fold's prepare OOMs, that fold silently falls back to per‑call `spd_solve`. No hard budget needed; note this in the prepare comment.

---

## 5. File‑by‑file roadmap + green gates

**Edits (in order):**

1. `cpp/src/core/cuda_kernels.hpp` — declare `add_scaled_identity` (after `:67`).
2. `cpp/src/core/cuda_kernels.cu` — define `add_scaled_identity_kernel` + launcher (mirror `:195-215`).
3. `cpp/src/core/cuda_dispatch.hpp` — declare `struct PreparedDualRidge;`, `prepare_dual_ridge`, `dual_ridge_solve`, `destroy_dual_ridge` (after `:120`).
4. `cpp/src/core/cuda_dispatch.cpp` — define `PreparedDualRidge`, `ReusableDualWorkspace`, and the three functions (after `:1260`); reuse `ReusableDeviceBuffer` (`:148`), `copy_h2d`/`copy_d2h` (`:303`/`:312`), `copy_d2d_contiguous` (`:495`), `check_cusolver`/`check_cuda` (`:352`/`:358`), `state()` (`:82`).
5. `cpp/src/core/sweep.cpp` —
   - forward‑declare `struct PreparedDualGpu;` unconditionally; define it under `#if defined(N4M_USE_CUDA)` near `:75`;
   - add `solve_dual_alpha` helper; route the three consumers through it; add the defaulted `gpu` param to `predict_ridge_from_dual_design` (`:505`), `ridge_dual_cross_heldout_sse` (`:549`), `fit_ridge_from_dual_design` (`:607`);
   - declare `fold_prepared_dual` near `:2139`; populate it in the prep loop at `~:2042`; pass `&fold_prepared_dual[fold_ix]` at the three grid call sites (`:2178`, `:2183`, `:2189`).
6. `cpp/tests/test_internal_linalg.cpp` — add `test_cuda_prepared_dual_ridge_matches_spd_solve()` (mirror `:305`) and register it in `run_internal_linalg_tests` (`:376-384`).

**No CMake change** — CUDA sources already wired (`cpp/src/CMakeLists.txt:85-88`, `:219-221`; link at `n4m_targets.cmake:146-149`).

**The new test** (the precise numeric gate for trap #2):
- Skip if `!cuda_runtime_available()`.
- Build SPD gram `K` (n=64, q=3) as in the existing test (`:315-334`) but **without** the `+1.0` diagonal bump (the prepared path adds λ on device).
- For `λ ∈ {1e-3, 0.1, 1.0, 10.0}`: reference `X_ref` via `spd_solve(n,q,Kλ,B,X_ref)` where `Kλ = K; Kλ[i*n+i]+=λ` (host‑formed, the B1 path). Prepared: `h = prepare_dual_ridge(n,q,K,B); dual_ridge_solve(h,λ,X_dev)`. Assert `max|X_ref−X_dev| ≤ 1e-8` (expect ~1e‑12).
- Not‑PD: a `K` whose `K+0·I` is singular (reuse the rank‑deficient diagonal matrix at `:362-365`) → `dual_ridge_solve(h, 0.0, …) == 1`.
- `destroy_dual_ridge(h)`.

**Green gates (run all, in order):**

```bash
# CUDA build + unit + internal tests (the precise equivalence gate)
cmake --preset blas-omp -DN4M_WITH_CUDA=ON           # or the repo's cuda preset
cmake --build --preset blas-omp -j
ctest --preset blas-omp --output-on-failure
./build/blas-omp/cpp/tests/n4m_tests --test-case="*"           # doctest
./build/blas-omp/cpp/tests/n4m_internal_tests                  # runs test_cuda_prepared_dual_ridge_matches_spd_solve

# dev-release (non-CUDA) build must still compile/link (forward-declared handle, nullptr path)
cmake --preset dev-release && cmake --build --preset dev-release --parallel
ctest --preset dev-release --output-on-failure

# catalog / ABI (must be unchanged — internal only)
python catalog/scripts/validate.py --check-references --strict-abi
python catalog/scripts/reconcile_abi.py --check
nm -D --defined-only build/blas-omp/cpp/src/libn4m.so.<ver> | awk '{print $3}' | sort -u \
  | diff -u cpp/abi/expected_symbols_linux.txt -      # expect empty diff
git diff --check

# run_moment_sweep parity: host vs B2-GPU candidate scores ≤ 1e-8
#   BERRY screen (the B1 benchmark dataset), score parity + timing:
N4M_CUDA_RIDGE_DISABLE=1 <berry-screen-cmd>  --out-csv /tmp/b2_host.csv
                          <berry-screen-cmd>  --out-csv /tmp/b2_gpu.csv
#   assert candidate_scores / RMSEP identical to ≤1e-8 (same check B1 used: "RMSEP preserved"),
#   and confirm the cudaMemcpy/cudaMalloc collapse via nsys on the BERRY screen.
```

Expected outcome: BERRY screen drops from the post‑B1 ~8.27 s by eliminating the 1.77 s `cudaMemcpy` (per‑λ K H2D) and the ~0.4 s malloc/free churn (1232 mallocs), with candidate scores bit‑equivalent to the host path within 1e‑8.
