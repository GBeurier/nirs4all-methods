# Phase 45 — Optional CUDA / cuBLAS Backend

**Status**: shipped 2026-05-16 · tag `phase-45-cuda-backend` ·
release `0.96.0+abi.1.13.0`. Default is **OFF**.

## Goal

Add `PLS4ALL_WITH_CUDA=ON` as a build option that routes the
`linalg::*` dispatch through host-side **cuBLAS** on an NVIDIA GPU.
Numerical parity with the cross-binding fixture is preserved. The
OFF build stays bit-identical to pre-Phase-45; public ABI stays at
1.13.0.

## Why host-side cuBLAS (no kernels)

cuBLAS exposes `cublasDgemv_v2`, `cublasDgemm_v2`, `cublasDger_v2`
as host-callable functions. Calling them from `.cpp` means:

* No `.cu` files. No `nvcc` in the build path. No
  `enable_language(CUDA)`. The host C++ compiler builds
  `cuda_dispatch.cpp` with `-I${CUDAToolkit_INCLUDE_DIRS}` and links
  `CUDA::cudart CUDA::cublas`.
* The same dispatch surface as Phase 43 BLAS — same three entry
  points (`gemv`, `gemm`, `ger`), same row-major contract. cuBLAS is
  column-major; we apply the standard row→col-major swap trick at
  the dispatch boundary.

Custom CUDA kernels are deferred to Phase 45c (when profiling shows
a kernel libp4a should own end-to-end, not just delegate to cuBLAS).

## Architecture

```
cpp/src/core/
├── cuda_dispatch.hpp   (new) — public-internal signatures
├── cuda_dispatch.cpp   (new) — Meyers-singleton handle,
│                       DevicePtr<T> RAII, cuBLAS calls
├── linalg.hpp          extended: CUDA branch is first #if,
│                       BLAS now #elif, scalar fallback
└── model.cpp           unchanged — calls go through linalg::*
```

### Singleton + RAII pattern

```cpp
struct CublasState {
    cublasHandle_t handle{};
    bool available{false};

    CublasState() noexcept {
        // cudaGetDeviceCount + cudaSetDevice(0) + cublasCreate_v2.
        // If any step fails, available stays false; subsequent
        // p4a_backend_is_available(P4A_BACKEND_CUDA) returns 0.
    }
    ~CublasState() { if (available) cublasDestroy_v2(handle); }
};
```

The state is a function-local static (Meyers singleton). Lazy
initialisation; C++11 thread-safe. `DevicePtr<T>` is a single-owner
RAII wrapper around `cudaMalloc`/`cudaFree`, used for every per-call
device buffer.

### Row-major → column-major math (the bit Codex caught)

For `C_row = op_row(A) * op_row(B)` we compute the column-major
view of the same memory:

```
(op_row(A) * op_row(B))^T  =  (op_row(B))^T * (op_row(A))^T
```

And `(op_row(M))^T = op_cm(M^cm)` with **op_cm == op_row** (no
flip). So we pass:

* `B` as the first cuBLAS matrix with `op_cm_b = op_row_b`,
* `A` as the second with `op_cm_a = op_row_a`,
* dimensions `m_cm=N_row, n_cm=M_row, k_cm=K_row`,
* leading dims unchanged (the row-major lda is the col-major lda of
  the same buffer).

A first attempt that flipped `op_cm` caused
`cublasDgemm parameter 8 had an illegal value` on the (200×20)×(200×2)
SIMPLS smoke. The fix landed before commit.

## C ABI behaviour

`p4a_backend_is_available(P4A_BACKEND_CUDA)` now reports
**compile-time AND runtime** state — it returns 1 only when the
build was configured with `PLS4ALL_WITH_CUDA=ON` **and** a
CUDA-capable GPU is present and `cublasCreate_v2` succeeded. This
is the first backend whose availability is a runtime probe; the
side-effect (CUDA context init) is documented and memoised in the
singleton.

`p4a_context_set_backend(ctx, P4A_BACKEND_CUDA)` accepts the
backend when both compile and runtime checks pass, returns
`P4A_ERR_BACKEND_UNAVAILABLE` otherwise with a clear error message.

## Live ship-gate

Host: RTX 4090 + RTX 5090, CUDA 12.8, OpenBLAS as fallback.

| Config | Tests | CUDA tests | Cross-binding |
|--------|-------|-----------|---------------|
| OFF/OFF | 265/265 | 3 skipped via `#if defined(P4A_USE_CUDA)` | 10/10 |
| CUDA-ON | 265/265 | 3 run live on GPU 0 | 10/10 |

The CUDA-ON parity gate regenerated the shared fixture against the
cuBLAS path; the 10 bindings then re-fit against that fixture
through their normal `p4a_pls_fit_simple` call which now routes
through the GPU. All 10 returned `PARITY GATE PASS` (the gate is
`rmse_rel < 1e-12`).

CLI: `pls4all_cli --selfcheck` returns OK; `set_backend(CUDA)`
succeeds on this host.

ABI symbol count unchanged at 160. Public surface unchanged at
1.13.0.

## Codex review findings

The Codex reviewer confirmed:

* `gemm` row↔col-major math is correct (verified by re-deriving
  `(op(A)*op(B))^T = op(B)^T * op(A)^T` from scratch).
* `gemv` and `ger` ops are correct.
* `DevicePtr<T>` + stack-unwind exception safety covers leak paths.
* `cuda_runtime_available()` is correctly `noexcept`; CUDA C API
  doesn't throw.
* OBJECT-library link propagation matches the Phase 43/44 pattern.
* `P4A_BACKEND_OPENCL` substitution in `test_error_messages.cpp` is
  valid — OPENCL has no compile-time enablement path anywhere.

One latent footgun documented: `gemm` / `ger` assume **contiguous**
row-major buffers (`lda == cols`). All internal call sites honour
this; the header documents the contract and Phase 45b will extend
the dispatch surface to accept strided inputs.

## Phase 45b backlog

* **Pinned host memory** (`cudaMallocHost`) to amortise H2D / D2H
  transfer cost — especially for the inner NIPALS `X^T y_score`
  loop where the same `xk` buffer is touched repeatedly.
* **Streams + async copies** to overlap GPU work with the next
  iteration's host setup.
* **Strided buffer support** in `cuda_dispatch::gemm/ger` for the
  rare `lda > cols` case (none in libp4a today; future MB-PLS
  variants might want it).
* **Persistent model device state**: once a model is trained, its
  `coefficients` matrix could live on the GPU for prediction-heavy
  workloads. This requires extending the model storage layer, not
  just the dispatch.

## Release

* Library version: `0.95.0` → `0.96.0`. ABI version unchanged at
  `1.13.0`. Additive: CUDA is opt-in.
* Tag: `phase-45-cuda-backend`.
