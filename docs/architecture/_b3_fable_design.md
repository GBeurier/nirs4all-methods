I now have a complete, source-grounded picture. Here is the roadmap.

---

# B3 Implementation Roadmap — Kill the per‑candidate moment/gram H2D in the materialized screen path

**Design only. Codex implements strictly from this. Internal `cuda_dispatch`/`core` only; ABI 1.22.0; fp64; host fallback kept.**

---

## 0. Executive summary & headline recommendation

The post‑B2 profile names `cudaMemcpy` (1.66 s, 1240 calls, 83.5 % of CUDA API on the 7.73 s BERRY screen) as the #1 CUDA cost, attributed to "moment build + per‑candidate operator→moment transform." **Reading the source overturns half of that attribution**, exactly the way the 5a design doc overturned its own first draft:

- **The operator→moment transform (`transform_moments_by_operator`, banded, structured) is dead code for the measured datasets.** It is gated off for `p > 1024` (`kMaxBandedOperatorMomentFeatures`) and `p > 48` (`kMaxWideOperatorMomentFeatures`) — `aom_sweep.cpp:28-29`. BERRY (p=2101), COLZA, and LUCAS (p=4200) all exceed both caps, so `allow_dense` and `allow_structured` are both **false** (`aom_sweep.cpp:2055-2056`, `:2224-2225`, `:2065`, `:2231`) and every chain is **materialized** and pushed through `run_moment_sweep` (`aom_sweep.cpp:2283-2289`, `:2327-2335`).
- **The real 1240 copies come from the *materialized* per‑candidate build**: inside `run_moment_sweep`, each chain rebuilds (a) the dual gram `K = Xw·Xwᵀ` (`sweep.cpp:389-394`) per fold + all‑sample, and (b) the moments `XᵀX/XᵀY/YᵀY` (`moments.cpp:190-201`) for the all‑sample set and each held‑out fold. Every one of those `linalg::gemm` calls dispatches to `cuda_dispatch::gemm` (`common/linalg.hpp:173-178`), which does a fresh `cudaMalloc` + H2D(A) + H2D(B) + `cublasDgemm` + D2H(C) + `cudaFree` **per call** (`cuda_dispatch.cpp:1728-1786`). The same materialized design matrix `Xt` (n×p) is re‑uploaded ~6–10× per candidate.

**Recommendation: a targeted (C) — fused single‑upload device builds in `cuda_dispatch`** that replace the multi‑GEMM, host‑orchestrated build of the moments and the gram, uploading each per‑candidate matrix **once** and reusing grow‑only device scratch.

- **Reject (A) literal** (thread‑local scratch only): already tried in B1, reverted, 1.00× — `cudaMalloc`+`cudaFree` was 14 ms total (`fused_moment_executor_design.md:37`, `:40-41`). It kills allocation, not the H2D/D2H. *Fold its zero‑cost malloc‑elimination into B3 via reused scratch, but do not ship it as the win.*
- **Reject (B) literal** (device‑resident base moments + on‑device operator transform): the operator‑transform route it optimizes never runs on the measured workloads (gated off, above). It would not touch the measured 1.66 s. Preserve those routes untouched and do not force a dense p×p GEMM there.
- **Ship (C) in two self‑contained increments** (B3‑1 then B3‑2), and **defer** the larger device‑resident‑consumer change (which is the only thing that removes the dominant p×p **D2H**) to B4 with the reasons stated in §6.

---

## 1. Task 1 — Trace and quantification of the 1240 `cudaMemcpy`

### 1.1 Dispatch mechanics (the per‑call round‑trip)

Every `linalg::gemm` in the core routes to cuBLAS in CUDA builds (`common/linalg.hpp:173-178`). The wrapper `cuda_dispatch::gemm` (`cuda_dispatch.cpp:1728-1786`):

- allocates three fresh `DevicePtr` (`:1752-1754`) → `cudaMalloc`/`cudaFree` per call (`:93-104`),
- `copy_h2d(dA)`, `copy_h2d(dB)`, and `copy_h2d(dC)` if `beta≠0` (`:1756-1759`),
- one `cublasDgemm_v2` (`:1772-1783`),
- `copy_d2h(C)` (`:1785`).

`copy_h2d`/`copy_d2h` are blocking `cudaMemcpy` (`:325-341`) — i.e. exactly the synchronous copies the profile counts. Crucially, when `A == B` (a symmetric product), the wrapper still uploads the operand **twice**.

### 1.2 Where the per‑candidate GEMMs live (the materialized path BERRY actually takes)

`run_moment_sweep` (`sweep.cpp`) is invoked once per head per chain. Per invocation it builds, for a wide problem (`p>n`, true for BERRY 2101>1434):

| Build | Site | GEMM shape | H2D per call | D2H per call |
|---|---|---|---|---|
| Dual gram `K = Xw·Xwᵀ` (all‑sample, `p>n`) | `sweep.cpp:2111-2118` → `:389-394` | (n,n,p), A=B=Xw | **Xw n×p ×2** | K n×n |
| Dual gram `K` per materialized fold | `sweep.cpp:2075-2108` → `:389-394` | (nt,nt,p) | Xtrain ×2 | K |
| Dual cross gram `K_cross` (when elected) | `sweep.cpp:2090-2092` → `:436-440` | (h,n,p) | Xheld + Xw | K_cross |
| Moments `XᵀX` (all‑sample, PLS route) | `sweep.cpp:2120-2124` → `moments.cpp:190-193` | (p,p,n), A=B=X | **X n×p ×2** | **XᵀX p×p** |
| Moments `XᵀY` (all‑sample) | `moments.cpp:194-197` | (p,q,n) | X + Y | XᵀY p×q |
| Moments `YᵀY` (all‑sample) | `moments.cpp:198-201` | (q,q,n) | Y ×2 | YᵀY q×q |
| Moments per held‑out fold (×cv) | `sweep.cpp:2135-2138` → `moments.cpp:273-296` | as above on subset | X subset, Y subset | **XᵀX p×p ×cv** |

So the dominant copy **volume** is:
- **The redundant H2D of the materialized design matrix.** Within one `compute_moments` the matrix `X` is uploaded 3× (twice for `XᵀX`, once for `XᵀY`); within one gram build `Xw` is uploaded 2×. The *same* `Xt` is also re‑uploaded across the K‑build, the moment build, and the PLS scorer. This is the "base moments re‑uploaded per candidate" the orchestrator measured. (BERRY n×p ≈ 24 MB; LUCAS ≈ 70 MB.)
- **The unavoidable‑for‑now D2H of the p×p moments.** `XᵀX` D2H is 35 MB (BERRY) / 141 MB (LUCAS), once for the all‑sample set and once per held‑out fold (`recompute_centered_moments`/`subtract_moments` consume it on the **host**, `moments.cpp:207-255`, `:299-350`).

The **third** per‑candidate p×p mover the orchestrator asked about is real but separate: the PLS scorer `pls1_moment_components_many_batched_tiled` re‑uploads the centered moment `C` (p×p) per tile (`cuda_dispatch.cpp:870-878`). That is a *consumer‑side* H2D, addressed only by the deferred device‑pointer scorer entry (§6).

### 1.3 Determination (answering the orchestrator's question)

> *"Which GEMM calls dominate — the per‑candidate transform? the gram K build? the PLS scorer tiles?"*

**The per‑candidate operator→moment transform does not dominate — it does not run** (gated off at p>1024). The dominant copies are the **materialized moment build (`moments.cpp:190-201`, most numerous — 1+cv calls/candidate, each with a p×p D2H) and the dual gram `K` build (`sweep.cpp:389-394`, largest single H2D — the full n×p design uploaded twice)**. The PLS scorer's per‑tile `C` upload (`cuda_dispatch.cpp:877`) is a secondary contributor. **What is re‑uploaded redundantly: the materialized n×p design matrix (uploaded 2–3× per build call and again across build/gram/scorer), not the base XᵀX.**

---

## 2. Task 2 — Design choice and value/risk, anchored to the profile

| Option | What it removes | Fraction of the 1.66 s | Risk | Verdict |
|---|---|---|---|---|
| **(A)** scratch‑only in `gemm` | per‑call malloc/free (14 ms total) | ~1 % | trivial | **Reject as the win** (already tried → 1.00×); fold the free malloc‑elimination into B3. |
| **(B) literal** device base moments + on‑device operator transform | copies on the operator‑moment route | **~0 %** on BERRY/COLZA/LUCAS — route is gated off (`aom_sweep.cpp:28-29`, `:2055`, `:2224`) | high (touches banded/structured) | **Reject for the measured workload.** Preserve those routes; do not force dense p×p GEMM. |
| **(C) targeted: fused single‑upload device builds** for the *materialized* moment & gram builds | the redundant n×p H2D (3×→1× X for moments, 2×→1× for gram) + per‑call malloc | the **H2D half** of the per‑candidate cost — the "re‑uploaded per candidate" fraction the orchestrator measured | low; numerically identical to today's cuBLAS path | **Recommended.** |
| (D, deferred → B4) device‑resident moment/gram *chain* through the consumers | additionally the dominant p×p **D2H** + scorer `C` H2D | the larger remaining fraction | high (consumer rewrite + device kernels for subtract/center, device‑pointer scorer) | **Defer**, see §6. |

**Why (C) and not the bigger (D) first:** (D) is where the p×p D2H (35–141 MB/candidate) dies, but it requires device‑side `subtract_moments`/`recompute_centered_moments` kernels, a device‑pointer entry to the PLS scorer, and routing the host eigen‑path/ridge consumers off host buffers — a consumer‑touching change that risks fold‑isolation and the `≤1e-8` contract. (C) is the **smallest increment that removes the largest *measured* fraction** (the re‑upload), is numerically identical to the current cuBLAS path (same `cublasDgemm`, just one shared upload), changes **no consumer**, and lays the device‑build foundation (D) extends. This mirrors the B2 pattern: B2 uploaded `K` once for the *solve*; B3 uploads the design once for the *build*.

---

## 3. Task 3 — Precise specification for Codex

### 3.1 New device API (lives in `cuda_dispatch`, internal namespace, **not** `N4M_API`)

Add to `cpp/src/core/cuda_dispatch.hpp` (mirroring the `spd_solve`/`prepare_dual_ridge` style, `cuda_dispatch.hpp:109-137`) and implement in `cpp/src/core/cuda_dispatch.cpp`:

```cpp
// B3-1: fused moment build. Uploads X (n x p) and Y (n x q) row-major ONCE,
// computes XtX (p x p), XtY (p x q), YtY (q x q) on device with cuBLAS (fp64),
// reusing thread-local grow-only buffers. Results written row-major to caller
// host buffers (sized p*p / p*q / q*q). yty may be nullptr to skip YtY.
// Returns 0 ok / 2 (no GPU | runtime | overflow). Caller falls back to the
// existing 3x linalg::gemm on any non-zero return.
int build_moments_device(std::size_t n, std::size_t p, std::size_t q,
                         const double* X, const double* Y,
                         double* XtX, double* XtY, double* YtY,
                         std::string* error);

// B3-2: fused gram build. Uploads Xw (n x p) ONCE, computes K = Xw*Xw^T (n x n)
// on device (cublasDgemm with the single resident operand; symmetric).
// Returns 0 ok / 2. Caller falls back to linalg::gemm on non-zero.
int build_gram_device(std::size_t n, std::size_t p,
                      const double* Xw, double* K, std::string* error);
```

Design notes binding Codex:
- **Single upload.** Upload `X`/`Xw`/`Y` to grow‑only `ReusableDeviceBuffer<double>` thread‑locals (`cuda_dispatch.cpp:149-186`), reused across calls — this is option (A)'s malloc‑elimination, free. The symmetric products (`XtX`, `K`) pass the *same* device pointer as both cuBLAS operands (no second upload), unlike the generic `gemm` wrapper.
- **Numerics = today's CUDA path.** Use `cublasDgemm_v2` with the identical row‑major↔column‑major mapping the wrapper already uses (`cuda_dispatch.cpp:1762-1783`): `XᵀX` ≡ `gemm(Trans_Yes,Trans_No,p,p,n,X,p,X,p)`, `XᵀY` ≡ `gemm(Trans_Yes,Trans_No,p,q,n,X,p,Y,q)`, `YᵀY` likewise, `K` ≡ `gemm(Trans_No,Trans_Yes,n,n,p,Xw,p,Xw,p)`. Because today's build already runs on cuBLAS, the fused build is **bit‑equivalent to the current CUDA path** and differs from the host path only by the existing, already‑accepted cuBLAS‑vs‑host delta. **Do not** substitute `dsyrk` in B3‑1 (it changes which triangle/rounding is produced and would need symmetrization); keep `dgemm` to make equivalence trivial. (`dsyrk` is a candidate optimization for a later increment, behind its own equivalence check.)
- **Overflow/int‑range guards** identical to `spd_solve` (`cuda_dispatch.cpp:1196-1211`): reject `p`/`n` beyond `INT_MAX`, reject element/byte overflow → return 2.
- **No persistent cross‑call cache, no `Context` state.** Buffers are thread‑local inside `cuda_dispatch` and strictly internal to the call — preserves "no ABI change" and the device‑state contract (`fused_moment_executor_design.md:142-145`).
- **Device selection.** Inherit the existing `CublasState` singleton (`cuda_dispatch.cpp:48-86`); do not add new device selection. (The hardcoded‑device‑0 caveat at `:54-56` is pre‑existing and out of B3 scope.)

### 3.2 Integration points (host call sites)

**B3‑1 — `compute_from_contiguous`, `cpp/src/core/moments.cpp:190-201`.** Replace the three `linalg::gemm` calls with:

```cpp
bool built_on_device = false;
#if defined(N4M_USE_CUDA)
if (n4m::cuda_dispatch::cuda_runtime_available()) {
    std::string err;
    if (n4m::cuda_dispatch::build_moments_device(
            nn, pp, qq, X.data(), Y.data(),
            out.xtx.data(), out.xty.data(), out.yty.data(), &err) == 0) {
        built_on_device = true;
    }
}
#endif
if (!built_on_device) {
    // existing three linalg::gemm calls verbatim (host / BLAS / CUDA-wrapper fallback)
}
return recompute_centered_moments(ctx, out);
```

This covers **both** `compute_moments` (`moments.cpp:257-271`) and `compute_moments_subset` (`moments.cpp:273-297`), since both funnel through `compute_from_contiguous`. No change to `MomentStats`, centering, or any consumer.

**B3‑2 — the gram builds in `sweep.cpp`.** Replace the `linalg::gemm` at `sweep.cpp:389-394` (inside `prepare_ridge_dual_design`) with a guarded `build_gram_device(rows, p, X_work, K)` + the existing `gemm` as fallback. The cross‑gram at `sweep.cpp:436-440` (`prepare_ridge_dual_cross`) is an A≠B product; it benefits only from the malloc/single‑upload of each operand — optional, lower priority, same pattern. **Leave `prepare_dual_ridge`'s host‑`K` upload (`cuda_dispatch.cpp:1331`) untouched in B3** (the K→prepare device hand‑off is deferred — §6 — because `K` is still needed on host for the single‑λ `solve_dual_spd` fallback and for `K_cross` prediction).

### 3.3 Gating

- Compile‑time: all new code under `#if defined(N4M_USE_CUDA)`; the host build never sees it (matches `sweep.cpp:486-497`, `:2071-2107`).
- Runtime: call the device build only when `cuda_dispatch::cuda_runtime_available()` (`cuda_dispatch.hpp:26`). Add **no new size threshold** for B3‑1 (the moment build is always large in this path); for B3‑2 reuse the existing `ridge_cuda_dual_enabled` notion only if a size floor is wanted, but the gram is only built when already on the dual path, so a plain availability check suffices.
- Keep the `N4M_CUDA_RIDGE_DISABLE` kill‑switch semantics intact for the ridge path (`sweep.cpp:469-472`); B3‑1's moment build is independent of it.

### 3.4 Host fallback

Any non‑zero return (`2` = no GPU/runtime/overflow) falls straight through to the unchanged `linalg::gemm` calls — which themselves dispatch to host scalar, BLAS, or the existing CUDA wrapper depending on the build. The non‑CUDA build compiles unchanged (the device branch is `#if`‑compiled out). This matches the B1/B2 fallback discipline exactly (`sweep.cpp:494-498`).

### 3.5 fp64 equivalence + test

Add a CUDA‑guarded unit test to `cpp/tests/test_internal_linalg.cpp` (pattern: `:304-370`, registered at `:475-476`), self‑skipping when `!cuda_runtime_available()`:

- **`test_cuda_build_moments_matches_host`** — synthetic wide problem (e.g. n=96, p=160, q=3 to exercise p>n and q>1), fill `X`/`Y` deterministically (as `:315-343`), compute `XᵀX/XᵀY/YᵀY` via the host reference (`linalg::gemm` or an explicit triple loop) and via `build_moments_device`, assert max‑abs diff `≤1e-8` (constant `kSpdTol` style, `:313`). Add a `yty=nullptr` path assertion.
- **`test_cuda_build_gram_matches_host`** — same for `K = Xw·Xwᵀ`, `≤1e-8`.

These are the per‑function equivalence gate. The end‑to‑end candidate‑score parity is the integration gate (§5).

### 3.6 ABI status

`build_moments_device`/`build_gram_device` live in `namespace n4m::cuda_dispatch`, hidden visibility, **not** decorated `N4M_API` — identical treatment to `spd_solve`/`prepare_dual_ridge`, which the worklog confirms stay **absent from the dynamic symbol table** (`aom_moment_worklog.md:32`, `:70`). No public header, no catalog row, no Python change. **ABI stays 1.22.0**; `expected_symbols_*.txt` and `reconcile_abi.py` are unaffected (verify: see §5 gate).

---

## 4. Task 4 — Correctness traps

1. **Fold isolation is preserved structurally.** B3‑1 is a drop‑in inside `compute_from_contiguous`, called once per fold's matrix (`compute_moments_subset`, `sweep.cpp:2135-2138`); B3‑2 is a drop‑in inside the per‑fold `prepare_ridge_dual_design` loop (`sweep.cpp:2076-2108`). Each call sees exactly one fold's data — the device build hoists **nothing** across folds. **Mandate:** the thread‑local scratch must be *overwritten* every call (no accumulation, no reuse of stale rows beyond the leading `n×p`/`p×p` region), exactly as `ReusableDualWorkspace` documents (`cuda_dispatch.cpp:1389-1392`).
2. **Stateful heads (MSC/EMSC re‑fit per fold).** These are applied during *materialization* (`transform_chain`, upstream of `run_moment_sweep`, e.g. `aom_sweep.cpp:2243-2253`). B3 operates strictly **after** materialization, on the already‑per‑fold matrix, so it cannot break per‑fold re‑fit. Do not move any build above the materialization boundary.
3. **P‑dependent detrend projector (and the "Q" invariant).** The detrend projector/inverse is rebuilt per P/fold in the **structured** route (`build_detrend_projection_basis` `aom_sweep.cpp:759-803`; `apply_detrend_projection_*` `:805-878`; `transform_moments_by_detrend_operator` `:883-930`). B3 **does not touch the structured/banded routes at all** — they keep using host `linalg::gemm`/banded solvers — so the projector invariant is untouched. (Note: there is no `attach_Q` symbol in the C++ core; treat it as this projector/structured‑route invariant, which B3's scope does not intersect.)
4. **Numerical equivalence vs host `linalg::gemm` (≤1e-8).** Guaranteed because the device build issues the *same* `cublasDgemm` the current CUDA wrapper already issues; the only change is sharing one upload. Verified by §3.5 unit tests and §5 candidate‑score parity. **Do not** swap in `dsyrk`/mixed precision in B3.
5. **Device memory budget.** B3‑1/B3‑2 hold at most one design matrix (n×p) + its products (p×p / n×n) per call — *less* than the current wrapper's three separate buffers, and far under the per‑job tile budget (`choose_pls_batch_tile_jobs`, `cuda_dispatch.cpp:480-506`). At LUCAS p=4200 a p×p fp64 is 141 MB (`fused_moment_executor_design.md:138`); B3 does not multiply this (it replaces, not adds, device buffers) and must not run concurrently with the scorer's tile budget on the same call (it doesn't — builds precede scoring). Reuse `cudaMemGetInfo`‑style guarding only if Codex adds a size floor; otherwise the grow‑only buffer's `bad_alloc`→return‑2→host‑fallback path (`cuda_dispatch.cpp:170-172`) is the budget backstop.
6. **Do not break the PLS moment scorer.** The scorer still consumes **host‑built** centered moments and re‑uploads `C` per tile (`cuda_dispatch.cpp:870-878`). B3 leaves the scorer entirely unchanged — `recompute_centered_moments` still runs on host (`moments.cpp:207-255`) and produces the same `cxx`/`cxy` the scorer expects. (Removing the scorer's `C` H2D is the deferred device‑pointer entry, §6.)

---

## 5. Task 5 — File‑by‑file roadmap, green gates, deferrals

### B3‑1 (ship first — fused moment build)
1. `cpp/src/core/cuda_dispatch.hpp` — declare `build_moments_device` (near `:109-137`).
2. `cpp/src/core/cuda_dispatch.cpp` — implement it: thread‑local grow‑only `ReusableDeviceBuffer` for `dX`,`dY`,`dXtX`,`dXtY`,`dYtY`; single H2D of X,Y; three `cublasDgemm_v2`; D2H of the three products; full try/catch→return 2 (pattern of `spd_solve`, `:1178-1282`).
3. `cpp/src/core/moments.cpp` — guarded device build + verbatim host fallback in `compute_from_contiguous` (`:190-201`).
4. `cpp/tests/test_internal_linalg.cpp` — `test_cuda_build_moments_matches_host`, register at `:475-476`.

### B3‑2 (ship next, still B3 — fused gram build)
5. `cpp/src/core/cuda_dispatch.{hpp,cpp}` — declare/implement `build_gram_device` (single upload, `dgemm` symmetric).
6. `cpp/src/core/sweep.cpp` — guarded swap at `:389-394` (and optionally `:436-440`), host `gemm` fallback.
7. `cpp/tests/test_internal_linalg.cpp` — `test_cuda_build_gram_matches_host`.

No CMake change needed: `cuda_dispatch.cpp` is already compiled into `n4m_core`/`n4m_c`, which already links `CUDA::cudart CUDA::cublas CUDA::cusolver` (`n4m_targets.cmake:146-148`); the internal test links `n4m_c_static` (`cpp/tests/CMakeLists.txt:89`). No new kernel (`cuda_kernels.cu`) is required — cuBLAS suffices.

### Green gates (per increment)
- **CUDA build** (`build/cuda-on`): `cmake --build --preset cuda-on` for `n4m_c`; run `n4m_tests` (expect 351) and `n4m_internal_tests` (new equivalence tests pass / self‑skip without GPU).
- **Host build** (`dev-release`): `cmake --build --preset dev-release` `n4m_c` + `n4m_tests` — confirms the `#if`‑guarded code compiles out and host numerics are unchanged.
- **Catalog/ABI**: `catalog/scripts/validate.py --check-references --strict-abi`; `reconcile_abi.py --check` → **702/702**; confirm `build_moments_device`/`build_gram_device` **absent** from `nm -D` (as `spd_solve` is, `aom_moment_worklog.md:70`). ABI **1.22.0** unchanged.
- **`git diff --check`**.
- **Candidate‑score parity (`run_moment_sweep`, ≤1e-8)**: run a wide‑p screen (BERRY) through the cross‑binding orchestrator / a small driver with the CUDA build and the host (`dev-release`) build, diff `candidate_scores` (the `[id, head, param, rmse]` rows assembled at `aom_sweep.cpp:2295-2315`) max‑abs `≤1e-8`. Because B3 doesn't change numerics vs the existing CUDA path, expect **bit‑identical to post‑B2 CUDA** and `≤1e-8` vs host — the same RMSEP‑preserved result B1/B2 reported (`aom_moment_worklog.md:33-34`).
- **Before/after**: BERRY/COLZA/LUCAS fit time vs `/tmp/bench_5b_baseline.csv`; expect a partial cut of the 1.66 s (the H2D re‑upload fraction). Report honestly — this removes the upload half, not the p×p D2H half.

### Realism: what B3 removes vs what is deferred
- **B3 removes** the redundant per‑candidate **H2D** of the materialized design matrix (3×→1× for moments, 2×→1× for the gram) plus all per‑build `cudaMalloc`/`cudaFree`. This is the "base re‑uploaded per candidate" fraction the orchestrator measured.
- **B3 does NOT remove** the dominant per‑candidate p×p **D2H** of `XᵀX` (35 MB BERRY / 141 MB LUCAS, 1+cv times) nor the scorer's per‑tile `C` H2D (`cuda_dispatch.cpp:877`). Those require **device‑resident consumers** and are explicitly **deferred to B4**.

---

## 6. Deferred to B4 (explicitly out of B3 scope, with the reason)

1. **Device‑resident moment/gram chain.** Keep `XᵀX`/`K` resident from build through the consumers, so the p×p D2H (the larger remaining fraction) never happens. Needs: device kernels for `subtract_moments`/`recompute_centered_moments` (currently host elementwise, `moments.cpp:207-255`, `:299-350`), and a **device‑pointer entry to the PLS scorer** so it consumes resident `C/s/yy` instead of re‑uploading per tile (`cuda_dispatch.cpp:870-879`) — the design doc's component 3 (`fused_moment_executor_design.md:127-128`). Consumer‑touching → higher risk to fold isolation and `≤1e-8`; do it after B3 proves the build fusion.
2. **K → `prepare_dual_ridge` device hand‑off.** Eliminate the build's K D2H followed immediately by `prepare_dual_ridge`'s K H2D (`cuda_dispatch.cpp:1331`). Entangled because `K` is still needed on host for the single‑λ `solve_dual_spd` fallback (`sweep.cpp:481-499`) and `K_cross` prediction — needs the host‑K to become conditional. Defer.
3. **Operator‑moment on‑device transform (option B literal).** Only relevant for `p ≤ 1024` datasets (banded/structured) and `p ≤ 48` (dense), which are not the measured benchmark. If a future mid‑p workload makes it matter, do it route‑aware and **never** force a dense p×p GEMM where the banded/structured route is cheaper.
