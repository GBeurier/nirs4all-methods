# GOAL — full GPU execution of the AOM/moment screening engine (2026-06-08 →)

Objective (user-set): make the three screening layers run **fully on GPU**, with a
measured before/after at each step.

1. **Moments on GPU** — base moments `XᵀX`/`XᵀY` built and kept **device-resident**, no
   per-candidate host round-trip.
2. **AOM on GPU** — the per-candidate operator→moment transform **and the Ridge GCV
   solve** on the device, batched over candidates.
3. **PLS tuning loop on GPU** — the screen batched over **chain × candidate × fold** in
   one launch (folds already batched), fed device-resident moments.

## Measured priority (5b-0, nsys on BERRY p=2101, 71.3 s) — see `fused_moment_executor_design.md`

| Rank | Host hotspot | Share | GPU target |
|---|---|---|---|
| 1 | `n4m_householder_qr` — **Ridge GCV solve** (CPU, O(p³/n³), single-thread) | **37%** | cuSOLVER SPD Cholesky/eigh |
| 2 | synchronous `cudaMemcpy` (888 blocking copies, 4.59 s) | ~20% samples | device residency / pinned mem |
| 3 | CPU OpenBLAS matmuls | ~6% | route to cuBLAS / device moments |
| — | `cuda_dispatch::gemm` (moment transform) | 2.6% | minor — batch later |

The draft "fused IKPLS scorer" and "dense moment transform" were **not** the
bottleneck (Codex-flagged, profile-confirmed). Reordered accordingly.

## Blocks (ordered by measured value; each behind the green gate, ABI 1.22.0, fp64 host-equivalence)

- **B1 — Ridge GCV solve → GPU (cuSOLVER).** ✅ DONE (2026-06-08). Device SPD Cholesky
  (`spd_solve`) on the 3 dual ridge sites, host fallback, equivalence-tested. **3.83×
  total** on BERRY/COLZA/LUCAS (BERRY 7.59×), RMSEP preserved, green gate green, ABI
  1.22.0. See worklog 2026-06-08.
- **B2 — Device-resident dual-Ridge (upload K once).** ✅ DONE (2026-06-08). Per-fold
  `prepare_dual_ridge` uploads K once + reuse-buffers Cholesky + `add_scaled_identity`
  λ-shift. **1.07× incremental** (cumulative 4.22×), RMSEP bit-identical, green gate
  green, ABI 1.22.0. Targeted the ridge-solve copies; broader moment copies → B3.
- **B3 — Device moment/gram build.** ⚙️ KEPT as a size-gated env opt-in (2026-06-08).
  The un-gated drop-in regressed (0.96×: the moment build is host-OpenBLAS-optimal and
  XᵀX is consumed on host, so GPU-only-build adds a D2H), so it is **gated**:
  `build_moments_device`/`build_gram_device` run only when `n·p² ≥
  N4M_CUDA_MOMENT_MIN_PRODUCT` (default 15e9; `0`=always GPU, higher=always host). Correct,
  bit-equivalent, equivalence-tested; a wash on consumer GPUs but a potential win on
  strong-fp64 datacenter GPUs (A100/H100) — zero cost when off. Default = host build. See
  worklog. (The operator-transform route Fable's trace expected is dead code, gated off
  p>1024 — not the target.)
- **B4 — (not pursued)** Full device-resident moment chain + device-pointer PLS scorer
  entry. The investigation (incl. LUCAS_SOC: moment build ~1% of a 2402s fit) showed the
  moment build is never the pipeline bottleneck at any scale, so B4 would not help either;
  the captured win is B1+B2 (4.22×). Pursue only if a future profile / different hardware
  justifies it.

## Process (multi-agent, user-set)

- **Fable 5** (`claude --model claude-fable-5`): block design/roadmap + code review.
- **Codex** (`codex exec`): implementation under Fable's design + Opus's directives.
- **Opus** (orchestrator): directives, final diff review, build + green gate
  (`n4m_tests`/`n4m_internal_tests` CUDA & dev-release, catalog/ABI checks), and the
  before/after benchmark on BERRY/COLZA/LUCAS (`/tmp/bench_5b_baseline.csv` is the
  "before"). Corrects when needed.

Green gate per block: CUDA `n4m_c` + `n4m_tests` + `n4m_internal_tests`; dev-release
`n4m_c`; `catalog/scripts/validate.py --check-references --strict-abi`,
`reconcile_abi.py --check`; `git diff --check`. ABI stays 1.22.0 (internal only).
