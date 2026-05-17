#!/bin/bash
# After niveau B completes, run niveau C (pls4all-only algos at condensed
# sizes/threads) then merge all 3 CSVs into the docs.

set -e

REPO=/home/delete/nirs4all/pls4all
LOG_B=/tmp/niveau_b.log
LOG_C=/tmp/niveau_c.log
LOG_RENDER=/tmp/render.log

cd "$REPO"

# 1) Wait for niveau B's orchestrator to finish — the run prints
# "CSV  → ..." as its very last line. Poll until we see it.
echo "[remainder] waiting for niveau B to finish..."
while ! grep -q "^CSV  → " "$LOG_B" 2>/dev/null; do
    sleep 30
done
echo "[remainder] niveau B done."

# 2) Niveau C — pls4all-only algos (no external counterpart). Condensed:
# 5 sizes × 2 thread counts × ~25 algos × 8 pls4all backends ≈ 2000 cells.
echo "[remainder] launching niveau C..."
PATH=/home/delete/miniconda3/envs/pls4all_r/bin:$PATH \
    /home/delete/miniconda3/bin/python \
    "$REPO/benchmarks/cross_binding/orchestrator.py" \
    --algorithms \
        continuum_regression kernel_pls o2pls \
        bagging_pls boosting_pls \
        sparse_pls_da group_sparse_pls fused_sparse_pls \
        pls_glm pls_qda pls_lda pls_logistic pls_cox \
        pds ds lw_pls weighted_pls \
        gpr_pls \
    --sizes 100x500 500x500 2500x500 2500x2500 \
    --threads 1 10 \
    --n-runs 3 \
    --only-pls4all \
    --libp4a-build blas-omp \
    --out-csv "$REPO/benchmarks/cross_binding/results/niveau_C.csv" \
    2>&1 | tee "$LOG_C" | tail -20

echo "[remainder] niveau C done."

# 3) Merge all CSVs and re-render the docs page.
echo "[remainder] merging + rendering..."
/home/delete/miniconda3/bin/python "$REPO/benchmarks/cross_binding/combine_and_render.py" \
    --csvs \
        "$REPO/benchmarks/cross_binding/results/niveau_A_pls.csv" \
        "$REPO/benchmarks/cross_binding/results/niveau_B.csv" \
        "$REPO/benchmarks/cross_binding/results/niveau_C.csv" \
    --out "$REPO/docs/benchmarks/cross_binding.md" \
    2>&1 | tee "$LOG_RENDER"

echo "[remainder] ALL DONE."
