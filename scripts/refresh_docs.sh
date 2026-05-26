#!/usr/bin/env bash
# Regenerate every documentation artifact from the current benchmark results.
#
# Run this after a benchmark sweep (e.g. benchmarks/cross_binding/orchestrator.py)
# so the github.io webpages AND the interactive parity/timing matrix reflect the
# latest results/ CSVs. The results CSVs are gitignored, so these committed
# artifacts are what the docs deploy (CI builds with the CSVs absent → falls
# back to the committed snapshots).
#
# Regenerates:
#   - docs/methods/<method|operator>.md + index.md   (build_methods.py)
#   - docs/_static/bench-data.json                    (build_landing.py — landing matrix)
#   - docs/_extras/method-benchmarks.json             (per-method table snapshot)
#   - docs/benchmarks/cross_binding{,_threads}.md     (combine_and_render.py)
#
# Usage:
#   scripts/refresh_docs.sh            # regenerate from benchmarks/cross_binding/results
#   RESULTS=/path/to/results scripts/refresh_docs.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

results="${RESULTS:-benchmarks/cross_binding/results}"
full_matrix="$results/full_matrix.csv"

if [[ ! -f "$full_matrix" ]]; then
    echo "error: $full_matrix not found — run the benchmark orchestrator first." >&2
    exit 1
fi

echo "[refresh_docs] 1/4 method + operator pages (build_methods.py)"
python docs/_extras/build_methods.py

echo "[refresh_docs] 2/4 landing matrix payload (build_landing.py)"
python docs/_extras/build_landing.py --results "$results" --out docs/_static/bench-data.json

echo "[refresh_docs] 3/4 per-method benchmark snapshot (method-benchmarks.json)"
python - <<'PY'
import sys
sys.path.insert(0, "docs/_extras")
from method_benchmark_tables import load_or_build_blocks
blocks, source = load_or_build_blocks()
print(f"  {len(blocks)} method blocks from {source}")
PY

echo "[refresh_docs] 4/4 cross-binding markdown pages (combine_and_render.py)"
python benchmarks/cross_binding/combine_and_render.py \
    --csvs "$full_matrix" \
    --out docs/benchmarks/cross_binding.md

echo "[refresh_docs] done. Review + commit:"
echo "  docs/methods/  docs/_static/bench-data.json  docs/_extras/method-benchmarks.json  docs/benchmarks/cross_binding*.md"
