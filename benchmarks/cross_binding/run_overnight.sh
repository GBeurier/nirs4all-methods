#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_PRESET="${BUILD_PRESET:-blas-omp}"
LIBP4A_BUILD="${LIBP4A_BUILD:-${BUILD_PRESET}}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
ALGORITHMS="${ALGORITHMS:-all}"
THREADS="${THREADS:-1 3 10}"
N_RUNS="${N_RUNS:-5}"
TIMEOUT="${TIMEOUT:-300}"
REFERENCE_BACKENDS="${REFERENCE_BACKENDS:-all}"
OUT_CSV="${OUT_CSV:-benchmarks/cross_binding/results/full_matrix.csv}"
LOG_DIR="${LOG_DIR:-benchmarks/cross_binding/results/logs}"
CANONICAL_ONLY="${CANONICAL_ONLY:-1}"
RENDER_DOCS="${RENDER_DOCS:-auto}"
BUILD_SITE="${BUILD_SITE:-auto}"
SITE_DIR="${SITE_DIR:-docs/_build/html}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"
PUBLISH_WEB="${PUBLISH_WEB:-0}"
COMMIT_WEB="${COMMIT_WEB:-0}"
PUSH_WEB="${PUSH_WEB:-0}"
DEPLOY_PAGES="${DEPLOY_PAGES:-0}"
WEB_COMMIT_MESSAGE="${WEB_COMMIT_MESSAGE:-Update benchmark web data}"

if [[ "${RENDER_DOCS}" == "auto" ]]; then
  if [[ "${OUT_CSV}" == "benchmarks/cross_binding/results/full_matrix.csv" ]]; then
    RENDER_DOCS=1
  else
    RENDER_DOCS=0
  fi
fi
if [[ "${BUILD_SITE}" == "auto" ]]; then
  BUILD_SITE="${RENDER_DOCS}"
fi
if [[ "${PUBLISH_WEB}" == "1" ]]; then
  COMMIT_WEB=1
  PUSH_WEB=1
  DEPLOY_PAGES=1
fi

mkdir -p "${LOG_DIR}" "$(dirname -- "${OUT_CSV}")"
LOG_FILE="${LOG_DIR}/overnight_$(date -u +%Y%m%dT%H%M%SZ).log"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is not in PATH. Activate the conda env or install cmake." >&2
  exit 2
fi
if ! command -v ninja >/dev/null 2>&1; then
  echo "ERROR: ninja is not in PATH. Activate the conda env or install ninja." >&2
  exit 2
fi
if [[ "${DEPLOY_PAGES}" == "1" ]] && ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh is required when DEPLOY_PAGES=1." >&2
  exit 2
fi

if [[ "${BUILD_PRESET}" == blas* ]]; then
  : "${CONDA_PREFIX:?CONDA_PREFIX must point to the conda env that provides OpenBLAS}"
  if [[ ! -f "${CONDA_PREFIX}/include/cblas.h" ]]; then
    echo "ERROR: ${CONDA_PREFIX}/include/cblas.h is missing." >&2
    echo "Install OpenBLAS headers, e.g. conda install -c conda-forge openblas." >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

canonical_flags=()
if [[ "${CANONICAL_ONLY}" == "1" ]]; then
  canonical_flags+=(--canonical-pls4all-only)
fi

resume_flags=()
if [[ "${RESUME}" == "1" ]]; then
  resume_flags+=(--resume-existing)
fi
if [[ "${FORCE}" == "1" ]]; then
  resume_flags+=(--force)
fi
if [[ "${RERUN_FAILED}" == "1" ]]; then
  resume_flags+=(--rerun-failed)
fi

{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) starting pls4all overnight benchmark"
  echo "# build=${BUILD_PRESET} clean_build=${CLEAN_BUILD} libp4a=${LIBP4A_BUILD} algorithms=${ALGORITHMS} threads=${THREADS} n_runs=${N_RUNS}"
  echo "# references=${REFERENCE_BACKENDS} canonical_only=${CANONICAL_ONLY} resume=${RESUME} force=${FORCE} rerun_failed=${RERUN_FAILED} render_docs=${RENDER_DOCS} build_site=${BUILD_SITE} publish_web=${PUBLISH_WEB} timeout=${TIMEOUT}"

  if [[ "${CLEAN_BUILD}" == "1" ]]; then
    rm -rf "build/${BUILD_PRESET}"
  fi
  cmake --preset "${BUILD_PRESET}"
  cmake --build --preset "${BUILD_PRESET}" -j "${BUILD_JOBS:-$(nproc)}"
  ctest --preset "${BUILD_PRESET}"

  python benchmarks/cross_binding/orchestrator.py \
    --algorithms ${ALGORITHMS} \
    --registry-cells \
    --threads ${THREADS} \
    --n-runs "${N_RUNS}" \
    --timeout "${TIMEOUT}" \
    --libp4a-build "${LIBP4A_BUILD}" \
    "${canonical_flags[@]}" \
    "${resume_flags[@]}" \
    --reference-backends "${REFERENCE_BACKENDS}" \
    --out-csv "${OUT_CSV}"

  if [[ "${RENDER_DOCS}" == "1" ]]; then
    python benchmarks/cross_binding/combine_and_render.py \
      --csvs "${OUT_CSV}" \
      --out docs/benchmarks/cross_binding.md

    python docs/_extras/build_landing.py \
      --results benchmarks/cross_binding/results \
      --out docs/_static/bench-data.json
  fi

  if [[ "${BUILD_SITE}" == "1" ]]; then
    python -m sphinx -b html docs "${SITE_DIR}" --keep-going
    echo "# HTML: ${SITE_DIR}/index.html"
  fi

  if [[ "${COMMIT_WEB}" == "1" ]]; then
    git add docs/_static/bench-data.json \
            docs/benchmarks/cross_binding.md \
            docs/benchmarks/cross_binding_threads.md
    if git diff --cached --quiet; then
      echo "# No web source changes to commit"
    else
      git commit -m "${WEB_COMMIT_MESSAGE}"
    fi
  fi

  if [[ "${PUSH_WEB}" == "1" ]]; then
    git push
  fi

  if [[ "${DEPLOY_PAGES}" == "1" ]]; then
    branch="$(git branch --show-current)"
    gh workflow run docs.yml --ref "${branch}"
    echo "# Triggered GitHub Pages workflow on ${branch}"
  fi

  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
  echo "# CSV: ${OUT_CSV}"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
