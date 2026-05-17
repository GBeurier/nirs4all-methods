#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

FULL_MATRIX="${FULL_MATRIX:-0}"
BUILD_PRESET="${BUILD_PRESET:-blas-omp}"
if [[ "${FULL_MATRIX}" == "1" ]]; then
  LIBP4A_BUILD="${LIBP4A_BUILD:-all-cpu}"
  CANONICAL_ONLY="${CANONICAL_ONLY:-0}"
  REGISTRY_CELLS="${REGISTRY_CELLS:-0}"
else
  LIBP4A_BUILD="${LIBP4A_BUILD:-${BUILD_PRESET}}"
  CANONICAL_ONLY="${CANONICAL_ONLY:-1}"
  REGISTRY_CELLS="${REGISTRY_CELLS:-1}"
fi
BUILD_PRESETS="${BUILD_PRESETS:-auto}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
ALGORITHMS="${ALGORITHMS:-all}"
SIZES="${SIZES:-}"
THREADS="${THREADS:-1 3 10}"
N_RUNS="${N_RUNS:-5}"
TIMEOUT="${TIMEOUT:-300}"
REFERENCE_BACKENDS="${REFERENCE_BACKENDS:-all}"
OUT_CSV="${OUT_CSV:-benchmarks/cross_binding/results/full_matrix.csv}"
LOG_DIR="${LOG_DIR:-benchmarks/cross_binding/results/logs}"
RENDER_DOCS="${RENDER_DOCS:-auto}"
BUILD_SITE="${BUILD_SITE:-auto}"
SITE_DIR="${SITE_DIR:-docs/_build/html}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"
PUBLISH_WEB="${PUBLISH_WEB:-0}"
COMMIT_WEB="${COMMIT_WEB:-0}"
PUSH_WEB="${PUSH_WEB:-0}"
DEPLOY_PAGES="${DEPLOY_PAGES:-auto}"
PAGES_BRANCH="${PAGES_BRANCH:-main}"
WEB_COMMIT_MESSAGE="${WEB_COMMIT_MESSAGE:-Update benchmark web data}"
CURRENT_BRANCH="$(git branch --show-current)"

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
  if [[ "${DEPLOY_PAGES}" == "auto" ]]; then
    DEPLOY_PAGES=0
  fi
elif [[ "${DEPLOY_PAGES}" == "auto" ]]; then
  DEPLOY_PAGES=0
fi

if [[ "${DEPLOY_PAGES}" == "1" && "${CURRENT_BRANCH}" != "${PAGES_BRANCH}" ]]; then
  echo "ERROR: GitHub Pages deployment is restricted to ${PAGES_BRANCH} in this repository." >&2
  echo "Current branch is ${CURRENT_BRANCH:-detached HEAD}." >&2
  echo "Run from ${PAGES_BRANCH} after merging, set PAGES_BRANCH if repository settings allow this branch, or use DEPLOY_PAGES=0 to only commit/push web source." >&2
  exit 2
fi

required_build_presets=()
case "${LIBP4A_BUILD}" in
  both)
    required_build_presets=(dev-release blas-omp)
    ;;
  all-cpu)
    required_build_presets=(dev-release blas-on omp-on blas-omp)
    ;;
  all)
    required_build_presets=(dev-release blas-on omp-on blas-omp cuda-on)
    ;;
  *)
    required_build_presets=("${LIBP4A_BUILD}")
    ;;
esac

if [[ "${BUILD_PRESETS}" == "auto" ]]; then
  build_presets=("${required_build_presets[@]}")
else
  read -r -a build_presets <<< "${BUILD_PRESETS}"
fi
if [[ "${#build_presets[@]}" -eq 0 ]]; then
  echo "ERROR: no CMake build presets selected." >&2
  exit 2
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

requires_openblas=0
for preset in "${build_presets[@]}"; do
  if [[ "${preset}" == "blas-on" || "${preset}" == "blas-omp" ]]; then
    requires_openblas=1
  fi
done
if [[ "${requires_openblas}" == "1" ]]; then
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

registry_flags=()
if [[ "${REGISTRY_CELLS}" == "1" ]]; then
  registry_flags+=(--registry-cells)
fi

size_flags=()
if [[ -n "${SIZES}" ]]; then
  read -r -a size_args <<< "${SIZES}"
  size_flags+=(--sizes "${size_args[@]}")
fi

read -r -a algorithm_args <<< "${ALGORITHMS}"
read -r -a thread_args <<< "${THREADS}"

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
  echo "# build_presets=${build_presets[*]} clean_build=${CLEAN_BUILD} libp4a=${LIBP4A_BUILD} algorithms=${ALGORITHMS} registry_cells=${REGISTRY_CELLS} sizes=${SIZES:-default} threads=${THREADS} n_runs=${N_RUNS}"
  echo "# references=${REFERENCE_BACKENDS} canonical_only=${CANONICAL_ONLY} full_matrix=${FULL_MATRIX} resume=${RESUME} force=${FORCE} rerun_failed=${RERUN_FAILED} render_docs=${RENDER_DOCS} build_site=${BUILD_SITE} publish_web=${PUBLISH_WEB} deploy_pages=${DEPLOY_PAGES} pages_branch=${PAGES_BRANCH} timeout=${TIMEOUT}"

  for preset in "${build_presets[@]}"; do
    if [[ "${CLEAN_BUILD}" == "1" ]]; then
      rm -rf "build/${preset}"
    fi
    cmake --preset "${preset}"
    cmake --build --preset "${preset}" -j "${BUILD_JOBS:-$(nproc)}"
    ctest --preset "${preset}"
  done

  python benchmarks/cross_binding/orchestrator.py \
    --algorithms "${algorithm_args[@]}" \
    "${registry_flags[@]}" \
    "${size_flags[@]}" \
    --threads "${thread_args[@]}" \
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
    gh workflow run docs.yml --ref "${CURRENT_BRANCH}"
    echo "# Triggered GitHub Pages workflow on ${CURRENT_BRANCH}"
  fi

  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
  echo "# CSV: ${OUT_CSV}"
} 2>&1 | tee "${LOG_FILE}"

echo "Log: ${LOG_FILE}"
