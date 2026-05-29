#!/usr/bin/env bash
# Install every external *donor* dependency the parity matrix needs, so that a
# "missing value" in the divergence/benchmark matrix never means "forgot to
# install a reference library". Idempotent — safe to re-run.
#
# Three dependency layers:
#   1. Python reference packages  -> the parity venv (parity/python_generator/.venv)
#   2. R reference packages       -> the R env in $N4M_R_ENV (else PATH Rscript)
#   3. Octave + oct2py + libPLS   -> Octave from the R env; libPLS is vendored in-tree
#
# Usage:
#   N4M_R_ENV=/home/delete/miniconda3/envs/pls4all_r scripts/setup_parity_references.sh
#   scripts/setup_parity_references.sh --no-r      # skip the R layer
#
# After install it runs parity/scripts/check_references.py --strict and fails if
# any REQUIRED reference is still missing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV_PY="${PARITY_VENV_PY:-$REPO/parity/python_generator/.venv/bin/python}"
DO_R=1
for a in "$@"; do [ "$a" = "--no-r" ] && DO_R=0; done

echo "==> [1/3] Python reference packages -> $VENV_PY"
if [ ! -x "$VENV_PY" ]; then
  echo "    parity venv not found; creating it"
  python3 -m venv "$REPO/parity/python_generator/.venv"
  "$VENV_PY" -m pip install -U pip
  # base references pinned by the generator
  [ -f "$REPO/parity/python_generator/requirements-lock.txt" ] && \
    "$VENV_PY" -m pip install -r "$REPO/parity/python_generator/requirements-lock.txt" || true
fi
# Reference libraries imported by benchmarks/parity_timing/registry.py.
# diPLSlib is pinned to 2.5.0 — the registry carries shims for that exact API.
"$VENV_PY" -m pip install --no-input \
  "scikit-learn" "numpy" "scipy" \
  "auswahl" "pyswarms" "tensorly" \
  "ikpls" "diPLSlib==2.5.0" "oct2py>=5.6"

echo "==> [2/3] R reference packages"
if [ "$DO_R" -eq 1 ]; then
  R_BIN="Rscript"
  if [ -n "${N4M_R_ENV:-}" ]; then R_BIN="$N4M_R_ENV/bin/Rscript"; fi
  if command -v "$R_BIN" >/dev/null 2>&1 || [ -x "$R_BIN" ]; then
    # mbpls intentionally excluded (unused; broken vs sklearn 1.8).
    "$R_BIN" --vanilla -e '
      pkgs <- c("plsVarSel","enpls","multiblock","plsRglm","plsRcox","chemometrics",
                "JICO","ropls","mixOmics","sgPLS","mdatools","mboost","softImpute",
                "survival","survAUC","rms","permute","prospectr","pls","spls",
                "OmicsPLS","kernlab","corpcor","genalg","baseline","EMSC","wavelets",
                "plsdepot","mvdalab","HDANOVA","multiway")
      have <- rownames(installed.packages())
      need <- setdiff(pkgs, have)
      if (length(need)) {
        cat("installing:", paste(need, collapse=", "), "\n")
        # Bioconductor packages (mixOmics, ropls) need BiocManager
        bioc <- intersect(need, c("mixOmics","ropls"))
        cran <- setdiff(need, bioc)
        if (length(cran)) install.packages(cran, repos="https://cloud.r-project.org")
        if (length(bioc)) {
          if (!requireNamespace("BiocManager", quietly=TRUE))
            install.packages("BiocManager", repos="https://cloud.r-project.org")
          BiocManager::install(bioc, update=FALSE, ask=FALSE)
        }
      } else cat("all R reference packages already present\n")
    '
  else
    echo "    no Rscript found (set N4M_R_ENV); skipping R layer"
  fi
else
  echo "    skipped (--no-r)"
fi

echo "==> [3/3] Octave / libPLS bridge"
# libPLS is vendored at bindings/octave/libPLS_1.95 (no install).
# Octave itself is expected from the R env (conda) or the system package manager.
if [ -n "${N4M_R_ENV:-}" ] && [ -x "$N4M_R_ENV/bin/octave-cli" ]; then
  echo "    octave-cli: $N4M_R_ENV/bin/octave-cli"
  export OCTAVE_EXECUTABLE="$N4M_R_ENV/bin/octave-cli"
elif command -v octave-cli >/dev/null 2>&1; then
  echo "    octave-cli: $(command -v octave-cli)"
else
  echo "    WARNING: no octave-cli found. Install via conda (-c conda-forge octave)"
  echo "             or your OS package manager; ecr/cars libPLS refs need it."
fi

echo "==> [4/4] nirs4all donor reference (aom_pls / aom_preprocess / pop_pls)"
# The live nirs4all donor needs Python >=3.11 (parity venv may be older) + nirs4all's
# non-DL deps. Build a dedicated reference venv and expose it via N4M_NIRS4ALL_PYTHON.
N4_SRC="$(cd "$REPO/.." && pwd)/nirs4all"
N4_VENV="$REPO/parity/.nirs4all_ref_venv"
PY311="$(command -v python3.11 || command -v python3.12 || command -v python3.13 || true)"
if [ -z "${N4M_NIRS4ALL_PYTHON:-}" ] && [ -d "$N4_SRC" ] && [ -n "$PY311" ]; then
  [ -x "$N4_VENV/bin/python" ] || "$PY311" -m venv "$N4_VENV"
  "$N4_VENV/bin/pip" install -q -U pip
  # nirs4all core + viz deps (DL backends — tensorflow/torch/jax — are lazy-loaded).
  "$N4_VENV/bin/pip" install -q numpy pandas scipy scikit-learn polars pyarrow h5py \
    PyWavelets pybaselines joblib jsonschema pyyaml packaging pydantic kennard-stone \
    ikpls matplotlib seaborn umap-learn shap optuna pyopls trendfitter
  if PYTHONPATH="$N4_SRC" "$N4_VENV/bin/python" -c "import nirs4all" 2>/dev/null; then
    export N4M_NIRS4ALL_PYTHON="$N4_VENV/bin/python"
    echo "    nirs4all donor venv ready -> N4M_NIRS4ALL_PYTHON=$N4M_NIRS4ALL_PYTHON"
  else
    echo "    WARNING: nirs4all still not importable in $N4_VENV (check its deps)."
  fi
elif [ -n "${N4M_NIRS4ALL_PYTHON:-}" ]; then
  echo "    using configured N4M_NIRS4ALL_PYTHON=$N4M_NIRS4ALL_PYTHON"
else
  echo "    SKIP: no python>=3.11 or no sibling nirs4all repo; aom/pop donor cells need it."
fi

echo
echo "==> Verifying reference completeness"
# nirs4all is the canonical donor for aom_pls/aom_preprocess/pop_pls/mb_pls/lw_pls,
# and the Octave bridge backs ecr/cars — require both so a green doctor really
# means a hole-free matrix (Codex review finding).
N4M_REQUIRE="--strict --require-octave"
[ -n "${N4M_NIRS4ALL_PYTHON:-}" ] && N4M_REQUIRE="$N4M_REQUIRE --require-nirs4all"
"$VENV_PY" "$REPO/parity/scripts/check_references.py" $N4M_REQUIRE
