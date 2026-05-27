# Reproducible scikit-learn reference environment (env-py-sklearn-1.4).
# Build:  docker build -t env-py-sklearn-1.4 parity/environments/env-py-sklearn-1.4
# The deferred reference generator (full Phase C, C6) runs inside this image so
# its snapshots carry a stable, hashable dependency set. No GHCR publish wired
# yet (see ../README.md).
FROM python:3.11-slim

# Pinned, deterministic install from the lock — no resolver freedom.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# Single-threaded math so reference outputs are bit-stable across hosts.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0

WORKDIR /work
