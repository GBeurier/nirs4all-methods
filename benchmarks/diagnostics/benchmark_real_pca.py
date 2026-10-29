#!/usr/bin/env python3
"""Opt-in representative native PCA benchmark; never downloads or modifies data.

The input manifest supplies real canonical CSV identities/licenses. Measurements
are descriptive for this host, not a guarantee about independent observations.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import threading
import time
from pathlib import Path

import numpy as np
from n4m.roles import method_class


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rss():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
        "SC_PAGE_SIZE"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["datasets"]
    results = []
    for item in rows:
        leaf = Path(item["local_leaf"])
        before = {name: sha(leaf / name) for name in item["input_hashes"]}
        assert before == item["input_hashes"]
        with (leaf / "X.csv").open() as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            axis = np.array(header[1:], dtype=float)
            x = np.array([[float(v) for v in row[1:]] for row in reader], dtype=float)
        assert list(x.shape) == item["shape"] and np.isfinite(x).all()
        assert len(axis) == x.shape[1] and np.all(np.diff(axis) > 0)
        gc.collect()
        measurements = []
        for _ in range(args.repeats):
            baseline = rss()
            observed = [baseline]
            stop = threading.Event()

            def sample(stop=stop, observed=observed):
                while not stop.wait(0.002):
                    observed.append(rss())

            sampler = threading.Thread(target=sample)
            sampler.start()
            start = time.perf_counter()
            model = method_class("preprocessing.feature_selection.flexible_pca")(
                n_components=5.0
            )
            model.fit(x)
            fitted = time.perf_counter()
            initial = model.transform(x)
            train_transformed = time.perf_counter()
            query = x[:32].copy() * 1.1 + 0.03
            projected = model.transform(query)
            transformed = time.perf_counter()
            unchanged = model.transform(x)
            assert np.array_equal(initial, unchanged), (
                "evaluation changed fitted reference"
            )
            assert np.isfinite(projected).all() and projected.shape == (32, 5)
            stop.set()
            sampler.join()
            observed.append(rss())
            measurements.append(
                {
                    "fit_seconds": fitted - start,
                    "train_transform_seconds": train_transformed - fitted,
                    "query_32_transform_seconds": transformed - train_transformed,
                    "peak_observed_rss_bytes": max(observed),
                    "baseline_rss_bytes": baseline,
                    "peak_increment_bytes": max(observed) - baseline,
                    "frozen_training_scores_bit_identical": True,
                }
            )
            del model, initial, unchanged, projected
            gc.collect()
        # A separate native descriptive operation exposes the requested train-only
        # means/loadings/eigenvalues without ever claiming exact full moments.
        start = time.perf_counter()
        qc = method_class("diagnostics.pca_reference_qc")(n_components=5).run(
            x[:32], X_target=x
        )
        qc_seconds = time.perf_counter() - start
        assert int(qc["exact_full_moments_status"][0]) == 6
        assert (
            np.isfinite(qc["loadings"]).all() and np.isfinite(qc["eigenvalues"]).all()
        )
        # Independent SVD oracle on the complete real matrix; compare invariant
        # subspaces/reconstruction, avoiding a false sign convention constraint.
        start = time.perf_counter()
        _, singular, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        oracle_seconds = time.perf_counter() - start
        p = qc["loadings"].T @ qc["loadings"]
        oracle = vt[:5].T @ vt[:5]
        projector_error = float(np.max(abs(p - oracle)))
        eigen_error = float(
            np.max(abs(qc["eigenvalues"].ravel() - singular[:5] ** 2 / (len(x) - 1)))
        )
        print(
            "ORACLE",
            item["id"],
            "projector",
            projector_error,
            "eigen_abs",
            eigen_error,
            "eigen_expected",
            (singular[:5] ** 2 / (len(x) - 1)).tolist(),
            flush=True,
        )
        assert projector_error < 2e-6 and eigen_error < 2e-6
        after = {name: sha(leaf / name) for name in item["input_hashes"]}
        assert after == before
        results.append(
            {
                "dataset_id": item["id"],
                "shape": item["shape"],
                "axis": item["axis"],
                "license": item["license_as_declared_by_catalog"],
                "origin_sources": item["origin_sources"],
                "input_hashes_before": before,
                "input_hashes_after": after,
                "measurements": measurements,
                "median_fit_seconds": statistics.median(
                    v["fit_seconds"] for v in measurements
                ),
                "pca_reference_qc_seconds": qc_seconds,
                "oracle_seconds": oracle_seconds,
                "oracle_projector_max_abs_error": projector_error,
                "oracle_eigenvalue_max_abs_error": eigen_error,
                "physical_independence": "not established or inferred; all rows used only for numerical/performance qualification",
                "exact_full_moments": "not requested by either fitted PCA or pca_reference_qc",
            }
        )
        args.out.write_text(
            json.dumps({"status": "in_progress", "datasets": results}, indent=2) + "\n"
        )
        print(
            item["id"],
            x.shape,
            "median fit",
            results[-1]["median_fit_seconds"],
            "projector error",
            projector_error,
            flush=True,
        )
    library = Path(os.environ["N4M_LIB_PATH"]).resolve()
    result = {
        "status": "qualified_locally",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
        },
        "library": str(library),
        "library_sha256": sha(library),
        "datasets": results,
        "memory_method": "2ms /proc RSS sampling during native fit/projections; sampled peak, not a rigorous allocation upper bound",
        "maximum_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
