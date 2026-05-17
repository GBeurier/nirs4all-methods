"""External reference backend driven by parity_timing.MethodSpec."""
from __future__ import annotations

import json
import os
import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   find_reference, load_method)


def main():
    algo, csv_dir, n, p, nc, runs, seed_base, pred_path = parse_args()
    ref_id = os.environ.get("BENCH_REFERENCE_ID", "")
    if not ref_id:
        print(json.dumps({"ok": False,
                          "reason": "BENCH_REFERENCE_ID not set",
                          "skipped": True}))
        return

    method = load_method(algo)
    params = adapted_params(method, n, p, nc)
    try:
        reference = find_reference(method, ref_id, params)
    except Exception as exc:
        print(json.dumps({"ok": False,
                          "reason": f"reference factory error: {exc}",
                          "skipped": True}))
        return
    if reference is None:
        print(json.dumps({"ok": False,
                          "reason": f"reference '{ref_id}' not registered for {algo}",
                          "skipped": True}))
        return

    def fit_predict(seed: int) -> np.ndarray:
        X, y = load_dataset(csv_dir, n, p, seed)
        Y, extras = benchmark_inputs(method, X, y, params, seed)
        reference.fit(X, Y, **extras)
        return np.asarray(reference.predict(X))

    stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    versions = collect_versions(
        "Python" if reference.language == "python" else reference.language,
        numpy=_safe_version("numpy"),
        reference_id=ref_id,
        reference_library=reference.library_name,
        reference_version=reference.library_version,
    )
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()
