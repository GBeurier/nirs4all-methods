"""External reference backend driven by parity_timing.MethodSpec."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                       load_dataset, collect_versions, _safe_version)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   find_reference, load_method)


def _warmup_count(n_runs: int) -> int:
    return max(1, min(3, int(n_runs)))


def _r_quote(path: Path | str) -> str:
    return json.dumps(str(path))


def _write_xy_case(tmp: Path, idx: int, X: np.ndarray,
                   Y: np.ndarray) -> tuple[Path, Path, Path, Path, Path]:
    case = tmp / f"case_{idx:03d}"
    case.mkdir(parents=True, exist_ok=True)
    x_path = case / "X.csv"
    y_path = case / "Y.csv"
    xp_path = case / "X_predict.csv"
    pred_path = case / "predictions.csv"
    idx_path = case / "selected_indices.csv"
    np.savetxt(x_path, np.asarray(X, dtype=np.float64), delimiter=",")
    np.savetxt(y_path, np.asarray(Y, dtype=np.float64).reshape(X.shape[0], -1),
               delimiter=",")
    np.savetxt(xp_path, np.asarray(X, dtype=np.float64), delimiter=",")
    return x_path, y_path, xp_path, pred_path, idx_path


def _stats_from_samples(samples: list[float]) -> dict:
    return {
        "ok": True,
        "n_runs": len(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _read_csv_array(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0,), dtype=np.float64)
    arr = np.loadtxt(path, delimiter=",")
    return np.asarray(arr, dtype=np.float64)


def _mask_from_indices(path: Path, n_features: int) -> np.ndarray:
    mask = np.zeros(int(n_features), dtype=np.float64)
    if path.exists() and path.stat().st_size > 0:
        try:
            idx = np.atleast_1d(np.loadtxt(path, delimiter=",",
                                           dtype=np.int64)) - 1
        except Exception:
            idx = np.empty((0,), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_features)]
        if idx.size:
            mask[idx] = 1.0
    return mask.reshape(1, -1)


def _selector_body(reference, x_path: Path, y_path: Path,
                   idx_path: Path, n: int, p: int) -> str | None:
    """Return one self-contained R selector body, or None if unsupported.

    Several registry references are selector adapters whose Python
    `_compute_indices()` helper used to fork Rscript for each run. Rebuilding
    the tiny R bodies here lets the cross-binding benchmark measure the
    external package call in one warm R process instead of timing R startup.
    """
    cls = reference.__class__.__name__
    xq = _r_quote(x_path)
    yq = _r_quote(y_path)
    iq = _r_quote(idx_path)
    if hasattr(reference, "_r_call"):
        return (
            "local({\n"
            "pdf(NULL)\n"
            "suppressPackageStartupMessages({library(pls); library(plsVarSel)})\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            f"{reference._r_call(n, p)}\n"
            f"write.table(matrix(as.integer(selected), ncol=1), file={iq}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n"
            "})\n"
        )
    if cls in {"_IplsForwardReference", "_IplsBackwardReference"}:
        method = "backward" if cls == "_IplsBackwardReference" else "forward"
        n_intervals = max(2, p // int(reference._w))
        return (
            "local({\n"
            "suppressPackageStartupMessages(library(mdatools))\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            f"res <- ipls(X, y, glob.ncomp={int(reference._k)}, "
            f"int.num={n_intervals}, method='{method}', silent=TRUE)\n"
            "selected <- as.integer(res$var.selected)\n"
            f"write.table(matrix(selected, ncol=1), file={iq}, sep=',', "
            "row.names=FALSE, col.names=FALSE)\n"
            "})\n"
        )
    if cls == "_CarsEnplsReference":
        return (
            "local({\n"
            "suppressPackageStartupMessages(library(enpls))\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            "set.seed(11)\n"
            f"res <- enpls.fs(X, y, maxcomp={int(reference._k)}, cvfolds=3, "
            "reptimes=10, method='mc', ratio=0.8)\n"
            "imp <- abs(res$variable.importance)\n"
            "o <- order(imp, decreasing=TRUE)\n"
            f"selected <- sort(head(o, {int(reference._top_k)}))\n"
            f"write.table(matrix(as.integer(selected), ncol=1), file={iq}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n"
            "})\n"
        )
    if cls == "_EmcuvePlsVarSelReference":
        return (
            "local({\n"
            "pdf(NULL)\n"
            "suppressPackageStartupMessages({library(pls); library(plsVarSel)})\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            "p <- ncol(X)\n"
            "votes <- integer(p)\n"
            f"for (e in 1:{int(reference._n_ens)}) {{\n"
            "  set.seed(11 + e)\n"
            f"  res <- mcuve_pls(y, X, ncomp={int(reference._k)}, "
            "N=3, ratio=0.75)\n"
            "  sel <- res$mcuve.selection\n"
            "  votes[sel] <- votes[sel] + 1L\n"
            "}\n"
            f"freq <- votes / {int(reference._n_ens)}\n"
            f"selected <- sort(which(freq >= {float(reference._vote)}))\n"
            f"write.table(matrix(as.integer(selected), ncol=1), file={iq}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n"
            "})\n"
        )
    if cls == "_RandomizationPermuteReference":
        return (
            "local({\n"
            "pdf(NULL)\n"
            "suppressPackageStartupMessages(library(pls))\n"
            f"set.seed({int(reference._seed)})\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            f"fit <- plsr(y ~ X, ncomp={int(reference._k)}, "
            "method='simpls', scale=FALSE)\n"
            f"obs <- abs(as.numeric(coef(fit, ncomp={int(reference._k)}, "
            "intercept=FALSE)))\n"
            f"p <- ncol(X); B <- {int(reference._n_perm)}\n"
            "counts <- rep(0L, p)\n"
            "for (b in 1:B) {\n"
            "  yp <- sample(y)\n"
            f"  fb <- plsr(yp ~ X, ncomp={int(reference._k)}, "
            "method='simpls', scale=FALSE)\n"
            f"  perm <- abs(as.numeric(coef(fb, ncomp={int(reference._k)}, "
            "intercept=FALSE)))\n"
            "  counts <- counts + as.integer(perm >= obs)\n"
            "}\n"
            "pvals <- (counts + 1) / (B + 1)\n"
            f"selected <- sort(which(pvals < {float(reference._alpha)}))\n"
            f"write.table(matrix(as.integer(selected), ncol=1), file={iq}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n"
            "})\n"
        )
    return None


def _time_r_reference_batched(reference, method, params, csv_dir, n: int,
                              p: int, runs: int, seed_base: int):
    """Time R registry references in one R process.

    The generic registry adapter is excellent for parity, but if it falls
    back to Rscript per prediction the dashboard shows R startup costs. This
    helper pre-materialises all datasets, builds one R script containing the
    warmups and timed runs, then reads the last prediction/mask back.
    """
    from benchmarks.parity_timing.registry import RAdapter, R_ENV, RSCRIPT

    if getattr(reference, "language", "") != "R":
        return None

    with tempfile.TemporaryDirectory(prefix="pls4all_rbatch_") as tmp_s:
        tmp = Path(tmp_s)
        warmups = [seed_base] * _warmup_count(runs)
        timed = [seed_base + i for i in range(runs)]
        cases: list[dict] = []
        script_parts = [
            "samples <- numeric(0)\n",
            "run_one <- function(expr, timed) {\n",
            "  if (timed) {\n",
            "    t0 <- as.numeric(Sys.time()) * 1000.0\n",
            "    force(expr)\n",
            "    samples <<- c(samples, as.numeric(Sys.time()) * 1000.0 - t0)\n",
            "  } else {\n",
            "    force(expr)\n",
            "  }\n",
            "}\n",
        ]
        last_pred_path: Path | None = None
        last_idx_path: Path | None = None
        for idx, (seed, is_timed) in enumerate(
                [(s, False) for s in warmups] + [(s, True) for s in timed]):
            X, y = load_dataset(csv_dir, n, p, seed)
            Y, extras = benchmark_inputs(method, X, y, params, seed)
            x_path, y_path, xp_path, out_path, idx_path = _write_xy_case(
                tmp, idx, X, Y)
            body = None
            is_mask = False
            if isinstance(reference, RAdapter):
                try:
                    reference.fit(X, Y, **extras)
                    body = reference._r_script(
                        x_path, y_path, out_path, xp_path,
                        int(X.shape[0]), int(X.shape[1]),
                        int(np.asarray(Y).reshape(X.shape[0], -1).shape[1]))
                except NotImplementedError:
                    body = None
            if body is None:
                body = _selector_body(reference, x_path, y_path, idx_path,
                                      int(X.shape[0]), int(X.shape[1]))
                is_mask = body is not None
            if body is None:
                return None
            cases.append({"pred": out_path, "idx": idx_path,
                          "is_mask": is_mask, "p": int(X.shape[1])})
            script_parts.append("run_one(local({\n")
            script_parts.append(body)
            script_parts.append(f"\n}}), {'TRUE' if is_timed else 'FALSE'})\n")
            if is_timed:
                last_pred_path = out_path
                last_idx_path = idx_path
        samples_path = tmp / "samples.csv"
        script_parts.append(
            f"write.table(matrix(samples, ncol=1), file={_r_quote(samples_path)}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n")
        script_path = tmp / "batch.R"
        script_path.write_text("".join(script_parts), encoding="utf-8")
        proc = subprocess.run([RSCRIPT, "--vanilla", str(script_path)],
                              capture_output=True, text=True, env=R_ENV,
                              timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(
                f"batched R reference failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-500:]}")
        samples = np.atleast_1d(np.loadtxt(samples_path, delimiter=","))
        samples_f = [float(x) for x in samples]
        last = cases[-1]
        if last["is_mask"]:
            last_preds = _mask_from_indices(last_idx_path or last["idx"],
                                            int(last["p"]))
        else:
            last_preds = _read_csv_array(last_pred_path or last["pred"])
        return _stats_from_samples(samples_f), last_preds


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

    try:
        batched = _time_r_reference_batched(
            reference, method, params, csv_dir, n, p, runs, seed_base)
    except Exception as exc:
        print(json.dumps({"ok": False,
                          "reason": f"batched R reference error: {exc}",
                          "skipped": True}))
        return

    if batched is None:
        def fit_predict(seed: int) -> np.ndarray:
            X, y = load_dataset(csv_dir, n, p, seed)
            Y, extras = benchmark_inputs(method, X, y, params, seed)
            reference.fit(X, Y, **extras)
            return np.asarray(reference.predict(X))

        stats, last_preds = time_runs_seeded(fit_predict, runs, seed_base)
    else:
        stats, last_preds = batched
    versions = collect_versions(
        "Python" if reference.language == "python" else reference.language,
        numpy=_safe_version("numpy"),
        reference_id=ref_id,
        reference_library=reference.library_name,
        reference_version=reference.library_version,
    )
    if batched is not None:
        versions["timing_driver"] = "rscript-batch"
    emit_record(stats, last_preds, pred_path, versions)


if __name__ == "__main__":
    main()
