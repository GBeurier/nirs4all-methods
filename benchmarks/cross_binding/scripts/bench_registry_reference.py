"""External reference backend driven by parity_timing.MethodSpec."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from _common import (parse_args, time_runs_seeded, emit_record,
                     load_dataset, collect_versions, _safe_version,
                     WARMUP_ABORT_MS)
from bench_registry_common import (adapted_params, benchmark_inputs,
                                   find_reference, load_method)


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


def _stats_from_samples(samples: list[float], *, statistic: str,
                        warmup_ms: float, decision: str,
                        max_runs: int, total_runs: int,
                        warmup_included: bool = False) -> dict:
    if len(samples) == 1:
        statistic = "single"
    reported = (sum(samples) / len(samples)
                if statistic == "mean" else float(np.median(samples)))
    return {
        "ok": True,
        "n_runs": len(samples),
        "median_ms": reported,
        "reported_ms": reported,
        "sample_median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "warmup_ms": warmup_ms,
        "warmup_included": warmup_included,
        "timing_statistic": statistic,
        "timing_decision": decision,
        "max_runs": max(1, int(max_runs)),
        "total_runs": max(1, int(total_runs)),
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
        if cls == "_IplsForwardReference":
            interval_setup = (
                f"starts <- seq(1L, ncol(X) - {int(reference._w)} + 1L, "
                f"by={int(reference._step)})\n"
                f"limits <- cbind(starts, starts + {int(reference._w)} - 1L)\n"
                "extra_args <- list()\n"
            )
        else:
            min_intervals = int(getattr(reference, "_min_intervals", 1))
            interval_setup = (
                f"starts <- seq(1L, ncol(X), by={int(reference._w)})\n"
                f"limits <- cbind(starts, pmin(starts + {int(reference._w)} - 1L, ncol(X)))\n"
                f"extra_args <- list(int.niter=max(1L, nrow(limits) - {min_intervals}), "
                "full=TRUE)\n"
            )
        return (
            "local({\n"
            "suppressPackageStartupMessages(library(mdatools))\n"
            f"X <- as.matrix(read.csv({xq}, header=FALSE))\n"
            f"y <- as.numeric(scan({yq}, quiet=TRUE))\n"
            f"{interval_setup}"
            f"args <- c(list(x=X, y=y, glob.ncomp={int(reference._k)}, "
            f"int.limits=limits, method='{method}', "
            "cv=list('ven', 3), silent=TRUE), extra_args)\n"
            "res <- do.call(ipls, args)\n"
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
        max_runs = max(1, int(runs))
        cases: list[dict] = []
        script_parts = [
            "now_ms <- function() as.numeric(Sys.time()) * 1000.0\n",
            "time_call <- function(fn) {\n",
            "  t0 <- now_ms()\n",
            "  fn()\n",
            "  now_ms() - t0\n",
            "}\n",
            "adaptive_target <- function(probe_ms, max_runs) {\n",
            "  max_runs <- max(2L, as.integer(max_runs))\n",
            "  if (probe_ms > 60000.0) return(list(target=2L, statistic='single', decision='probe_gt_60s'))\n",
            "  if (probe_ms > 30000.0) return(list(target=min(max_runs, 2L), statistic='single', decision='probe_gt_30s'))\n",
            "  if (probe_ms > 5000.0) return(list(target=min(max_runs, 3L), statistic='mean', decision='probe_gt_5s'))\n",
            "  if (probe_ms > 1000.0) return(list(target=min(max_runs, 10L), statistic='median', decision='probe_gt_1s'))\n",
            "  if (probe_ms > 100.0) return(list(target=min(max_runs, 20L), statistic='median', decision='probe_gt_100ms'))\n",
            "  list(target=min(max_runs, 40L), statistic='median', decision='probe_le_100ms')\n",
            "}\n",
        ]
        for idx, seed in enumerate([seed_base] + [
                seed_base + i for i in range(max_runs)]):
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
            script_parts.append(f"case_{idx:03d} <- function() {{\n")
            script_parts.append(body)
            script_parts.append("\n}\n")
        samples_path = tmp / "samples.csv"
        meta_path = tmp / "timing_meta.csv"
        case_names = ", ".join(f"case_{idx:03d}" for idx in range(len(cases)))
        script_parts.append(
            f"case_fns <- list({case_names})\n"
            f"max_runs <- {max_runs}L\n"
            "samples <- numeric(0)\n"
            "last_case <- 0L\n"
            "warmup_included <- FALSE\n"
            "warmup_ms <- time_call(case_fns[[1L]])\n"
            f"if (warmup_ms > {WARMUP_ABORT_MS:.1f}) {{\n"
            "  samples <- c(warmup_ms)\n"
            "  timing_statistic <- 'single'\n"
            "  timing_decision <- 'warmup_gt_5min'\n"
            "  warmup_included <- TRUE\n"
            "} else if (max_runs < 2L) {\n"
            "  samples <- c(warmup_ms)\n"
            "  timing_statistic <- 'single'\n"
            "  timing_decision <- 'max_runs_1_warmup_only'\n"
            "  warmup_included <- TRUE\n"
            "} else {\n"
            "  samples <- c(time_call(case_fns[[2L]]))\n"
            "  last_case <- 1L\n"
            "  target <- adaptive_target(samples[[1]], max_runs)\n"
            "  timing_statistic <- target$statistic\n"
            "  timing_decision <- target$decision\n"
            "  target_samples <- max(1L, target$target - 1L)\n"
            "  if (target_samples > 1L) {\n"
            "    for (i in 2:target_samples) {\n"
            "      case_idx <- i + 1L\n"
            "      samples <- c(samples, time_call(case_fns[[case_idx]]))\n"
            "      last_case <- i\n"
            "    }\n"
            "  }\n"
            "}\n"
            f"write.table(matrix(samples, ncol=1), file={_r_quote(samples_path)}, "
            "sep=',', row.names=FALSE, col.names=FALSE)\n"
            "meta <- data.frame(warmup_ms=warmup_ms, last_case=last_case,\n"
            "                   timing_statistic=timing_statistic,\n"
            "                   timing_decision=timing_decision,\n"
            "                   warmup_included=warmup_included,\n"
            "                   total_runs=ifelse(warmup_included, 1L, 1L + length(samples)),\n"
            "                   max_runs=max_runs)\n"
            f"write.table(meta, file={_r_quote(meta_path)}, sep=',', "
            "row.names=FALSE, col.names=TRUE)\n")
        script_path = tmp / "batch.R"
        script_path.write_text("".join(script_parts), encoding="utf-8")
        timeout_s = int(os.environ.get("BENCH_CELL_TIMEOUT", "900"))
        proc = subprocess.run([RSCRIPT, "--vanilla", str(script_path)],
                              capture_output=True, text=True, env=R_ENV,
                              timeout=timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(
                f"batched R reference failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-500:]}")
        samples = np.atleast_1d(np.loadtxt(samples_path, delimiter=","))
        samples_f = [float(x) for x in samples]
        with meta_path.open() as f:
            meta_rows = list(__import__("csv").DictReader(f))
        meta = meta_rows[0] if meta_rows else {}
        # Parity prediction is always the seed_base case (case 0), so it is
        # comparable across backends regardless of how many adaptive timing
        # runs executed. (`last_case` in the meta only drives timing stats.)
        parity_case = cases[0]
        if parity_case["is_mask"]:
            parity_preds = _mask_from_indices(parity_case["idx"],
                                              int(parity_case["p"]))
        else:
            parity_preds = _read_csv_array(parity_case["pred"])
        stats = _stats_from_samples(
            samples_f,
            statistic=str(meta.get("timing_statistic") or "median"),
            warmup_ms=float(meta.get("warmup_ms") or 0.0),
            decision=str(meta.get("timing_decision") or "unknown"),
            max_runs=int(float(meta.get("max_runs") or max_runs)),
            total_runs=int(float(meta.get("total_runs") or len(samples_f))),
            warmup_included=str(meta.get("warmup_included") or "").lower()
            == "true",
        )
        stats["prediction_seed"] = int(seed_base)
        return stats, parity_preds


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
