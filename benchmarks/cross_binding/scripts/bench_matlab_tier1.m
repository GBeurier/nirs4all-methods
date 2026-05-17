% MATLAB / Octave tier-1 dispatcher.
% Reads args from BENCH_* env vars (set by orchestrator).
algo       = getenv("BENCH_ALGO");
csv_dir    = getenv("BENCH_CSV_DIR");
n          = str2double(getenv("BENCH_N"));
p          = str2double(getenv("BENCH_P"));
nc         = str2double(getenv("BENCH_NCOMP"));
runs       = str2double(getenv("BENCH_NRUNS"));
seed_base  = str2double(getenv("BENCH_SEED_BASE"));
pred_path  = getenv("BENCH_PRED_PATH");
threads    = str2double(getenv("BENCH_THREADS"));

if exist("maxNumCompThreads", "builtin") || exist("maxNumCompThreads", "file")
    try
        maxNumCompThreads(threads);
    catch
        % maxNumCompThreads may not honour > physical cores; ignore.
    end
end

function preds = fit_predict(algo, csv_dir, n, p, nc, seed)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);
    switch algo
        case "pls"
            res = pls4all.sparse_simpls(X, y, nc, 0.0);
            preds = res.predictions(:);
        case "sparse_simpls"
            res = pls4all.sparse_simpls(X, y, nc, 0.05);
            preds = res.predictions(:);
        case "cppls"
            res = pls4all.cppls(X, y, nc, 0.5);
            preds = res.predictions(:);
        case "ecr"
            res = pls4all.ecr(X, y, nc, 0.5);
            preds = res.predictions(:);
        case "mir_pls"
            res = pls4all.mir_pls(X, y, nc);
            preds = res.predictions(:);
        case "ridge_pls"
            res = pls4all.ridge_pls(X, y, nc, 1.0);
            preds = res.predictions(:);
        case "robust_pls"
            res = pls4all.robust_pls(X, y, nc, 1.345, 20);
            preds = res.predictions(:);
        case "missing_aware_nipals"
            res = pls4all.missing_aware_nipals(X, y, nc);
            preds = res.predictions(:);
        otherwise
            error("pls4all:bench", "matlab_tier1: unsupported algo %s", algo);
    end
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(algo, csv_dir, n, p, nc, s), runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libp4a-linked MEX";
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
