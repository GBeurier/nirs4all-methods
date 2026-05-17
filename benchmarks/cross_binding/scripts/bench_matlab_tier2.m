% MATLAB / Octave tier-2 dispatcher → pls4all.fit() factory.
algo       = getenv("BENCH_ALGO");
csv_dir    = getenv("BENCH_CSV_DIR");
n          = str2double(getenv("BENCH_N"));
p          = str2double(getenv("BENCH_P"));
nc         = str2double(getenv("BENCH_NCOMP"));
runs       = str2double(getenv("BENCH_NRUNS"));
seed_base  = str2double(getenv("BENCH_SEED_BASE"));
pred_path  = getenv("BENCH_PRED_PATH");
threads    = str2double(getenv("BENCH_THREADS"));
try
    maxNumCompThreads(threads);
catch
end

function preds = fit_predict(algo, csv_dir, n, p, nc, seed)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);
    switch algo
        case {"pls", "pls_simpls", "simpls"}
            mdl = pls4all.fit("pls", X, y, "NumComponents", nc);
        case "sparse_simpls"
            mdl = pls4all.fit("sparse_simpls", X, y, ...
                "NumComponents", nc, "Lambda", 0.05);
        case "cppls"
            mdl = pls4all.fit("cppls", X, y, "NumComponents", nc, "Gamma", 0.5);
        case "ecr"
            mdl = pls4all.fit("ecr", X, y, "NumComponents", nc, "Alpha", 0.5);
        case "mir_pls"
            mdl = pls4all.fit("mir_pls", X, y, "NumComponents", nc);
        case "ridge_pls"
            mdl = pls4all.fit("ridge_pls", X, y, "NumComponents", nc, "Lambda", 1.0);
        case "robust_pls"
            mdl = pls4all.fit("robust_pls", X, y, "NumComponents", nc, "HuberK", 1.345);
        case "missing_aware_nipals"
            mdl = pls4all.fit("missing_aware_nipals", X, y, "NumComponents", nc);
        case "continuum_regression"
            mdl = pls4all.fit("continuum_regression", X, y, "NumComponents", nc, "Tau", 0.5);
        case "bagging_pls"
            mdl = pls4all.fit("bagging_pls", X, y, "NumComponents", nc, "NEstimators", 50);
        case "boosting_pls"
            mdl = pls4all.fit("boosting_pls", X, y, "NumComponents", nc, "NEstimators", 50);
        otherwise
            error("pls4all:bench", "matlab_tier2: unsupported algo %s", algo);
    end
    preds = predict(mdl, X);
    preds = preds(:);
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(algo, csv_dir, n, p, nc, s), runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libp4a-linked MEX + classdefs";
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
