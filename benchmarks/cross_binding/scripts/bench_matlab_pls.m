% External: Octave statistics package `plsregress` (SIMPLS).
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

pkg load statistics

if ~strcmp(algo, "pls")
    % plsregress doesn't support sparse / cppls / etc. Emit skip.
    reason = strrep(sprintf("algo '%s' not implemented by Octave plsregress", ...
                            algo), '"', '\"');
    fprintf(['{"ok":false,"reason":"%s","skipped":true,' ...
             '"versions":{"language":"Octave %s",' ...
             '"statistics":"linked Octave statistics pkg"}}\n'], ...
            reason, OCTAVE_VERSION);
    return
end

function preds = fit_predict(csv_dir, n, p, nc, seed)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);
    [~, ~, ~, ~, BETA, ~, ~] = plsregress(X, y, nc);
    preds = [ones(size(X, 1), 1) X] * BETA;
    preds = preds(:);
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(csv_dir, n, p, nc, s), runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.statistics = "Octave statistics pkg (plsregress)";
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
