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
    fprintf("%s\n", jenc(struct( ...
        "ok", false, ...
        "reason", sprintf("algo '%s' not implemented by Octave plsregress", algo), ...
        "skipped", true, ...
        "versions", struct( ...
            "language", sprintf("Octave %s", OCTAVE_VERSION), ...
            "statistics", "linked Octave statistics pkg") ...
    )));
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

% jenc helper used only for the skip path (since _emit_octave provides its
% own JSON emitter, we inline a tiny one here for the early return).
function s = jenc(v)
    if islogical(v)
        if v, s = "true"; else, s = "false"; end
    elseif isnumeric(v) && isscalar(v)
        s = sprintf("%.15g", v);
    elseif ischar(v) || (isstring(v) && isscalar(v))
        s = sprintf("\"%s\"", strrep(char(v), '"', '\"'));
    elseif iscell(v) || (isnumeric(v) && ~isscalar(v))
        parts = arrayfun(@(x) jenc(x), v, "UniformOutput", false);
        s = sprintf("[%s]", strjoin(parts, ","));
    elseif isstruct(v)
        fn = fieldnames(v); parts = cell(1, length(fn));
        for k = 1:length(fn)
            parts{k} = sprintf("\"%s\":%s", fn{k}, jenc(v.(fn{k})));
        end
        s = sprintf("{%s}", strjoin(parts, ","));
    else, s = "null";
    end
end
