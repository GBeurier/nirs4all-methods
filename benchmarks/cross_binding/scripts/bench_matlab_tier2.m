% MATLAB / Octave tier-2 dispatcher — registry-driven via pls4all.fit().
%
% `pls4all.fit` is the unified classdef factory; it covers ~19 of the
% canonical 71 methods (regressors only — no selectors, no diagnostics,
% no transformers). For the rest the cell records `ok=false` with a
% clear "no classdef factory entry" reason so the dashboard shows `—`
% honestly.

algo       = getenv("BENCH_ALGO");
csv_dir    = getenv("BENCH_CSV_DIR");
n          = str2double(getenv("BENCH_N"));
p          = str2double(getenv("BENCH_P"));
nc         = str2double(getenv("BENCH_NCOMP"));
runs       = str2double(getenv("BENCH_NRUNS"));
seed_base  = str2double(getenv("BENCH_SEED_BASE"));
pred_path  = getenv("BENCH_PRED_PATH");
threads    = str2double(getenv("BENCH_THREADS"));
params_json= getenv("BENCH_MATLAB_PARAMS_JSON");
try
    maxNumCompThreads(threads);
catch
end

% Algos covered by pls4all.fit (classdef factory). Everything else is
% reported as "not bound" — the dashboard shows that cell as `—`.
SUPPORTED = {"pls", "pls_simpls", "simpls", "sparse_simpls", "cppls", ...
             "ecr", "weighted_pls", "robust_pls", "ridge_pls", ...
             "continuum_regression", "recursive_pls", "n_pls", ...
             "kernel_pls", "kernel_pls_rbf", "o2pls", "mb_pls", ...
             "pls_glm", "mir_pls", "missing_aware_nipals", ...
             "bagging_pls", "boosting_pls"};
matched = false;
for k = 1:numel(SUPPORTED)
    if strcmp(algo, SUPPORTED{k})
        matched = true;
        break;
    end
end
if ~matched
    error("pls4all:bench", ...
        "matlab_tier2: no pls4all.fit classdef entry for '%s'", algo);
end

if isempty(params_json)
    params = struct();
else
    params = jsondecode(params_json);
end

% adapted_params may have clamped n_components below argv (classifier
% algos cap at n_classes - 1, n_pls bounded by (j, k)).
if isfield(params, "n_components")
    nc = double(params.n_components);
end

% pls4all.fit (classdef factory) only accepts a 1-D y. Methods whose
% canonical cell uses multi-target Y (n_targets > 1) cannot run
% faithfully through tier-2; skip them with a clean reason.
n_targets = 1;
if isfield(params, "n_targets")
    n_targets = double(params.n_targets);
end
if n_targets > 1
    error("pls4all:bench", ...
        "matlab_tier2: %s uses n_targets=%d; pls4all.fit only accepts 1-D y", ...
        algo, n_targets);
end

% Map registry param keys to pls4all.fit Name-Value params.
nv = {"NumComponents", nc};
if isfield(params, "sparsity_lambda")
    nv = [nv, {"Lambda", double(params.sparsity_lambda)}];
elseif isfield(params, "ridge_lambda")
    nv = [nv, {"Lambda", double(params.ridge_lambda)}];
end
if isfield(params, "gamma")
    nv = [nv, {"Gamma", double(params.gamma)}];
end
if isfield(params, "alpha")
    nv = [nv, {"Alpha", double(params.alpha)}];
end
if isfield(params, "tau")
    nv = [nv, {"Tau", double(params.tau)}];
end
if isfield(params, "huber_k")
    nv = [nv, {"HuberK", double(params.huber_k)}];
end
if isfield(params, "max_irls_iter")
    nv = [nv, {"MaxIrlsIter", double(params.max_irls_iter)}];
end
if isfield(params, "window_size")
    nv = [nv, {"WindowSize", double(params.window_size)}];
end
if isfield(params, "mode_j")
    nv = [nv, {"ModeJ", double(params.mode_j)}];
end
if isfield(params, "mode_k")
    nv = [nv, {"ModeK", double(params.mode_k)}];
end
if isfield(params, "n_estimators")
    nv = [nv, {"NEstimators", double(params.n_estimators)}];
end
if isfield(params, "learning_rate")
    nv = [nv, {"LearningRate", double(params.learning_rate)}];
end
if isfield(params, "seed")
    nv = [nv, {"Seed", double(params.seed)}];
end
if isfield(params, "features_per_subspace")
    nv = [nv, {"FeaturesPerSubspace", double(params.features_per_subspace)}];
end
if isfield(params, "n_predictive")
    nv = [nv, {"NPredictive", double(params.n_predictive)}];
end
if isfield(params, "n_x_orthogonal")
    nv = [nv, {"NXOrthogonal", double(params.n_x_orthogonal)}];
end
if isfield(params, "n_y_orthogonal")
    nv = [nv, {"NYOrthogonal", double(params.n_y_orthogonal)}];
end
if isfield(params, "kernel_type")
    nv = [nv, {"KernelType", double(params.kernel_type)}];
end
if isfield(params, "coef0")
    nv = [nv, {"Coef0", double(params.coef0)}];
end
if isfield(params, "degree")
    nv = [nv, {"Degree", double(params.degree)}];
end
if isfield(params, "block_sizes")
    nv = [nv, {"BlockSizes", double(reshape(params.block_sizes, 1, []))}];
end

% pls4all.fit doesn't accept Weights from JSON params (weighted_pls
% derives sample_weights from y); we build it inline below.

function preds = fit_predict(algo, csv_dir, n, p, nc, seed, nv)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);
    % Reproduce the registry's per-method extras locally.
    if strcmp(algo, "weighted_pls")
        w = double(abs(y(:)) + 0.5);
        nv = [nv, {"Weights", w}];
    end
    % kernel_pls_rbf is just kernel_pls with KernelType=1 (rbf in the
    % registry convention). Strip any earlier KernelType so we don't
    % append it twice and trip the input parser.
    dispatch_algo = algo;
    if strcmp(algo, "kernel_pls_rbf")
        dispatch_algo = "kernel_pls";
        % Remove any previously-appended KernelType pair.
        keep = true(1, numel(nv));
        i = 1;
        while i <= numel(nv) - 1
            if ischar(nv{i}) || isstring(nv{i})
                if strcmp(string(nv{i}), "KernelType")
                    keep(i) = false; keep(i+1) = false;
                end
            end
            i = i + 2;
        end
        nv = nv(keep);
        nv = [nv, {"KernelType", 1}];
    end
    mdl = pls4all.fit(char(dispatch_algo), X, y, nv{:});
    preds = predict(mdl, X);
    preds = preds(:);
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(algo, csv_dir, n, p, nc, s, nv), runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libp4a-linked MEX + classdefs";
versions.registry_method = algo;
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
