% MATLAB / Octave tier-2 dispatcher — registry-driven via pls4all.fit().
%
% pls4all.fit owns the classdef surface for regressors and now falls back to
% GenericMethodResult for selectors, diagnostics, transformers and other
% MethodResult kernels. The benchmark still prepares the same registry
% extras as tier 1 so binding parity compares like with like.

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
x_target_dir = getenv("BENCH_R_X_TARGET_DIR");
needs_x_target = strcmp(getenv("BENCH_R_NEEDS_X_TARGET"), "1");
needs_labels   = strcmp(getenv("BENCH_R_NEEDS_LABELS"), "1");
needs_sw       = strcmp(getenv("BENCH_R_NEEDS_SAMPLE_WEIGHTS"), "1");
needs_groups   = strcmp(getenv("BENCH_R_NEEDS_GROUP_ASSIGNMENT"), "1");
registry_pkey  = getenv("BENCH_PREDICTION_KEY");
if isempty(registry_pkey)
    registry_pkey = "predictions";
end

try
    maxNumCompThreads(threads);
catch
end

if isempty(params_json)
    params = struct();
else
    params = jsondecode(params_json);
end

if isfield(params, "n_components")
    nc = double(params.n_components);
end

if isfield(params, "interval_step")
    params.step = int32(params.interval_step);
    params = rmfield(params, "interval_step");
end
for key = {"n_samples", "n_features"}
    name = key{1};
    if isfield(params, name)
        params = rmfield(params, name);
    end
end

int_keys = {"block_sizes", "n_components_per_block", "n_unique_per_block", ...
             "y_labels", "group_assignment", "n_blocks", ...
             "max_iter", "max_irls_iter", "n_estimators", ...
             "n_neighbors", "n_iterations", "window_size", ...
             "initial_intervals", "top_k", "min_selected", ...
             "seed", "n_classes", "n_predictive", ...
             "n_x_orthogonal", "n_y_orthogonal", ...
             "features_per_subspace", "n_steps", "n_folds", ...
             "noise_features", "noise_seed", "kernel_type", ...
             "degree", "n_targets", "mode_j", "mode_k", ...
             "window_half_width"};
for k = int_keys
    name = k{1};
    if isfield(params, name)
        params.(name) = int32(params.(name));
    end
end
for k = {"block_sizes", "n_components_per_block", "n_unique_per_block", ...
          "y_labels", "group_assignment"}
    name = k{1};
    if isfield(params, name)
        params.(name) = reshape(params.(name), 1, []);
    end
end
function preds = fit_predict(algo, csv_dir, n, p, nc, seed, params, ...
                              x_target_dir, needs_labels, needs_sw, ...
                              needs_groups, needs_x_target, pkey)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);

    if needs_labels
        n_classes = 2;
        if isfield(params, "n_classes")
            n_classes = double(params.n_classes);
        end
        [~, ord] = sort(y);
        labels = zeros(1, n, "int32");
        labels(ord) = int32(mod(0:n-1, n_classes));
        params.y_labels = labels;
    end
    if needs_sw
        params.sample_weights = double(abs(y(:)) + 0.5)';
    end
    if needs_groups
        n_groups = min(5, max(1, p));
        width = max(1, ceil(p / n_groups));
        ga = min(int32(floor((0:p-1) / width)), int32(n_groups - 1));
        params.group_assignment = ga;
    end
    if needs_x_target
        params.X_target = pls4all_bench_load_x_target(x_target_dir, n, p, seed);
    end
    if strcmp(algo, "pls_cox")
        if ~isfield(params, "sample_weights")
            params.sample_weights = double(abs(y(:)) + 0.5)';
        end
        if ~isfield(params, "y_labels")
            [~, ord] = sort(y);
            labels = zeros(1, n, "int32");
            labels(ord) = int32(mod(0:n-1, 2));
            params.y_labels = labels;
        end
        params.survival_times = double(reshape(params.sample_weights, 1, []));
        params.event_indicators = int32(reshape(params.y_labels > 0, 1, []));
    end

    classifier_like = {"pls_lda", "pls_logistic", "pls_qda", ...
                       "sparse_pls_da", "pls_cox", "ds", "pds"};
    is_classifier_like = any(strcmp(algo, classifier_like));
    Y = y(:);
    if is_classifier_like
        Y = zeros(numel(y), 1);
    end
    n_targets = 1;
    if isfield(params, "n_targets")
        n_targets = double(params.n_targets);
    end
    if n_targets > 1 && ~is_classifier_like
        cols = zeros(numel(y), n_targets);
        cols(:, 1) = y(:);
        for j = 1:(n_targets - 1)
            col_idx = mod(j - 1, p) + 1;
            cols(:, j + 1) = 0.5 * y(:) + 0.1 * X(:, col_idx);
        end
        Y = cols;
    end

    dispatch_algo = algo;
    if strcmp(algo, "pls")
        dispatch_algo = "sparse_simpls";
        if ~isfield(params, "sparsity_lambda")
            params.sparsity_lambda = 0.0;
        end
    elseif strcmp(algo, "kernel_pls_rbf")
        dispatch_algo = "kernel_pls";
        if ~isfield(params, "kernel_type")
            params.kernel_type = int32(1);
        end
    end

    X_call = X;
    Y_call = Y;
    if strcmp(algo, "one_se_rule")
        max_components = nc;
        if isfield(params, "max_components")
            max_components = double(params.max_components);
        end
        n_folds = 5;
        if isfield(params, "n_folds")
            n_folds = double(params.n_folds);
        end
        params.fold_rmse_matrix = pls4all_bench_fold_rmse(max_components, n_folds);
        X_call = zeros(2, 1);
        Y_call = zeros(2, 1);
    elseif strcmp(algo, "pls_monitoring")
        split = floor(size(X, 1) / 2);
        params.X_monitor = double(X((split + 1):end, :));
        X_call = X(1:split, :);
        Y_call = Y(1:split, :);
    end

    % Item #21: cppls/ridge_pls/sparse_simpls need the per-method parity
    % convention (solver / scale_x) the orchestrator injected into params.
    % The concrete tier-2 wrapper classes (CpplsRegression / RidgePlsRegression
    % / SparsePlsRegression) rebuild params from lambda/gamma only and drop it,
    % so call the MEX directly with the full params struct — the dispatcher
    % reads solver/scale_x/scale_y from it. Mirrors the python_tier2 route.
    if any(strcmp(algo, {"cppls", "ridge_pls", "sparse_simpls"}))
        res = pls4all.n4m_method_fit_mex(char(dispatch_algo), X_call, Y_call, ...
                                         int32(nc), params);
        preds = res.(char(pkey));
        return;
    end

    nv = pls4all_bench_name_values(algo, nc, params);
    nv_run = [nv, {"Params", params, "PredictionKey", char(pkey)}];
    mdl = pls4all.fit(char(dispatch_algo), X_call, Y_call, nv_run{:});
    preds = predict(mdl, X_call);
end

function nv = pls4all_bench_name_values(algo, nc, params)
    nv = {"NumComponents", nc};
    if isfield(params, "sparsity_lambda")
        nv = [nv, {"Lambda", double(params.sparsity_lambda)}];
    elseif isfield(params, "ridge_lambda")
        nv = [nv, {"Lambda", double(params.ridge_lambda)}];
    end
    if strcmp(algo, "pls")
        nv = [nv, {"Lambda", 0.0}];
    end
    if isfield(params, "gamma")
        nv = [nv, {"Gamma", double(params.gamma)}];
    end
    if isfield(params, "alpha")
        nv = [nv, {"Alpha", double(params.alpha)}];
    end
    if isfield(params, "di_lambda")
        nv = [nv, {"DiLambda", double(params.di_lambda)}];
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
    elseif strcmp(algo, "kernel_pls_rbf")
        nv = [nv, {"KernelType", 1}];
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
    if isfield(params, "X_target")
        nv = [nv, {"XTarget", double(params.X_target)}];
    end
    if isfield(params, "sample_weights")
        nv = [nv, {"Weights", double(reshape(params.sample_weights, [], 1))}];
    end
    if isfield(params, "poisson") && double(params.poisson) ~= 0
        nv = [nv, {"Family", "poisson"}];
    end
end

function fr = pls4all_bench_fold_rmse(max_components, n_folds)
    idx = reshape(0:(max_components * n_folds - 1), n_folds, max_components)';
    fr = 0.5 + 0.5 * mod(idx * 37 + 11, 997) / 996;
end

function Xt = pls4all_bench_load_x_target(x_target_dir, n, p, seed)
    path = fullfile(x_target_dir, sprintf("xtarget_%dx%d_seed%d.csv", n, p, seed));
    if exist(path, "file") ~= 2
        error("pls4all:bench", "X_target sidecar not found: %s", path);
    end
    Xt = dlmread(path, ",");
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(algo, csv_dir, n, p, nc, s, params, ...
                     x_target_dir, needs_labels, needs_sw, needs_groups, ...
                     needs_x_target, registry_pkey), ...
    runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libn4m-linked MEX + classdefs";
versions.registry_method = algo;
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
