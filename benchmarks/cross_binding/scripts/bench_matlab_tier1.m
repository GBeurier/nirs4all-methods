% MATLAB / Octave tier-1 dispatcher — registry-driven.
%
% Routes through `pls4all.n4m_method_fit_mex(algo, X, Y, n_components,
% params_struct)`, the MEX-level dispatcher that the 66 per-algo MATLAB
% wrappers wrap. Per-algo params arrive as a JSON string in env
% `BENCH_MATLAB_PARAMS_JSON` (built by the orchestrator from the same
% `adapted_params` helper Python uses). Deterministic extras
% (labels / sample_weights / group_assignment / X_target) are reproduced
% from X / y / p / seed so we don't depend on Python's PCG64.

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

if exist("maxNumCompThreads", "builtin") || exist("maxNumCompThreads", "file")
    try
        maxNumCompThreads(threads);
    catch
        % maxNumCompThreads may not honour > physical cores; ignore.
    end
end

% Parse params from JSON env var.
if isempty(params_json)
    params = struct();
else
    params = jsondecode(params_json);
end

% adapted_params on the Python side may have clamped n_components below
% the argv value (classifier algos cap it to n_classes - 1, n_pls is
% capped by the (j, k) factor pair). Prefer the value from JSON params.
if isfield(params, "n_components")
    nc = double(params.n_components);
end

% Algo aliasing — the MEX dispatcher needs the matching C kernel name.
% "pls" runs through sparse_simpls(lambda=0); kernel_pls_rbf is
% kernel_pls with KernelType=1. "pcr" requires the Model API which is
% out of scope for the MEX dispatcher — fail the cell cleanly.
dispatch_algo = algo;
if strcmp(algo, "pls")
    dispatch_algo = "sparse_simpls";
    if ~isfield(params, "sparsity_lambda")
        params.sparsity_lambda = 0.0;
    end
elseif strcmp(algo, "kernel_pls_rbf")
    dispatch_algo = "kernel_pls";
    if ~isfield(params, "kernel_type")
        params.kernel_type = int32(1);  % RBF
    end
elseif strcmp(algo, "approximate_press")
    dispatch_algo = "approximate_press_compute";
    if isfield(params, "max_components")
        nc = double(params.max_components);
    end
elseif strcmp(algo, "one_se_rule")
    dispatch_algo = "one_se_rule_compute";
    if isfield(params, "max_components")
        nc = double(params.max_components);
    end
elseif strcmp(algo, "pls_diagnostic_t2") || strcmp(algo, "pls_diagnostic_q") || strcmp(algo, "pls_diagnostic_dmodx")
    dispatch_algo = "pls_diagnostics_compute";
elseif strcmp(algo, "pls_monitoring")
    dispatch_algo = "pls_monitoring_run";
elseif strcmp(algo, "variable_select_vip")
    dispatch_algo = "variable_select_rank";
    params.method = int32(0);
elseif strcmp(algo, "variable_select_coef")
    dispatch_algo = "variable_select_rank";
    params.method = int32(1);
elseif strcmp(algo, "variable_select_sr")
    dispatch_algo = "variable_select_rank";
    params.method = int32(2);
end
% PCR / OPLS need the Model API on the MATLAB side too — the MEX
% dispatcher doesn't expose pcr_svd / opls_nipals as method names.
% pls4all.PcrRegression / pls4all.OplsRegression wrap the same C
% kernel via the model entry point.
use_model_api = false;
model_api_cls = '';
if strcmp(algo, "pcr")
    use_model_api = true;
    model_api_cls = 'pls4all.PcrRegression';
elseif strcmp(algo, "opls")
    use_model_api = true;
    model_api_cls = 'pls4all.OplsRegression';
end

% Registry uses `interval_step`; C dispatcher reads `step`.
if isfield(params, "interval_step")
    params.step = int32(params.interval_step);
    params = rmfield(params, "interval_step");
end

% Strip Python-only metadata keys so the C dispatcher's per-algo
% whitelists don't reject them.
for key = {"n_samples", "n_features"}
    name = key{1};
    if isfield(params, name)
        params = rmfield(params, name);
    end
end

% Integer coercion: jsondecode of a JSON array of ints yields a double
% column vector, which the MEX side rejects when it expects int32.
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

% Block-sizes / similar vector params can land as column vectors after
% jsondecode — the MEX dispatcher prefers row vectors.
for k = {"block_sizes", "n_components_per_block", "n_unique_per_block", ...
          "y_labels", "group_assignment"}
    name = k{1};
    if isfield(params, name)
        params.(name) = reshape(params.(name), 1, []);
    end
end

function preds = fit_predict(dispatch_algo, algo, csv_dir, n, p, nc, ...
                              params, seed, needs_labels, needs_sw, ...
                              needs_groups, needs_x_target, x_target_dir, ...
                              pkey, use_model_api, model_api_cls)
    [X, y] = pls4all_bench_load_xy(csv_dir, n, p, seed);

    if use_model_api
        % PCR / OPLS via the classdef wrapper around the model API.
        ctor = str2func(model_api_cls);
        mdl = ctor(double(X), double(y), int32(nc));
        preds = double(reshape(predict(mdl, double(X)), [], 1));
        return;
    end

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

    cmps = {"pls_lda", "pls_logistic", "pls_qda", "sparse_pls_da", ...
             "pls_cox", "ds", "pds"};
    is_classifier_like = false;
    for ck = 1:numel(cmps)
        if strcmp(algo, cmps{ck})
            is_classifier_like = true;
            break;
        end
    end
    Y = y(:);
    if is_classifier_like
        Y = zeros(numel(y), 1);
    end
    % Multi-target methods (so_pls / rosa) need an n×n_targets matrix
    % matching `benchmark_inputs` on the Python side (column 0 = y,
    % column j>0 = 0.5*y + 0.1*X[:, (j-1) mod p]).
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

    res = pls4all.n4m_method_fit_mex(char(dispatch_algo), ...
                                       double(X_call), double(Y_call), ...
                                       int32(nc), params);

    % Honour registry prediction_key so classifiers return decision
    % scores (not labels) for parity comparison. Return matrices intact
    % (don't flatten with `:`) — write_npy_f64 transposes 2-D arrays
    % before serialising so the row-major flat-equivalent matches Python.
    if any(strcmp(pkey, {"selected_indices", "mask", "support"}))
        if isfield(res, "mask")
            preds = double(reshape(res.mask, 1, []));
            return;
        end
        if isfield(res, "support")
            preds = double(reshape(res.support, 1, []));
            return;
        end
        if isfield(res, "selected_indices")
            sel = double(res.selected_indices(:));
            mask = zeros(1, p);
            valid = sel(sel > 0 & sel <= p);
            if ~isempty(valid)
                mask(valid) = 1.0;
            end
            preds = mask;
            return;
        end
        preds = zeros(1, p);
        return;
    end
    if strcmp(pkey, "decision_scores") && isfield(res, "decision_scores")
        preds = double(res.decision_scores);
        return;
    end
    if isfield(res, char(pkey))
        preds = double(res.(char(pkey)));
        return;
    end
    if isfield(res, "predictions")
        preds = double(res.predictions);
        return;
    end
    if isfield(res, "selected_indices")
        sel = double(res.selected_indices(:));
        mask = zeros(1, p);
        valid = sel(sel > 0 & sel <= p);
        if ~isempty(valid)
            mask(valid) = 1.0;
        end
        preds = mask;
        return;
    end
    error("pls4all:bench", ...
        "matlab_tier1: %s returned no '%s' field", algo, pkey);
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
    @(s) fit_predict(dispatch_algo, algo, csv_dir, n, p, nc, params, ...
                      s, needs_labels, needs_sw, needs_groups, ...
                      needs_x_target, x_target_dir, registry_pkey, ...
                      use_model_api, model_api_cls), ...
    runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libn4m-linked MEX";
versions.registry_method = algo;
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
