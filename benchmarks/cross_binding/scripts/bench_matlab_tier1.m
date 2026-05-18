% MATLAB / Octave tier-1 dispatcher — registry-driven.
%
% Routes through `pls4all.p4a_method_fit_mex(algo, X, Y, n_components,
% params_struct)`, the MEX-level dispatcher that the 66 per-algo MATLAB
% wrappers wrap. Per-algo params arrive as a JSON string in env
% `BENCH_MATLAB_PARAMS_JSON` (built by the orchestrator from the same
% `adapted_params` helper Python uses). Deterministic extras
% (labels / sample_weights / group_assignment) are reproduced from y / p
% so we don't depend on Python's PCG64. X_target-needing methods
% (di_pls / ds / pds) report a clear "needs X_target sidecar" failure.

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
needs_x_target = strcmp(getenv("BENCH_R_NEEDS_X_TARGET"), "1");
needs_labels   = strcmp(getenv("BENCH_R_NEEDS_LABELS"), "1");
needs_sw       = strcmp(getenv("BENCH_R_NEEDS_SAMPLE_WEIGHTS"), "1");
needs_groups   = strcmp(getenv("BENCH_R_NEEDS_GROUP_ASSIGNMENT"), "1");

if exist("maxNumCompThreads", "builtin") || exist("maxNumCompThreads", "file")
    try
        maxNumCompThreads(threads);
    catch
        % maxNumCompThreads may not honour > physical cores; ignore.
    end
end

if needs_x_target
    error("pls4all:bench", ...
        "matlab_tier1: %s needs X_target sidecar (not yet wired)", algo);
end

% Parse params from JSON env var.
if isempty(params_json)
    params = struct();
else
    params = jsondecode(params_json);
end

% Algo aliasing — the C dispatcher named "pls" with sparse_simpls
% (sparsity_lambda=0.0) and "pcr" with pcr_svd.
dispatch_algo = algo;
if strcmp(algo, "pls")
    dispatch_algo = "sparse_simpls";
    if ~isfield(params, "sparsity_lambda")
        params.sparsity_lambda = 0.0;
    end
elseif strcmp(algo, "pcr")
    dispatch_algo = "pcr_svd";
end

% Strip Python-only metadata keys so the C dispatcher's per-algo
% whitelists don't reject them.
for key = ["n_samples", "n_features"]
    if isfield(params, key)
        params = rmfield(params, key);
    end
end

% Integer coercion: jsondecode of a JSON array of ints yields a double
% column vector, which the MEX side rejects when it expects int32.
int_keys = ["block_sizes", "n_components_per_block", "n_unique_per_block", ...
             "y_labels", "group_assignment", "n_blocks", ...
             "max_iter", "max_irls_iter", "n_estimators", ...
             "n_neighbors", "n_iterations", "window_size", ...
             "initial_intervals", "top_k", "min_selected", ...
             "seed", "n_classes", "n_predictive", ...
             "n_x_orthogonal", "n_y_orthogonal", ...
             "features_per_subspace", "n_steps", "n_folds", ...
             "noise_features", "noise_seed", "kernel_type", ...
             "degree", "n_targets", "mode_j", "mode_k", ...
             "window_half_width"];
for k = int_keys
    if isfield(params, k)
        params.(k) = int32(params.(k));
    end
end

% Block-sizes / similar vector params can land as column vectors after
% jsondecode — the MEX dispatcher prefers row vectors.
for k = ["block_sizes", "n_components_per_block", "n_unique_per_block", ...
          "y_labels", "group_assignment"]
    if isfield(params, k)
        params.(k) = reshape(params.(k), 1, []);
    end
end

function preds = fit_predict(dispatch_algo, algo, csv_dir, n, p, nc, ...
                              params, seed, needs_labels, needs_sw, ...
                              needs_groups)
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

    % Some classifier algos repurpose Y as a label placeholder; the MEX
    % dispatcher rebuilds the right Y on the C side using the labels we
    % stuck in params.
    Y = y(:);
    if any(strcmp(algo, ["pls_lda", "pls_logistic", "pls_qda", ...
                          "sparse_pls_da", "pls_cox", "ds", "pds"]))
        Y = zeros(numel(y), 1);
    end

    res = pls4all.p4a_method_fit_mex(char(dispatch_algo), ...
                                       double(X), double(Y), ...
                                       int32(nc), params);

    if isfield(res, "predictions")
        preds = double(res.predictions(:));
    elseif isfield(res, "selected_indices")
        sel = double(res.selected_indices(:));
        mask = zeros(1, p);
        valid = sel(sel > 0 & sel <= p);
        if ~isempty(valid)
            mask(valid) = 1.0;
        end
        preds = mask(:);
    else
        error("pls4all:bench", ...
            "matlab_tier1: %s returned neither predictions nor selected_indices", ...
            algo);
    end
end

[stats, last_preds] = pls4all_bench_run( ...
    @(s) fit_predict(dispatch_algo, algo, csv_dir, n, p, nc, params, ...
                      s, needs_labels, needs_sw, needs_groups), ...
    runs, seed_base);

versions = struct();
versions.language = sprintf("Octave %s", OCTAVE_VERSION);
versions.pls4all = "from libp4a-linked MEX";
versions.registry_method = algo;
versions.blas = "linked-BLAS";

pls4all_bench_emit(stats, last_preds, pred_path, versions);
