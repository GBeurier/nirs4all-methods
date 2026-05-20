#!/usr/bin/env julia
# SPDX-License-Identifier: CECILL-2.1

using DelimitedFiles
using Pls4all

const WARMUP_ABORT_MS = 5 * 60 * 1000.0
const PROBE_ABORT_MS = 60 * 1000.0
const DEFAULT_MAX_RUNS = 40

if length(ARGS) != 8
    println(stderr, "usage: bench_julia_pls4all.jl algo csv_dir n p n_components max_total_runs seed_base predictions_path")
    exit(2)
end

const ALGO = ARGS[1]
const CSV_DIR = ARGS[2]
const N_SAMPLES = parse(Int, ARGS[3])
const N_FEATURES = parse(Int, ARGS[4])
const N_COMPONENTS_ARG = parse(Int, ARGS[5])
const MAX_RUNS = parse(Int, ARGS[6])
const SEED_BASE = parse(Int, ARGS[7])
const PRED_PATH = ARGS[8]
const PREDICTION_KEY = get(ENV, "BENCH_PREDICTION_KEY", "predictions")

const ALGO_PLS_REGRESSION = 0
const ALGO_OPLS = 4
const ALGO_PCR = 10
const SOLVER_NIPALS = 0
const SOLVER_SIMPLS = 1
const SOLVER_SVD = 5
const DEFLATION_REGRESSION = 0
const DEFLATION_ORTHOGONAL = 4

function load_params()
    expr = get(ENV, "BENCH_JULIA_PARAMS_JL", "")
    isempty(expr) && return Dict{String,Any}()
    parsed = Base.invokelatest(eval, Meta.parse(expr))
    return Dict{String,Any}(String(k) => v for (k, v) in parsed)
end

const PARAMS = load_params()

param(name::AbstractString, default) = get(PARAMS, name, default)
pint(name::AbstractString, default) = Int(param(name, default))
pfloat(name::AbstractString, default) = Float64(param(name, default))
pbool(name::AbstractString, default) = Bool(param(name, default))

function pfloatvec(name::AbstractString, default)
    return Float64.(collect(param(name, default)))
end

function pintvec(name::AbstractString, default)
    return Int32.(collect(param(name, default)))
end

function load_dataset(seed::Integer)
    file = joinpath(CSV_DIR, "data_$(N_SAMPLES)x$(N_FEATURES)_seed$(seed).csv")
    arr = Matrix{Float64}(readdlm(file, ',', Float64; skipstart=1))
    X = Matrix{Float64}(arr[:, 1:N_FEATURES])
    y = Vector{Float64}(arr[:, N_FEATURES + 1])
    return X, y
end

function load_x_target(seed::Integer, X)
    get(ENV, "BENCH_JULIA_NEEDS_X_TARGET", "0") == "1" || return X
    dir = get(ENV, "BENCH_JULIA_X_TARGET_DIR", "")
    file = joinpath(dir, "xtarget_$(N_SAMPLES)x$(N_FEATURES)_seed$(seed).csv")
    isfile(file) || return X
    return Matrix{Float64}(readdlm(file, ',', Float64))
end

function build_inputs(X, y, seed::Integer)
    n, p = size(X)
    n_targets = pint("n_targets", 1)
    Y = Matrix{Float64}(undef, n, n_targets)
    Y[:, 1] = y
    for j in 2:n_targets
        Y[:, j] = 0.5 .* y .+ 0.1 .* X[:, mod(j - 2, p) + 1]
    end

    n_classes = pint("n_classes", 0)
    labels = zeros(Int32, n)
    if n_classes >= 2
        order = sortperm(y)
        for rank in 0:(n - 1)
            labels[order[rank + 1]] = Int32(mod(rank, n_classes))
        end
    else
        sorted = sort(y)
        median_y = sorted[fld(length(sorted), 2) + 1]
        labels .= Int32.(y .> median_y)
    end

    sample_weights = abs.(y) .+ 0.5
    n_groups = max(1, min(5, p))
    width = max(1, cld(p, n_groups))
    group_assignment = Int32[
        min(fld(j - 1, width), n_groups - 1) for j in 1:p
    ]

    return Dict{String,Any}(
        "Y" => Y,
        "X_target" => load_x_target(seed, X),
        "labels" => labels,
        "sample_weights" => sample_weights,
        "group_assignment" => group_assignment,
    )
end

function make_cfg(n_components=pint("n_components", N_COMPONENTS_ARG))
    cfg = Config()
    set_algorithm!(cfg, ALGO_PLS_REGRESSION)
    set_solver!(cfg, SOLVER_SIMPLS)
    set_deflation!(cfg, DEFLATION_REGRESSION)
    set_n_components!(cfg, max(1, Int(n_components)))
    set_center_x!(cfg, true)
    set_scale_x!(cfg, false)
    set_center_y!(cfg, true)
    set_scale_y!(cfg, false)
    set_store_scores!(cfg, true)
    return cfg
end

function make_plan(n_samples::Integer, n_folds::Integer=3)
    plan = ValidationPlan(n_samples)
    fold_size = max(1, fld(n_samples, n_folds))
    for f in 0:(n_folds - 1)
        start = f * fold_size
        stop = f < n_folds - 1 ? (f + 1) * fold_size : n_samples
        train = Int64[i for i in 0:(n_samples - 1) if i < start || i >= stop]
        test = Int64[i for i in start:(stop - 1)]
        add_fold!(plan, train, test)
    end
    return plan
end

function split_blocks(X, n_blocks::Integer)
    n, p = size(X)
    cols_per_block = fld(p, n_blocks)
    blocks = Matrix{Float64}[]
    for b in 1:n_blocks
        start = (b - 1) * cols_per_block + 1
        stop = b < n_blocks ? b * cols_per_block : p
        push!(blocks, Matrix{Float64}(X[:, start:stop]))
    end
    return blocks
end

function selector_mask(result, n_features::Integer)
    out = zeros(Float64, 1, n_features)
    idxs = try
        vector_int64(result, "selected_indices")
    catch
        Int64.(vector_int(result, "selected_indices"))
    end
    for idx in idxs
        j = Int(idx)
        if 0 <= j < n_features
            out[1, j + 1] = 1.0
        end
    end
    return out
end

extract(result, key::AbstractString, n_features::Integer) =
    key == "mask" ? selector_mask(result, n_features) : matrix(result, key)

function finish(result, key::AbstractString, n_features::Integer)
    try
        return extract(result, key, n_features)
    finally
        close(result)
    end
end

function compact_bank(n_operators::Integer)
    specs = [
        (0, Float64[]),
        (8, [11.0, 2.0]),
        (8, [21.0, 3.0]),
        (9, [11.0, 2.0, 1.0]),
        (9, [21.0, 3.0, 1.0]),
        (9, [11.0, 2.0, 2.0]),
        (7, [1.0]),
        (7, [2.0]),
        (15, [1.0]),
    ]
    bank = OperatorBank()
    for (kind, args) in specs[1:max(1, min(n_operators, length(specs)))]
        add_operator!(bank, kind, args)
    end
    return bank
end

function preprocess_bank(n_operators::Integer)
    kinds = [0, 1, 3, 2, 4]
    bank = OperatorBank()
    for kind in kinds[1:max(1, min(n_operators, length(kinds)))]
        add_operator!(bank, kind)
    end
    return bank
end

function one_se_matrix(max_components::Integer, n_folds::Integer)
    return [
        0.5 + 0.5 * mod(((row - 1) * n_folds + (col - 1)) * 37 + 11, 997) / 996
        for row in 1:max_components, col in 1:n_folds
    ]
end

function core_predict(ctx, cfg, X, Y, algorithm::Integer, solver::Integer,
                      deflation::Integer)
    set_algorithm!(cfg, algorithm)
    set_solver!(cfg, solver)
    set_deflation!(cfg, deflation)
    set_n_components!(cfg, pint("n_components", N_COMPONENTS_ARG))
    model = model_fit(ctx, cfg, X, Y)
    try
        return model_predict(ctx, model, X)
    finally
        close(model)
    end
end

function selector_dispatch(ctx, cfg, X, Y, input, seed::Integer)
    plan = make_plan(size(X, 1), pint("cv", 3))
    try
        if ALGO == "vissa_select"
            return finish(vissa_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 20), pint("n_submodels", 100),
                pfloat("ratio_kept", 0.1), pfloat("threshold", 0.5),
                pfloat("floor_probability", 0.01), pint("seed", seed)),
                "mask", N_FEATURES)
        elseif ALGO == "pso_select"
            return finish(pso_select(ctx, cfg, X, Y, plan,
                pint("n_swarm", 10), pint("n_iterations", 10),
                pfloat("w", 0.729), pfloat("c1", 1.494),
                pfloat("c2", 1.494), pfloat("v_max", 4.0),
                pint("seed", seed)), "mask", N_FEATURES)
        elseif ALGO in ("variable_select_vip", "variable_select_coef", "variable_select_sr")
            method = ALGO == "variable_select_vip" ? 0 :
                     ALGO == "variable_select_coef" ? 1 : 2
            model = model_fit(ctx, cfg, X, Y)
            try
                return finish(variable_select_rank(ctx, model, X, method,
                    pint("top_k", 10)), "mask", N_FEATURES)
            finally
                close(model)
            end
        elseif ALGO == "interval_select"
            return finish(interval_select(ctx, cfg, X, Y, plan,
                pint("interval_width", 5), pint("interval_step", pint("step", 1))),
                "mask", N_FEATURES)
        elseif ALGO == "bipls_select"
            return finish(bipls_select(ctx, cfg, X, Y, plan,
                pint("interval_width", 5), pint("min_intervals", 1)),
                "mask", N_FEATURES)
        elseif ALGO == "sipls_select"
            return finish(sipls_select(ctx, cfg, X, Y, plan,
                pint("interval_width", 5), pint("combination_size", 2)),
                "mask", N_FEATURES)
        elseif ALGO == "stability_select"
            return finish(stability_select(ctx, cfg, X, Y, plan,
                pint("top_k", 10)), "mask", N_FEATURES)
        elseif ALGO == "uve_select"
            return finish(uve_select(ctx, cfg, X, Y, plan,
                pint("noise_features", 5), pint("noise_seed", seed)),
                "mask", N_FEATURES)
        elseif ALGO == "spa_select"
            return finish(spa_select(ctx, cfg, X, Y, pint("top_k", 10)),
                "mask", N_FEATURES)
        elseif ALGO == "cars_select"
            return finish(cars_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 10), pint("min_features", 5)),
                "mask", N_FEATURES)
        elseif ALGO == "random_frog_select"
            return finish(random_frog_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 10), pint("initial_size", 5),
                pint("min_size", 3), pint("max_size", 10),
                pint("top_k", 10), pint("seed", seed)), "mask", N_FEATURES)
        elseif ALGO == "scars_select"
            return finish(scars_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 10), pint("min_features", 5),
                pfloat("sample_fraction", 0.8), pint("seed", seed)),
                "mask", N_FEATURES)
        elseif ALGO == "ga_select"
            return finish(ga_select(ctx, cfg, X, Y, plan,
                pint("n_generations", 10), pint("population_size", 12),
                pint("min_features", 3), pint("max_features", 10),
                pfloat("mutation_rate", 0.1), pint("seed", seed)),
                "mask", N_FEATURES)
        elseif ALGO == "shaving_select"
            return finish(shaving_select(ctx, cfg, X, Y, plan,
                pint("n_steps", 5), pint("min_features", 5),
                pfloat("shave_fraction", 0.2)), "mask", N_FEATURES)
        elseif ALGO == "bve_select"
            return finish(bve_select(ctx, cfg, X, Y, plan,
                pint("n_steps", 5), pint("min_features", 5)),
                "mask", N_FEATURES)
        elseif ALGO == "t2_select"
            return finish(t2_select(ctx, cfg, X, Y, plan,
                pfloatvec("alpha_thresholds", [0.05, 0.1]),
                pint("min_selected", 2)), "mask", N_FEATURES)
        elseif ALGO == "wvc_select"
            return finish(wvc_select(ctx, X, Y,
                pint("n_components", N_COMPONENTS_ARG), pint("top_k", 10),
                pbool("normalize", true)), "mask", N_FEATURES)
        elseif ALGO == "wvc_threshold_select"
            return finish(wvc_threshold_select(ctx, X, Y,
                pint("n_components", N_COMPONENTS_ARG), pbool("normalize", true),
                pfloat("score_threshold", 0.0), pfloat("threshold_factor", 1.0),
                pint("min_selected", 2)), "mask", N_FEATURES)
        elseif ALGO == "emcuve_select"
            return finish(emcuve_select(ctx, cfg, X, Y, plan,
                pint("noise_features", 5), pint("noise_seed", seed),
                pint("n_ensembles", 5), pfloat("vote_threshold", 0.5)),
                "mask", N_FEATURES)
        elseif ALGO == "randomization_select"
            return finish(randomization_select(ctx, cfg, X, Y,
                pint("n_permutations", 10), pint("randomization_seed", seed),
                pfloat("alpha", 0.05)), "mask", N_FEATURES)
        elseif ALGO == "rep_select"
            return finish(rep_select(ctx, cfg, X, Y, plan,
                pint("n_steps", 5), pint("min_features", 5),
                pint("remove_count", 1)), "mask", N_FEATURES)
        elseif ALGO == "ipw_select"
            return finish(ipw_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 5), pint("top_k", 10),
                pfloat("damping", 0.5), pfloat("weight_floor", 0.01)),
                "mask", N_FEATURES)
        elseif ALGO == "st_select"
            return finish(st_select(ctx, cfg, X, Y, plan,
                pfloatvec("thresholds", [0.2, 0.4, 0.6]),
                pint("min_selected", 2)), "mask", N_FEATURES)
        elseif ALGO == "ecr"
            return finish(ecr_fit(ctx, cfg, X, Y, pfloat("alpha", 0.5)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "iriv_select"
            return finish(iriv_select(ctx, cfg, X, Y, plan,
                pint("max_rounds", 2), pint("seed", seed)), "mask", N_FEATURES)
        elseif ALGO == "irf_select"
            return finish(irf_select(ctx, cfg, X, Y, plan,
                pint("n_iterations", 10), pint("window_size", 5),
                pint("initial_intervals", 3), pint("top_k", 3),
                pint("seed", seed)), "mask", N_FEATURES)
        elseif ALGO == "vip_spa_select"
            return finish(vip_spa_select(ctx, cfg, X, Y,
                pint("top_k", 10), pfloat("vip_threshold", 0.3)),
                "mask", N_FEATURES)
        else
            error("unsupported Julia benchmark algorithm: $ALGO")
        end
    finally
        close(plan)
    end
end

function fit_predict(seed::Integer)
    X, y = load_dataset(seed)
    input = build_inputs(X, y, seed)
    Y = input["Y"]
    ctx = Context()
    cfg = make_cfg()
    try
        if ALGO == "pls"
            return core_predict(ctx, cfg, X, Y, ALGO_PLS_REGRESSION,
                SOLVER_SIMPLS, DEFLATION_REGRESSION)
        elseif ALGO == "pcr"
            return core_predict(ctx, cfg, X, Y, ALGO_PCR,
                SOLVER_SVD, DEFLATION_REGRESSION)
        elseif ALGO == "opls"
            return core_predict(ctx, cfg, X, Y, ALGO_OPLS,
                SOLVER_NIPALS, DEFLATION_ORTHOGONAL)
        elseif ALGO == "sparse_simpls"
            return finish(sparse_simpls_fit(ctx, cfg, X, Y,
                pfloat("sparsity_lambda", 0.01)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "di_pls"
            return finish(di_pls_fit(ctx, cfg, X, Y, input["X_target"],
                pfloat("di_lambda", 0.1)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "recursive_pls"
            return finish(recursive_pls_run(ctx, cfg, X, Y,
                pint("window_size", max(5, fld(N_SAMPLES, 4)))),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "cppls"
            return finish(cppls_fit(ctx, cfg, X, Y, pfloat("gamma", 0.5)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "weighted_pls"
            return finish(weighted_pls_fit(ctx, cfg, X, Y,
                input["sample_weights"]), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "robust_pls"
            return finish(robust_pls_fit(ctx, cfg, X, Y,
                pfloat("huber_k", 1.35), pint("max_irls_iter", 5)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "ridge_pls"
            return finish(ridge_pls_fit(ctx, cfg, X, Y,
                pfloat("ridge_lambda", 0.01)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "continuum_regression"
            return finish(continuum_regression_fit(ctx, cfg, X, Y,
                pfloat("tau", 0.5)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "n_pls"
            return finish(n_pls_fit(ctx, cfg, X, pint("mode_j", 1),
                pint("mode_k", N_FEATURES), Y), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "kernel_pls_rbf"
            return finish(kernel_pls_fit(ctx, cfg, X, Y,
                pint("kernel_type", 1), pfloat("gamma", 0.1),
                pfloat("coef0", 1.0), pint("degree", 3)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "o2pls"
            return finish(o2pls_fit(ctx, cfg, X, Y,
                pint("n_predictive", 1), pint("n_x_orthogonal", 0),
                pint("n_y_orthogonal", 0)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "approximate_press"
            return finish(approximate_press_compute(ctx, cfg, X, Y,
                pint("max_components", pint("n_components", N_COMPONENTS_ARG))),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO in ("pls_diagnostic_t2", "pls_diagnostic_q", "pls_diagnostic_dmodx")
            model = model_fit(ctx, cfg, X, Y)
            try
                return finish(pls_diagnostics_compute(ctx, model, X),
                    PREDICTION_KEY, N_FEATURES)
            finally
                close(model)
            end
        elseif ALGO == "pls_monitoring"
            split = fld(N_SAMPLES, 2)
            Xref = Matrix{Float64}(X[1:split, :])
            Xmon = Matrix{Float64}(X[(split + 1):end, :])
            Yref = Matrix{Float64}(Y[1:split, :])
            model = model_fit(ctx, cfg, Xref, Yref)
            try
                return finish(pls_monitoring_run(ctx, model, Xref, Xmon,
                    pfloat("alpha", 0.05)), PREDICTION_KEY, N_FEATURES)
            finally
                close(model)
            end
        elseif ALGO == "one_se_rule"
            return finish(one_se_rule_compute(ctx,
                one_se_matrix(pint("max_components", 3), pint("n_folds", 3))),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "so_pls"
            blocks = split_blocks(X, pint("n_blocks", 3))
            comps = pintvec("n_components_per_block",
                fill(1, length(blocks)))
            return finish(so_pls_fit(ctx, cfg, blocks, Y, comps),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "on_pls"
            blocks = split_blocks(X, pint("n_blocks", 3))
            uniques = pintvec("n_unique_per_block",
                fill(1, length(blocks)))
            return finish(on_pls_fit(ctx, cfg, blocks, pint("n_joint", 1),
                uniques), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "rosa"
            blocks = split_blocks(X, pint("n_blocks", 3))
            return finish(rosa_fit(ctx, cfg, blocks, Y,
                pint("n_components", N_COMPONENTS_ARG)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "gpr_pls"
            return finish(gpr_pls_fit(ctx, cfg, X, Y,
                pfloat("length_scale", 1.0), pfloat("noise_level", 0.1),
                pint("seed", seed)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "bagging_pls"
            return finish(bagging_pls_fit(ctx, cfg, X, Y,
                pint("n_estimators", 5), pint("seed", seed)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "boosting_pls"
            return finish(boosting_pls_fit(ctx, cfg, X, Y,
                pint("n_estimators", 5), pfloat("learning_rate", 0.1)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "random_subspace_pls"
            return finish(random_subspace_pls_fit(ctx, cfg, X, Y,
                pint("n_estimators", 5),
                pint("features_per_subspace", max(1, fld(N_FEATURES, 2))),
                pint("seed", seed)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pls_glm"
            return finish(pls_glm_fit(ctx, cfg, X, Y,
                pbool("poisson", false)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pls_qda"
            return finish(pls_qda_fit(ctx, cfg, X, input["labels"]),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pls_cox"
            times = input["sample_weights"]
            events = Int32[label > 0 ? 1 : 0 for label in input["labels"]]
            return finish(pls_cox_fit(ctx, cfg, X, times, events),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pds"
            return finish(pds_fit(ctx, X, input["X_target"],
                pint("window_half_width", 1)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "ds"
            return finish(ds_fit(ctx, X, input["X_target"]),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "mir_pls"
            return finish(mir_pls_fit(ctx, cfg, X, Y), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "missing_aware_nipals"
            return finish(missing_aware_nipals_fit(ctx, cfg, X, Y),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "sparse_pls_da"
            return finish(sparse_pls_da_fit(ctx, cfg, X, input["labels"]),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "group_sparse_pls"
            return finish(group_sparse_pls_fit(ctx, cfg, X, Y,
                input["group_assignment"], pfloat("group_lambda", 0.05)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "fused_sparse_pls"
            return finish(fused_sparse_pls_fit(ctx, cfg, X, Y,
                pfloat("l1_lambda", 0.01), pfloat("fusion_lambda", 0.01)),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "mb_pls"
            blocks = split_blocks(X, pint("n_blocks", 3))
            sizes = Int64[size(b, 2) for b in blocks]
            return finish(mb_pls_fit(ctx, cfg, X, Y, sizes),
                PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "lw_pls"
            return finish(lw_pls_fit(ctx, cfg, X, Y,
                pint("n_neighbors", 30)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pls_lda"
            return finish(pls_lda_fit(ctx, cfg, X, input["labels"],
                pint("n_classes", 2)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "pls_logistic"
            return finish(pls_logistic_fit(ctx, cfg, X, input["labels"],
                pint("n_classes", 2)), PREDICTION_KEY, N_FEATURES)
        elseif ALGO == "aom_preprocess"
            bank = preprocess_bank(pint("n_operators", 3))
            gate = GatingStrategy(pint("gating_mode", 0))
            try
                return finish(aom_preprocess_fit(ctx, bank, gate, X, Y),
                    PREDICTION_KEY, N_FEATURES)
            finally
                close(gate)
                close(bank)
            end
        elseif ALGO in ("aom_pls", "pop_pls")
            bank = compact_bank(pint("n_operators", 9))
            plan = make_plan(N_SAMPLES, pint("cv", 3))
            try
                result = ALGO == "aom_pls" ?
                    aom_global_select(ctx, cfg, bank, X, Y, plan,
                        pint("max_components", 3)) :
                    aom_per_component_select(ctx, cfg, bank, X, Y, plan,
                        pint("max_components", 3))
                try
                    return predictions(result)
                finally
                    close(result)
                end
            finally
                close(plan)
                close(bank)
            end
        else
            return selector_dispatch(ctx, cfg, X, Y, input, seed)
        end
    finally
        close(cfg)
        close(ctx)
    end
end

flatten_matrix(A) = vec(transpose(Float64.(A)))

function median_value(values)
    sorted = sort(collect(values))
    mid = fld(length(sorted), 2) + 1
    return isodd(length(sorted)) ? sorted[mid] :
        0.5 * (sorted[mid - 1] + sorted[mid])
end

mean_value(values) = sum(values) / length(values)

function adaptive_target(probe_ms, max_total)
    capped = max(2, max_total)
    probe_ms > PROBE_ABORT_MS && return (2, "single", "probe_gt_60s")
    probe_ms > 30000.0 && return (min(capped, 2), "single", "probe_gt_30s")
    probe_ms > 5000.0 && return (min(capped, 3), "mean", "probe_gt_5s")
    probe_ms > 1000.0 && return (min(capped, 10), "median", "probe_gt_1s")
    probe_ms > 100.0 && return (min(capped, 20), "median", "probe_gt_100ms")
    return (min(capped, DEFAULT_MAX_RUNS), "median", "probe_le_100ms")
end

function stats(samples, statistic, warmup_ms, decision, total_runs;
               warmup_included=false)
    stat = length(samples) == 1 ? "single" : statistic
    reported = stat == "mean" ? mean_value(samples) : median_value(samples)
    return Dict{String,Any}(
        "ok" => true,
        "n_runs" => length(samples),
        "median_ms" => reported,
        "reported_ms" => reported,
        "sample_median_ms" => median_value(samples),
        "mean_ms" => mean_value(samples),
        "min_ms" => minimum(samples),
        "max_ms" => maximum(samples),
        "warmup_ms" => warmup_ms,
        "warmup_included" => warmup_included,
        "timing_statistic" => stat,
        "timing_decision" => decision,
        "max_runs" => max(1, MAX_RUNS),
        "total_runs" => total_runs,
    )
end

elapsed_ms(start_ns) = (time_ns() - start_ns) / 1_000_000.0

function time_runs_seeded()
    t0 = time_ns()
    warmup_preds = flatten_matrix(fit_predict(SEED_BASE))
    warmup_ms = elapsed_ms(t0)
    if warmup_ms > WARMUP_ABORT_MS || MAX_RUNS < 2
        decision = warmup_ms > WARMUP_ABORT_MS ? "warmup_gt_5min" :
            "max_runs_1_warmup_only"
        return stats([warmup_ms], "single", warmup_ms, decision, 1;
            warmup_included=true), warmup_preds
    end

    samples = Float64[]
    t0 = time_ns()
    last_preds = flatten_matrix(fit_predict(SEED_BASE))
    probe_ms = elapsed_ms(t0)
    push!(samples, probe_ms)
    target_total, statistic, decision = adaptive_target(probe_ms, MAX_RUNS)
    target_samples = max(1, target_total - 1)
    for i in 1:(target_samples - 1)
        t0 = time_ns()
        last_preds = flatten_matrix(fit_predict(SEED_BASE + i))
        push!(samples, elapsed_ms(t0))
    end
    return stats(samples, statistic, warmup_ms, decision, 1 + length(samples)),
        last_preds
end

function write_npy(path::AbstractString, data)
    mkpath(dirname(path))
    values = Float64.(collect(data))
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': ($(length(values)),), }"
    pad = mod(16 - mod(10 + ncodeunits(header) + 1, 16), 16)
    header *= repeat(" ", pad) * "\n"
    open(path, "w") do io
        write(io, UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])
        write(io, htol(UInt16(ncodeunits(header))))
        write(io, Vector{UInt8}(codeunits(header)))
        write(io, values)
    end
end

function json_escape(s::AbstractString)
    out = replace(String(s), "\\" => "\\\\", "\"" => "\\\"")
    out = replace(out, "\n" => "\\n", "\r" => "\\r", "\t" => "\\t")
    return out
end

function json_value(x)
    if x isa AbstractString
        return "\"" * json_escape(x) * "\""
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x === nothing
        return "null"
    elseif x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        return isfinite(x) ? string(x) : "null"
    elseif x isa Dict
        parts = String[]
        for (k, v) in x
            push!(parts, "\"" * json_escape(String(k)) * "\":" * json_value(v))
        end
        return "{" * join(parts, ",") * "}"
    else
        return "\"" * json_escape(string(x)) * "\""
    end
end

timing, predictions_vec = time_runs_seeded()
write_npy(PRED_PATH, predictions_vec)
timing["predictions_path"] = PRED_PATH
timing["versions"] = Dict{String,Any}(
    "language" => "Julia $(Base.VERSION)",
    "pls4all" => Pls4all.version(),
    "registry_method" => ALGO,
)
println(json_value(timing))
