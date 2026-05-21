# SPDX-License-Identifier: CECILL-2.1

export Context, Config, MethodResult, NativeModel, ValidationPlan,
       OperatorBank, GatingStrategy, MatrixView, AomGlobalResult,
       AomPerComponentResult,
       set_algorithm!, set_solver!, set_deflation!, set_n_components!,
       set_center_x!, set_scale_x!, set_center_y!, set_scale_y!,
       set_store_scores!, set_store_diagnostics!, set_n_samples!,
       add_fold!, add_operator!,
       matrix, vector_int, vector_int64, scalar, predictions, model_fit,
       model_array, model_predict,
       sparse_simpls_fit, di_pls_fit, recursive_pls_run, cppls_fit,
       weighted_pls_fit, robust_pls_fit, ridge_pls_fit,
       continuum_regression_fit, n_pls_fit, kernel_pls_fit, o2pls_fit,
       approximate_press_compute, sparse_pls_da_fit, group_sparse_pls_fit,
       fused_sparse_pls_fit, pls_monitoring_run, one_se_rule_compute,
       so_pls_fit, on_pls_fit,
       rosa_fit, bagging_pls_fit, gpr_pls_fit, boosting_pls_fit,
       random_subspace_pls_fit, pls_glm_fit, pls_qda_fit, pls_cox_fit,
       pds_fit, ds_fit, mir_pls_fit, missing_aware_nipals_fit,
       pls_diagnostics_compute,
       mb_pls_fit, lw_pls_fit, pls_lda_fit, pls_logistic_fit,
       aom_preprocess_fit, aom_global_select, aom_per_component_select,
       variable_select_rank, interval_select, stability_select, uve_select,
       spa_select, cars_select, random_frog_select, scars_select,
       ga_select, pso_select, vissa_select, shaving_select, bve_select,
       t2_select, wvc_select, wvc_threshold_select, emcuve_select,
       randomization_select, bipls_select, sipls_select, rep_select,
       ipw_select, st_select, ecr_fit, iriv_select, irf_select,
       vip_spa_select

const P4A_DTYPE_F64 = Int32(1)

struct MatrixView
    data::Ptr{Cvoid}
    rows::Int64
    cols::Int64
    row_stride::Int64
    col_stride::Int64
    dtype::Int32
    reserved0::Int32
end

mutable struct Context
    handle::Ptr{Cvoid}
end

mutable struct Config
    handle::Ptr{Cvoid}
end

mutable struct MethodResult
    handle::Ptr{Cvoid}
end

mutable struct NativeModel
    handle::Ptr{Cvoid}
end

mutable struct ValidationPlan
    handle::Ptr{Cvoid}
end

mutable struct OperatorBank
    handle::Ptr{Cvoid}
end

mutable struct GatingStrategy
    handle::Ptr{Cvoid}
end

mutable struct AomGlobalResult
    handle::Ptr{Cvoid}
end

mutable struct AomPerComponentResult
    handle::Ptr{Cvoid}
end

function _check(status::Integer, ctx::Union{Context,Nothing}=nothing)
    status == 0 && return nothing
    msg = ""
    if ctx !== nothing && ctx.handle != C_NULL
        raw = @p4a(:p4a_context_last_error, Cstring, (Ptr{Cvoid},), ctx.handle)
        raw != C_NULL && (msg = unsafe_string(raw))
    end
    if isempty(msg)
        raw = @p4a(:p4a_status_to_string, Cstring, (Int32,), Int32(status))
        raw != C_NULL && (msg = unsafe_string(raw))
    end
    error("pls4all error $(Int(status)): $msg")
end

function _owned_ptr(out::Base.RefValue{Ptr{Cvoid}}, name::AbstractString)
    out[] != C_NULL || error("$name returned a NULL handle")
    return out[]
end

function Context()
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_context_create, Int32, (Ref{Ptr{Cvoid}},), out))
    obj = Context(_owned_ptr(out, "p4a_context_create"))
    finalizer(close, obj)
    return obj
end

function Config()
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_config_create, Int32, (Ref{Ptr{Cvoid}},), out))
    obj = Config(_owned_ptr(out, "p4a_config_create"))
    finalizer(close, obj)
    return obj
end

function ValidationPlan(n_samples::Integer=0)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_validation_plan_create, Int32, (Ref{Ptr{Cvoid}},), out))
    obj = ValidationPlan(_owned_ptr(out, "p4a_validation_plan_create"))
    finalizer(close, obj)
    n_samples > 0 && set_n_samples!(obj, n_samples)
    return obj
end

function OperatorBank()
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_operator_bank_create, Int32, (Ref{Ptr{Cvoid}},), out))
    obj = OperatorBank(_owned_ptr(out, "p4a_operator_bank_create"))
    finalizer(close, obj)
    return obj
end

function GatingStrategy(mode::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_gating_strategy_create, Int32,
                (Ref{Ptr{Cvoid}}, Int32), out, Int32(mode)))
    obj = GatingStrategy(_owned_ptr(out, "p4a_gating_strategy_create"))
    finalizer(close, obj)
    return obj
end

function Base.close(ctx::Context)
    ctx.handle == C_NULL && return nothing
    @p4a(:p4a_context_destroy, Cvoid, (Ptr{Cvoid},), ctx.handle)
    ctx.handle = C_NULL
    return nothing
end

function Base.close(cfg::Config)
    cfg.handle == C_NULL && return nothing
    @p4a(:p4a_config_destroy, Cvoid, (Ptr{Cvoid},), cfg.handle)
    cfg.handle = C_NULL
    return nothing
end

function Base.close(r::MethodResult)
    r.handle == C_NULL && return nothing
    @p4a(:p4a_method_result_destroy, Cvoid, (Ptr{Cvoid},), r.handle)
    r.handle = C_NULL
    return nothing
end

function Base.close(m::NativeModel)
    m.handle == C_NULL && return nothing
    @p4a(:p4a_model_destroy, Cvoid, (Ptr{Cvoid},), m.handle)
    m.handle = C_NULL
    return nothing
end

function Base.close(p::ValidationPlan)
    p.handle == C_NULL && return nothing
    @p4a(:p4a_validation_plan_destroy, Cvoid, (Ptr{Cvoid},), p.handle)
    p.handle = C_NULL
    return nothing
end

function Base.close(b::OperatorBank)
    b.handle == C_NULL && return nothing
    @p4a(:p4a_operator_bank_destroy, Cvoid, (Ptr{Cvoid},), b.handle)
    b.handle = C_NULL
    return nothing
end

function Base.close(g::GatingStrategy)
    g.handle == C_NULL && return nothing
    @p4a(:p4a_gating_strategy_destroy, Cvoid, (Ptr{Cvoid},), g.handle)
    g.handle = C_NULL
    return nothing
end

function Base.close(r::AomGlobalResult)
    r.handle == C_NULL && return nothing
    @p4a(:p4a_aom_global_result_destroy, Cvoid, (Ptr{Cvoid},), r.handle)
    r.handle = C_NULL
    return nothing
end

function Base.close(r::AomPerComponentResult)
    r.handle == C_NULL && return nothing
    @p4a(:p4a_aom_per_component_result_destroy, Cvoid, (Ptr{Cvoid},), r.handle)
    r.handle = C_NULL
    return nothing
end

set_algorithm!(cfg::Config, v::Integer) =
    _check(@p4a(:p4a_config_set_algorithm, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_solver!(cfg::Config, v::Integer) =
    _check(@p4a(:p4a_config_set_solver, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_deflation!(cfg::Config, v::Integer) =
    _check(@p4a(:p4a_config_set_deflation, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_n_components!(cfg::Config, v::Integer) =
    _check(@p4a(:p4a_config_set_n_components, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_center_x!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_center_x, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_scale_x!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_scale_x, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_center_y!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_center_y, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_scale_y!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_scale_y, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_store_scores!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_store_scores, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))
set_store_diagnostics!(cfg::Config, v::Bool) =
    _check(@p4a(:p4a_config_set_store_diagnostics, Int32,
                (Ptr{Cvoid}, Int32), cfg.handle, Int32(v)))

function set_n_samples!(plan::ValidationPlan, n::Integer)
    _check(@p4a(:p4a_validation_plan_set_n_samples, Int32,
                (Ptr{Cvoid}, Int64), plan.handle, Int64(n)))
end

function add_fold!(plan::ValidationPlan, train, test)
    tr = Int64.(collect(train))
    te = Int64.(collect(test))
    GC.@preserve tr te begin
        _check(@p4a(:p4a_validation_plan_add_fold, Int32,
                    (Ptr{Cvoid}, Ptr{Int64}, Int64, Ptr{Int64}, Int64),
                    plan.handle, pointer(tr), Int64(length(tr)),
                    pointer(te), Int64(length(te))))
    end
end

function add_operator!(bank::OperatorBank, kind::Integer,
                       params::AbstractVector{<:Real}=Float64[])
    p = Float64.(collect(params))
    GC.@preserve p begin
        _check(@p4a(:p4a_operator_bank_add, Int32,
                    (Ptr{Cvoid}, Int32, Ptr{Float64}, Int32),
                    bank.handle, Int32(kind),
                    isempty(p) ? Ptr{Float64}(C_NULL) : pointer(p),
                    Int32(length(p))))
    end
end

function _matrix_view(A::AbstractMatrix{<:Real})
    n, p = size(A)
    rm = collect(transpose(Float64.(A)))
    view = MatrixView(Ptr{Cvoid}(pointer(rm)), Int64(n), Int64(p),
                      Int64(p), Int64(1), P4A_DTYPE_F64, Int32(0))
    return rm, view
end

function _method_result(ctx::Union{Context,Nothing}, status::Integer,
                        ptr::Ptr{Cvoid}, name::AbstractString)
    _check(status, ctx)
    ptr != C_NULL || error("$name returned a NULL handle")
    obj = MethodResult(ptr)
    finalizer(close, obj)
    return obj
end

function model_fit(ctx::Context, cfg::Config, X, Y)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve xb yb begin
        status = @p4a(:p4a_model_fit, Int32,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                       Ref{MatrixView}, Ref{Ptr{Cvoid}}),
                      ctx.handle, cfg.handle, xv, yv, out)
        _check(status, ctx)
    end
    obj = NativeModel(_owned_ptr(out, "p4a_model_fit"))
    finalizer(close, obj)
    return obj
end

function _array_copy_and_free(handle::Ptr{Cvoid})
    handle != C_NULL || error("native array handle is NULL")
    view_ref = Ref{MatrixView}()
    try
        _check(@p4a(:p4a_array_view, Int32,
                    (Ptr{Cvoid}, Ref{MatrixView}), handle, view_ref))
        view = view_ref[]
        rows = Int(view.rows)
        cols = Int(view.cols)
        (rows == 0 || cols == 0) && return Array{Float64}(undef, rows, cols)
        raw = unsafe_wrap(Array, Ptr{Float64}(view.data), (cols, rows))
        return Array(transpose(raw))
    finally
        @p4a(:p4a_array_free, Cvoid, (Ptr{Cvoid},), handle)
    end
end

function model_array(ctx::Context, model::NativeModel, kind::Integer)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(@p4a(:p4a_model_get_array, Int32,
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Ref{Ptr{Cvoid}}),
                ctx.handle, model.handle, Int32(kind), out), ctx)
    return _array_copy_and_free(out[])
end

function model_predict(ctx::Context, model::NativeModel, X)
    xb, xv = _matrix_view(X)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve xb begin
        _check(@p4a(:p4a_model_predict_alloc, Int32,
                    (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                     Ref{Ptr{Cvoid}}),
                    ctx.handle, model.handle, xv, out), ctx)
    end
    return _array_copy_and_free(out[])
end

function matrix(r::MethodResult, name::AbstractString)
    data = Ref{Ptr{Float64}}(C_NULL)
    rows = Ref{Int64}(0)
    cols = Ref{Int64}(0)
    _check(@p4a(:p4a_method_result_get_double_matrix, Int32,
                (Ptr{Cvoid}, Cstring, Ref{Ptr{Float64}},
                 Ref{Int64}, Ref{Int64}),
                r.handle, name, data, rows, cols))
    n = Int(rows[]) * Int(cols[])
    n == 0 && return Array{Float64}(undef, Int(rows[]), Int(cols[]))
    raw = unsafe_wrap(Array, data[], (Int(cols[]), Int(rows[])))
    return Array(transpose(raw))
end

function vector_int(r::MethodResult, name::AbstractString)
    data = Ref{Ptr{Int32}}(C_NULL)
    len = Ref{Int32}(0)
    _check(@p4a(:p4a_method_result_get_int_vector, Int32,
                (Ptr{Cvoid}, Cstring, Ref{Ptr{Int32}}, Ref{Int32}),
                r.handle, name, data, len))
    len[] == 0 && return Int32[]
    return copy(unsafe_wrap(Array, data[], Int(len[])))
end

function vector_int64(r::MethodResult, name::AbstractString)
    data = Ref{Ptr{Int64}}(C_NULL)
    len = Ref{Int64}(0)
    _check(@p4a(:p4a_method_result_get_int64_vector, Int32,
                (Ptr{Cvoid}, Cstring, Ref{Ptr{Int64}}, Ref{Int64}),
                r.handle, name, data, len))
    len[] == 0 && return Int64[]
    return copy(unsafe_wrap(Array, data[], Int(len[])))
end

function _aom_predictions(handle::Ptr{Cvoid}, getter::Symbol)
    data = Ref{Ptr{Float64}}(C_NULL)
    rows = Ref{Int64}(0)
    cols = Ref{Int64}(0)
    if getter === :p4a_aom_global_result_get_predictions
        _check(@p4a(:p4a_aom_global_result_get_predictions, Int32,
                    (Ptr{Cvoid}, Ref{Ptr{Float64}}, Ref{Int64}, Ref{Int64}),
                    handle, data, rows, cols))
    else
        _check(@p4a(:p4a_aom_per_component_result_get_predictions, Int32,
                    (Ptr{Cvoid}, Ref{Ptr{Float64}}, Ref{Int64}, Ref{Int64}),
                    handle, data, rows, cols))
    end
    raw = unsafe_wrap(Array, data[], (Int(cols[]), Int(rows[])))
    return Array(transpose(raw))
end

predictions(r::AomGlobalResult) =
    _aom_predictions(r.handle, :p4a_aom_global_result_get_predictions)
predictions(r::AomPerComponentResult) =
    _aom_predictions(r.handle, :p4a_aom_per_component_result_get_predictions)

function scalar(r::MethodResult, name::AbstractString)
    out = Ref{Float64}(NaN)
    _check(@p4a(:p4a_method_result_get_scalar, Int32,
                (Ptr{Cvoid}, Cstring, Ref{Float64}),
                r.handle, name, out))
    return out[]
end

macro result_call(cname, ctx, rettype, argtypes, args...)
    quote
        local out = Ref{Ptr{Cvoid}}(C_NULL)
        local status = @p4a($(QuoteNode(cname)), Int32,
                            $(esc(argtypes)), $(map(esc, args)...), out)
        _method_result($(esc(ctx)), status, out[], string($(QuoteNode(cname))))
    end
end

function sparse_simpls_fit(ctx::Context, cfg::Config, X, Y, λ::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_sparse_simpls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}}), ctx.handle, cfg.handle, xv, yv, Float64(λ))
    end
end

function di_pls_fit(ctx::Context, cfg::Config, Xs, Ys, Xt, λ::Real)
    xsb, xsv = _matrix_view(Xs); ysb, ysv = _matrix_view(Ys)
    xtb, xtv = _matrix_view(Xt)
    GC.@preserve xsb ysb xtb begin
        @result_call(p4a_di_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ref{MatrixView}, Float64, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xsv, ysv, xtv, Float64(λ))
    end
end

function recursive_pls_run(ctx::Context, cfg::Config, X, Y, window_size::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_recursive_pls_run, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Int32(window_size))
    end
end

function cppls_fit(ctx::Context, cfg::Config, X, Y, γ::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_cppls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Float64(γ))
    end
end

function weighted_pls_fit(ctx::Context, cfg::Config, X, Y, weights)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    w = Float64.(collect(weights))
    GC.@preserve xb yb w begin
        @result_call(p4a_weighted_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Float64}, Int64, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, pointer(w), Int64(length(w)))
    end
end

function robust_pls_fit(ctx::Context, cfg::Config, X, Y,
                        huber_k::Real, max_irls_iter::Integer=5)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_robust_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Int32, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Float64(huber_k), Int32(max_irls_iter))
    end
end

function ridge_pls_fit(ctx::Context, cfg::Config, X, Y, λ::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_ridge_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Float64(λ))
    end
end

function continuum_regression_fit(ctx::Context, cfg::Config, X, Y, τ::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_continuum_regression_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Float64(τ))
    end
end

function n_pls_fit(ctx::Context, cfg::Config, X, mode_j::Integer,
                   mode_k::Integer, Y)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_n_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Int32, Int32,
             Ref{MatrixView}, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, Int32(mode_j), Int32(mode_k), yv)
    end
end

function kernel_pls_fit(ctx::Context, cfg::Config, X, Y, kernel_type::Integer,
                        gamma::Real=0.0, coef0::Real=1.0,
                        degree::Integer=3)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_kernel_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Int32, Float64, Float64, Int32,
             Ref{MatrixView}, Ref{MatrixView}, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, Int32(kernel_type), Float64(gamma),
            Float64(coef0), Int32(degree), xv, yv)
    end
end

function o2pls_fit(ctx::Context, cfg::Config, X, Y, n_pred::Integer,
                   n_x_orthogonal::Integer=0, n_y_orthogonal::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_o2pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Int32, Int32, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Int32(n_pred),
            Int32(n_x_orthogonal), Int32(n_y_orthogonal))
    end
end

function approximate_press_compute(ctx::Context, cfg::Config, X, Y,
                                   max_components::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_approximate_press_compute, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Ref{Ptr{Cvoid}})
            , ctx.handle, cfg.handle, xv, yv, Int32(max_components))
    end
end

function pls_diagnostics_compute(ctx::Context, model::NativeModel, X,
                                 X_reference=nothing)
    xb, xv = _matrix_view(X)
    if X_reference === nothing
        GC.@preserve xb begin
            @result_call(p4a_pls_diagnostics_compute, ctx, MethodResult,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Cvoid},
                 Ref{Ptr{Cvoid}}),
                 ctx.handle, model.handle, xv, C_NULL)
        end
    else
        rb, rv = _matrix_view(X_reference)
        GC.@preserve xb rb begin
            @result_call(p4a_pls_diagnostics_compute, ctx, MethodResult,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
                 Ref{Ptr{Cvoid}}),
                 ctx.handle, model.handle, xv, rv)
        end
    end
end

function pls_monitoring_run(ctx::Context, model::NativeModel, X_reference,
                            X_monitor, alpha::Real=0.05)
    rb, rv = _matrix_view(X_reference); mb, mv = _matrix_view(X_monitor)
    GC.@preserve rb mb begin
        @result_call(p4a_pls_monitoring_run, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, model.handle, rv, mv, Float64(alpha))
    end
end

function variable_select_rank(ctx::Context, model::NativeModel, X,
                              method::Integer, top_k::Integer)
    xb, xv = _matrix_view(X)
    GC.@preserve xb begin
        @result_call(p4a_variable_select_rank, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Int32, Int32,
             Ref{Ptr{Cvoid}}),
             ctx.handle, model.handle, xv, Int32(method), Int32(top_k))
    end
end

function sparse_pls_da_fit(ctx::Context, cfg::Config, X, labels)
    xb, xv = _matrix_view(X); y = Int32.(collect(labels))
    GC.@preserve xb y begin
        @result_call(p4a_sparse_pls_da_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Int32}, Int64,
             Ref{Ptr{Cvoid}}), ctx.handle, cfg.handle, xv, pointer(y),
             Int64(length(y)))
    end
end

function group_sparse_pls_fit(ctx::Context, cfg::Config, X, Y, groups, λ::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    g = Int32.(collect(groups))
    GC.@preserve xb yb g begin
        @result_call(p4a_group_sparse_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Int32}, Int64, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, pointer(g), Int64(length(g)),
             Float64(λ))
    end
end

function fused_sparse_pls_fit(ctx::Context, cfg::Config, X, Y,
                              l1::Real, fusion::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_fused_sparse_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Float64(l1), Float64(fusion))
    end
end

function one_se_rule_compute(ctx::Context, fold_rmse::AbstractMatrix{<:Real})
    rows, cols = size(fold_rmse)
    rm = vec(collect(transpose(Float64.(fold_rmse))))
    GC.@preserve rm begin
        @result_call(p4a_one_se_rule_compute, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Float64}, Int32, Int32, Ref{Ptr{Cvoid}}),
            ctx.handle, pointer(rm), Int32(rows), Int32(cols))
    end
end

function pds_fit(ctx::Context, Xs, Xt, window_half_width::Integer)
    xsb, xsv = _matrix_view(Xs); xtb, xtv = _matrix_view(Xt)
    GC.@preserve xsb xtb begin
        @result_call(p4a_pds_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView}, Int32,
             Ref{Ptr{Cvoid}}),
            ctx.handle, xsv, xtv, Int32(window_half_width))
    end
end

function ds_fit(ctx::Context, Xs, Xt)
    xsb, xsv = _matrix_view(Xs); xtb, xtv = _matrix_view(Xt)
    GC.@preserve xsb xtb begin
        @result_call(p4a_ds_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView}, Ref{Ptr{Cvoid}}),
            ctx.handle, xsv, xtv)
    end
end

function mir_pls_fit(ctx::Context, cfg::Config, X, Y)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_mir_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ref{Ptr{Cvoid}}), ctx.handle, cfg.handle, xv, yv)
    end
end

function missing_aware_nipals_fit(ctx::Context, cfg::Config, X, Y)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_missing_aware_nipals_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ref{Ptr{Cvoid}}), ctx.handle, cfg.handle, xv, yv)
    end
end

function bagging_pls_fit(ctx::Context, cfg::Config, X, Y,
                         n_estimators::Integer, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_bagging_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(n_estimators), UInt64(seed))
    end
end

function gpr_pls_fit(ctx::Context, cfg::Config, X, Y, length_scale::Real,
                     noise_level::Real, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_gpr_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Float64, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Float64(length_scale),
             Float64(noise_level), UInt64(seed))
    end
end

function boosting_pls_fit(ctx::Context, cfg::Config, X, Y,
                          n_estimators::Integer, learning_rate::Real=0.1)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_boosting_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(n_estimators),
             Float64(learning_rate))
    end
end

function random_subspace_pls_fit(ctx::Context, cfg::Config, X, Y,
                                 n_estimators::Integer,
                                 features_per_subspace::Integer,
                                 seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_random_subspace_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Int32, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(n_estimators),
             Int32(features_per_subspace), UInt64(seed))
    end
end

function pls_glm_fit(ctx::Context, cfg::Config, X, Y, poisson::Bool=false)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_pls_glm_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(poisson))
    end
end

function pls_qda_fit(ctx::Context, cfg::Config, X, labels)
    xb, xv = _matrix_view(X); y = Int32.(collect(labels))
    GC.@preserve xb y begin
        @result_call(p4a_pls_qda_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Int32}, Int64,
             Ref{Ptr{Cvoid}}), ctx.handle, cfg.handle, xv, pointer(y),
             Int64(length(y)))
    end
end

function pls_cox_fit(ctx::Context, cfg::Config, X, survival_times,
                     event_indicators)
    xb, xv = _matrix_view(X)
    t = Float64.(collect(survival_times))
    e = Int32.(collect(event_indicators))
    GC.@preserve xb t e begin
        @result_call(p4a_pls_cox_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Float64}, Int64,
             Ptr{Int32}, Int64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, pointer(t), Int64(length(t)),
             pointer(e), Int64(length(e)))
    end
end

function mb_pls_fit(ctx::Context, cfg::Config, X, Y, block_sizes)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    sizes = Int64.(collect(block_sizes))
    GC.@preserve xb yb sizes begin
        @result_call(p4a_mb_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Int64}, Int64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, pointer(sizes), Int64(length(sizes)))
    end
end

function lw_pls_fit(ctx::Context, cfg::Config, X, Y, n_neighbors::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_lw_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(n_neighbors))
    end
end

function pls_lda_fit(ctx::Context, cfg::Config, X, labels, n_classes::Integer)
    xb, xv = _matrix_view(X); y = Int32.(collect(labels))
    GC.@preserve xb y begin
        @result_call(p4a_pls_lda_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Int32}, Int64,
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, pointer(y), Int64(length(y)),
             Int32(n_classes))
    end
end

function pls_logistic_fit(ctx::Context, cfg::Config, X, labels,
                          n_classes::Integer)
    xb, xv = _matrix_view(X); y = Int32.(collect(labels))
    GC.@preserve xb y begin
        @result_call(p4a_pls_logistic_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ptr{Int32}, Int64,
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, pointer(y), Int64(length(y)),
             Int32(n_classes))
    end
end

function ecr_fit(ctx::Context, cfg::Config, X, Y, α::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_ecr_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Float64(α))
    end
end

function _block_views(blocks)
    buffers = Vector{Any}()
    views = MatrixView[]
    for b in blocks
        buf, view = _matrix_view(b)
        push!(buffers, buf)
        push!(views, view)
    end
    return buffers, views
end

function so_pls_fit(ctx::Context, cfg::Config, blocks, Y, comps)
    bufs, views = _block_views(blocks); yb, yv = _matrix_view(Y)
    c = Int32.(collect(comps))
    GC.@preserve bufs views yb c begin
        @result_call(p4a_so_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{MatrixView}, Int32,
             Ref{MatrixView}, Ptr{Int32}, Int64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, pointer(views), Int32(length(views)),
             yv, pointer(c), Int64(length(c)))
    end
end

function on_pls_fit(ctx::Context, cfg::Config, blocks, n_joint::Integer, uniques)
    bufs, views = _block_views(blocks)
    u = Int32.(collect(uniques))
    GC.@preserve bufs views u begin
        @result_call(p4a_on_pls_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{MatrixView}, Int32, Int32,
             Ptr{Int32}, Int64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, pointer(views), Int32(length(views)),
             Int32(n_joint), pointer(u), Int64(length(u)))
    end
end

function rosa_fit(ctx::Context, cfg::Config, blocks, Y, n_components::Integer)
    bufs, views = _block_views(blocks); yb, yv = _matrix_view(Y)
    GC.@preserve bufs views yb begin
        @result_call(p4a_rosa_fit, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{MatrixView}, Int32,
             Ref{MatrixView}, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, pointer(views), Int32(length(views)),
             yv, Int32(n_components))
    end
end

function aom_preprocess_fit(ctx::Context, bank::OperatorBank,
                            gate::GatingStrategy, X, Y=nothing)
    xb, xv = _matrix_view(X)
    if Y === nothing
        GC.@preserve xb begin
            @result_call(p4a_aom_preprocess_fit, ctx, MethodResult,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                 Ptr{Cvoid}, Ref{Ptr{Cvoid}}),
                 ctx.handle, bank.handle, gate.handle, xv, C_NULL)
        end
    else
        yb, yv = _matrix_view(Y)
        GC.@preserve xb yb begin
            @result_call(p4a_aom_preprocess_fit, ctx, MethodResult,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                 Ref{MatrixView}, Ref{Ptr{Cvoid}}),
                 ctx.handle, bank.handle, gate.handle, xv, yv)
        end
    end
end

function aom_global_select(ctx::Context, cfg::Config, bank::OperatorBank,
                           X, Y, plan::ValidationPlan,
                           max_components::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve xb yb begin
        status = @p4a(:p4a_aom_global_select, Int32,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                       Ref{MatrixView}, Ptr{Cvoid}, Int32,
                       Ref{Ptr{Cvoid}}),
                      ctx.handle, cfg.handle, bank.handle, xv, yv,
                      plan.handle, Int32(max_components), out)
        _check(status, ctx)
    end
    obj = AomGlobalResult(_owned_ptr(out, "p4a_aom_global_select"))
    finalizer(close, obj)
    return obj
end

function aom_per_component_select(ctx::Context, cfg::Config,
                                  bank::OperatorBank, X, Y,
                                  plan::ValidationPlan,
                                  max_components::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    GC.@preserve xb yb begin
        status = @p4a(:p4a_aom_per_component_select, Int32,
                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView},
                       Ref{MatrixView}, Ptr{Cvoid}, Int32,
                       Ref{Ptr{Cvoid}}),
                      ctx.handle, cfg.handle, bank.handle, xv, yv,
                      plan.handle, Int32(max_components), out)
        _check(status, ctx)
    end
    obj = AomPerComponentResult(
        _owned_ptr(out, "p4a_aom_per_component_select"))
    finalizer(close, obj)
    return obj
end

function _selector_result(ctx, name, status, out)
    _method_result(ctx, status, out[], name)
end

function interval_select(ctx::Context, cfg::Config, X, Y,
                         plan::ValidationPlan, width::Integer, step::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_interval_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(width),
             Int32(step))
    end
end

function stability_select(ctx::Context, cfg::Config, X, Y,
                          plan::ValidationPlan, top_k::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_stability_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(top_k))
    end
end

function uve_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    noise_features::Integer, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_uve_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(noise_features), UInt64(seed))
    end
end

function spa_select(ctx::Context, cfg::Config, X, Y, top_k::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_spa_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(top_k))
    end
end

function cars_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                     n_iterations::Integer, min_features::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_cars_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(n_iterations), Int32(min_features))
    end
end

function random_frog_select(ctx::Context, cfg::Config, X, Y,
                            plan::ValidationPlan, n_iterations::Integer,
                            initial_size::Integer, min_size::Integer,
                            max_size::Integer, top_k::Integer,
                            seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_random_frog_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Int32, Int32, Int32, UInt64,
             Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_iterations),
             Int32(initial_size), Int32(min_size), Int32(max_size),
             Int32(top_k), UInt64(seed))
    end
end

function scars_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                      n_iterations::Integer, min_features::Integer,
                      sample_fraction::Real, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_scars_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Float64, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(n_iterations), Int32(min_features),
             Float64(sample_fraction), UInt64(seed))
    end
end

function ga_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                   n_generations::Integer, population_size::Integer,
                   min_features::Integer, max_features::Integer,
                   mutation_rate::Real, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_ga_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Int32, Int32, Float64, UInt64,
             Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_generations),
             Int32(population_size), Int32(min_features), Int32(max_features),
             Float64(mutation_rate), UInt64(seed))
    end
end

function pso_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    n_swarm::Integer, n_iterations::Integer, w::Real=0.729,
                    c1::Real=1.494, c2::Real=1.494, vmax::Real=4.0,
                    seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_pso_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Float64, Float64, Float64, Float64,
             UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_swarm),
             Int32(n_iterations), Float64(w), Float64(c1), Float64(c2),
             Float64(vmax), UInt64(seed))
    end
end

function vissa_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                      n_iterations::Integer=20, n_submodels::Integer=100,
                      ratio_kept::Real=0.1, threshold::Real=0.5,
                      floor_probability::Real=0.01, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_vissa_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Float64, Float64, Float64, UInt64,
             Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(n_iterations), Int32(n_submodels), Float64(ratio_kept),
             Float64(threshold), Float64(floor_probability), UInt64(seed))
    end
end

function shaving_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                        n_steps::Integer, min_features::Integer,
                        shave_fraction::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_shaving_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_steps),
             Int32(min_features), Float64(shave_fraction))
    end
end

function bve_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    n_steps::Integer, min_features::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_bve_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_steps),
             Int32(min_features))
    end
end

function t2_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                   alpha_thresholds, min_selected::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    a = Float64.(collect(alpha_thresholds))
    GC.@preserve xb yb a begin
        @result_call(p4a_t2_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Ptr{Float64}, Int64, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, pointer(a),
             Int64(length(a)), Int32(min_selected))
    end
end

function wvc_select(ctx::Context, X, Y, n_components::Integer, top_k::Integer,
                    normalize::Bool=true)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_wvc_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView}, Int32, Int32,
             Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, xv, yv, Int32(n_components), Int32(top_k),
             Int32(normalize))
    end
end

function wvc_threshold_select(ctx::Context, X, Y, n_components::Integer,
                              normalize::Bool=true, score_threshold::Real=0.0,
                              threshold_factor::Real=1.0,
                              min_selected::Integer=1)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_wvc_threshold_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView}, Int32, Int32,
             Float64, Float64, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, xv, yv, Int32(n_components), Int32(normalize),
             Float64(score_threshold), Float64(threshold_factor),
             Int32(min_selected))
    end
end

function emcuve_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                       noise_features::Integer, noise_seed::Integer,
                       n_ensembles::Integer, vote_threshold::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_emcuve_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, UInt64, Int32, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(noise_features), UInt64(noise_seed), Int32(n_ensembles),
             Float64(vote_threshold))
    end
end

function randomization_select(ctx::Context, cfg::Config, X, Y,
                              n_permutations::Integer,
                              seed::Integer=0, alpha::Real=0.05)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_randomization_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Int32, UInt64, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Int32(n_permutations),
             UInt64(seed), Float64(alpha))
    end
end

function bipls_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                      interval_width::Integer, min_intervals::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_bipls_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(interval_width), Int32(min_intervals))
    end
end

function sipls_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                      interval_width::Integer, combination_size::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_sipls_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(interval_width), Int32(combination_size))
    end
end

function rep_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    n_steps::Integer, min_features::Integer,
                    remove_count::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_rep_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, Int32(n_steps),
             Int32(min_features), Int32(remove_count))
    end
end

function ipw_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    n_iterations::Integer, top_k::Integer,
                    damping::Real, weight_floor::Real)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_ipw_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Float64, Float64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(n_iterations), Int32(top_k), Float64(damping),
             Float64(weight_floor))
    end
end

function st_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                   thresholds, min_selected::Integer)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    thr = Float64.(collect(thresholds))
    GC.@preserve xb yb thr begin
        @result_call(p4a_st_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Ptr{Float64}, Int64, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle, pointer(thr),
             Int64(length(thr)), Int32(min_selected))
    end
end

function iriv_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                     max_rounds::Integer, seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_iriv_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, UInt64, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(max_rounds), UInt64(seed))
    end
end

function irf_select(ctx::Context, cfg::Config, X, Y, plan::ValidationPlan,
                    n_iterations::Integer, window_size::Integer,
                    initial_intervals::Integer, top_k::Integer,
                    seed::Integer=0)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_irf_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Ptr{Cvoid}, Int32, Int32, Int32, Int32, UInt64,
             Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, plan.handle,
             Int32(n_iterations), Int32(window_size),
             Int32(initial_intervals), Int32(top_k), UInt64(seed))
    end
end

function vip_spa_select(ctx::Context, cfg::Config, X, Y, top_k::Integer,
                        vip_threshold::Real=0.3)
    xb, xv = _matrix_view(X); yb, yv = _matrix_view(Y)
    GC.@preserve xb yb begin
        @result_call(p4a_vip_spa_select, ctx, MethodResult,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{MatrixView}, Ref{MatrixView},
             Float64, Int32, Ref{Ptr{Cvoid}}),
             ctx.handle, cfg.handle, xv, yv, Float64(vip_threshold),
             Int32(top_k))
    end
end
