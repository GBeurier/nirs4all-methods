# SPDX-License-Identifier: CECILL-2.1

import MLJModelInterface

const MMI = MLJModelInterface

export PLSRegressor, PCRRegressor, OPLSRegressor

abstract type AbstractPls4allRegressor <: MMI.Deterministic end

Base.@kwdef mutable struct PLSRegressor <: AbstractPls4allRegressor
    n_components::Int = 2
    center_x::Bool = true
    scale_x::Bool = false
    center_y::Bool = true
    scale_y::Bool = false
end

Base.@kwdef mutable struct PCRRegressor <: AbstractPls4allRegressor
    n_components::Int = 2
    center_x::Bool = true
    scale_x::Bool = false
    center_y::Bool = true
    scale_y::Bool = false
end

Base.@kwdef mutable struct OPLSRegressor <: AbstractPls4allRegressor
    n_components::Int = 2
    center_x::Bool = true
    scale_x::Bool = false
    center_y::Bool = true
    scale_y::Bool = false
end

function MMI.clean!(model::AbstractPls4allRegressor)
    model.n_components >= 1 && return ""
    model.n_components = 1
    return "n_components must be >= 1; reset to 1."
end

function _mlj_config!(cfg::Config, model::PLSRegressor)
    set_algorithm!(cfg, 0)  # P4A_ALGO_PLS_REGRESSION
    set_solver!(cfg, 1)     # P4A_SOLVER_SIMPLS
    set_deflation!(cfg, 0)  # P4A_DEFLATION_REGRESSION
    _mlj_common_config!(cfg, model)
end

function _mlj_config!(cfg::Config, model::PCRRegressor)
    set_algorithm!(cfg, 10) # P4A_ALGO_PCR
    set_solver!(cfg, 5)     # P4A_SOLVER_SVD
    set_deflation!(cfg, 0)  # P4A_DEFLATION_REGRESSION
    _mlj_common_config!(cfg, model)
end

function _mlj_config!(cfg::Config, model::OPLSRegressor)
    set_algorithm!(cfg, 4)  # P4A_ALGO_OPLS
    set_solver!(cfg, 0)     # P4A_SOLVER_NIPALS
    set_deflation!(cfg, 4)  # P4A_DEFLATION_ORTHOGONAL
    _mlj_common_config!(cfg, model)
end

function _mlj_common_config!(cfg::Config, model::AbstractPls4allRegressor)
    set_n_components!(cfg, model.n_components)
    set_center_x!(cfg, model.center_x)
    set_scale_x!(cfg, model.scale_x)
    set_center_y!(cfg, model.center_y)
    set_scale_y!(cfg, model.scale_y)
    set_store_scores!(cfg, true)
    return cfg
end

function _fitresult_snapshot(ctx::Context, native::NativeModel,
                             feature_names, target_was_vector::Bool)
    coefficients = model_array(ctx, native, 0)
    x_mean = vec(model_array(ctx, native, 2))
    y_mean = vec(model_array(ctx, native, 4))
    return (native=native,
            coefficients=coefficients,
            x_mean=x_mean,
            y_mean=y_mean,
            feature_names=feature_names,
            target_was_vector=target_was_vector,
            n_features=size(coefficients, 1),
            n_targets=size(coefficients, 2))
end

function MMI.fit(model::AbstractPls4allRegressor, verbosity, X, y)
    message = MMI.clean!(model)
    verbosity > 0 && !isempty(message) && @warn message

    Xmat, feature_names = _feature_matrix(X)
    Ymat, _, target_was_vector = _target_matrix(y)
    size(Xmat, 1) == size(Ymat, 1) || throw(ArgumentError(
        "X rows ($(size(Xmat, 1))) must equal y rows ($(size(Ymat, 1)))"))

    ctx = Context()
    cfg = Config()
    try
        _mlj_config!(cfg, model)
        native = model_fit(ctx, cfg, Xmat, Ymat)
        fitresult = _fitresult_snapshot(ctx, native, feature_names,
                                         target_was_vector)
        train_predictions = model_predict(ctx, native, Xmat)
        report = (training_predictions=target_was_vector ?
                    vec(train_predictions) : train_predictions,
                  feature_names=feature_names)
        return fitresult, nothing, report
    finally
        close(cfg)
        close(ctx)
    end
end

function MMI.predict(model::AbstractPls4allRegressor, fitresult, Xnew)
    columns = fitresult.feature_names
    Xmat, _ = columns === nothing ?
        _feature_matrix(Xnew) :
        _feature_matrix(Xnew; columns=columns)
    size(Xmat, 2) == fitresult.n_features || throw(DimensionMismatch(
        "Xnew has $(size(Xmat, 2)) features, but the model was fitted with $(fitresult.n_features)"))

    ctx = Context()
    try
        yhat = model_predict(ctx, fitresult.native, Xmat)
        return fitresult.target_was_vector && size(yhat, 2) == 1 ?
            vec(yhat) : yhat
    finally
        close(ctx)
    end
end

function MMI.fitted_params(model::AbstractPls4allRegressor, fitresult)
    return (coefficients=fitresult.coefficients,
            x_mean=fitresult.x_mean,
            y_mean=fitresult.y_mean,
            feature_names=fitresult.feature_names,
            n_components=model.n_components)
end

const _PLS4ALL_INPUT_SCITYPE = MMI.Table(MMI.Continuous)
const _PLS4ALL_TARGET_SCITYPE = AbstractVector{MMI.Continuous}

MMI.metadata_pkg.(
    (PLSRegressor, PCRRegressor, OPLSRegressor);
    name="Pls4all",
    uuid="773cca83-73c7-4c8b-9de3-b5f2b2ce6491",
    url="https://github.com/GBeurier/pls4all",
    julia=false,
    license="CECILL-2.1",
    is_wrapper=true,
)

MMI.metadata_model(PLSRegressor;
    input=_PLS4ALL_INPUT_SCITYPE,
    target=_PLS4ALL_TARGET_SCITYPE,
    supports_weights=false,
    load_path="Pls4all.PLSRegressor",
    human_name="pls4all PLS regressor",
)

MMI.metadata_model(PCRRegressor;
    input=_PLS4ALL_INPUT_SCITYPE,
    target=_PLS4ALL_TARGET_SCITYPE,
    supports_weights=false,
    load_path="Pls4all.PCRRegressor",
    human_name="pls4all PCR regressor",
)

MMI.metadata_model(OPLSRegressor;
    input=_PLS4ALL_INPUT_SCITYPE,
    target=_PLS4ALL_TARGET_SCITYPE,
    supports_weights=false,
    load_path="Pls4all.OPLSRegressor",
    human_name="pls4all OPLS regressor",
)
