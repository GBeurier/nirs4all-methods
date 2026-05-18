# SPDX-License-Identifier: CECILL-2.1
#
# Pls4all.Sklearn — tier-2 idiomatic Julia wrapper around the tier-1
# pls_fit / pls_predict primitives.
#
# Provides a mutable model struct with an `fit!` / `predict` surface
# compatible with the de-facto Julia ML conventions (StatsAPI-style).
# Heavier framework adapters (MLJModelInterface, ScikitLearn.jl) can be
# layered on top as separate packages; this module deliberately avoids
# pulling MLJ as a hard dependency.
#
# Example:
#
#     using Pls4all.Sklearn
#     m = PLSRegression(n_components=5)
#     fit!(m, X, y)
#     yhat = predict(m, Xnew)

module Sklearn

import ..Pls4all

export PLSRegression, fit!, predict, score

mutable struct PLSRegression
    n_components::Int
    # Fitted state (populated by fit!)
    coefficients::Union{Matrix{Float64}, Nothing}
    x_mean::Union{Vector{Float64}, Nothing}
    y_mean::Union{Vector{Float64}, Nothing}
    n_features_in::Int
    n_targets::Int

    function PLSRegression(; n_components::Integer=2)
        n_components >= 1 || throw(ArgumentError(
            "n_components must be >= 1; got $n_components"))
        new(Int(n_components), nothing, nothing, nothing, 0, 0)
    end
end

"""
    fit!(model::PLSRegression, X, y) -> model

Train the model in-place. `X` is `(n, p)` and `y` is either a length-`n`
vector or an `(n, q)` matrix.
"""
function fit!(model::PLSRegression, X::AbstractMatrix{<:Real},
              y::AbstractVecOrMat{<:Real})
    X64 = Matrix{Float64}(X)
    Y64 = if y isa AbstractVector
        reshape(Vector{Float64}(y), :, 1)
    else
        Matrix{Float64}(y)
    end
    size(X64, 1) == size(Y64, 1) || throw(ArgumentError(
        "X and y must share the first dimension; got $(size(X64)) and $(size(Y64))"))
    res = Pls4all.pls_fit(X64, Y64; n_components=model.n_components)
    model.coefficients = res.coefficients
    model.x_mean = collect(res.x_mean)
    model.y_mean = collect(res.y_mean)
    model.n_features_in = size(X64, 2)
    model.n_targets = size(Y64, 2)
    return model
end

"""
    predict(model::PLSRegression, X) -> Vector or Matrix

Returns predictions on `X`. For single-target regression the returned
shape is `(n,)`; for multi-target it is `(n, q)`. The prediction math
mirrors the C ABI contract (cpp/src/core/model.cpp::fill_prediction):

    yhat = (X - x_mean) * coefficients + y_mean
"""
function predict(model::PLSRegression, X::AbstractMatrix{<:Real})
    model.coefficients === nothing && throw(ErrorException(
        "PLSRegression is not fitted yet"))
    size(X, 2) == model.n_features_in || throw(DimensionMismatch(
        "X has $(size(X, 2)) features, but model was fitted with $(model.n_features_in)"))
    X64 = Matrix{Float64}(X)
    yhat = (X64 .- transpose(model.x_mean)) * model.coefficients .+ transpose(model.y_mean)
    return model.n_targets == 1 ? vec(yhat) : yhat
end

"""
    score(model::PLSRegression, X, y) -> Float64 or Vector{Float64}

Coefficient of determination R² on `(X, y)`. Returns a scalar for
single-target regression, a vector for multi-target.
"""
function score(model::PLSRegression, X::AbstractMatrix{<:Real},
                y::AbstractVecOrMat{<:Real})
    yhat = predict(model, X)
    Yobs = y isa AbstractVector ? reshape(collect(Float64.(y)), :, 1) :
                                   Matrix{Float64}(y)
    Yhat = yhat isa AbstractVector ? reshape(yhat, :, 1) : yhat
    ss_res = sum((Yhat .- Yobs) .^ 2, dims=1)
    means = mean(Yobs; dims=1)
    ss_tot = sum((Yobs .- means) .^ 2, dims=1)
    r2 = vec(1 .- ss_res ./ ss_tot)
    return length(r2) == 1 ? r2[1] : r2
end

# `mean` is in Statistics; use it lazily to avoid hard-deps.
function mean(A; dims)
    return sum(A; dims=dims) ./ size(A, dims)
end

end # module Sklearn
