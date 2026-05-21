# SPDX-License-Identifier: CECILL-2.1

using Pls4all
using Pls4all.Sklearn
using MLJModelInterface
using Test

const MMI = MLJModelInterface

function _table_toy()
    x1 = [sin(i * 0.2) + 0.05i for i in 1:24]
    x2 = [cos(i * 0.3) - 0.02i for i in 1:24]
    x3 = [sin(i * 0.5) + cos(i * 0.1) for i in 1:24]
    x4 = [0.1i - sin(i * 0.4) for i in 1:24]
    y = 0.7 .* x1 .- 0.3 .* x3 .+ 0.05 .* x4
    table = (x1=x1, x2=x2, x3=x3, x4=x4)
    shuffled = (x4=x4, x2=x2, x1=x1, x3=x3)
    return table, shuffled, y
end

@testset "Tables.jl inputs" begin
    table, shuffled, y = _table_toy()

    X = table_matrix(table)
    @test size(X) == (24, 4)
    @test response_matrix(y) == reshape(Float64.(y), :, 1)

    fit = Pls4all.pls_fit(table, y; n_components=2)
    @test fit.feature_names == [:x1, :x2, :x3, :x4]
    @test size(fit.predictions) == (24, 1)

    reg = PLSRegression(n_components=2)
    fit!(reg, table, y)
    pred = Pls4all.Sklearn.predict(reg, table)
    pred_shuffled = Pls4all.Sklearn.predict(reg, shuffled)
    @test length(pred) == 24
    @test pred ≈ pred_shuffled
end

@testset "MLJModelInterface regressors" begin
    table, shuffled, y = _table_toy()

    for model in (PLSRegressor(n_components=2),
                  PCRRegressor(n_components=2),
                  OPLSRegressor(n_components=1))
        fitresult, cache, report = MMI.fit(model, 0, table, y)
        @test cache === nothing
        @test length(report.training_predictions) == length(y)
        pred = MMI.predict(model, fitresult, shuffled)
        @test length(pred) == length(y)
        @test all(isfinite, pred)

        params = MMI.fitted_params(model, fitresult)
        @test params.feature_names == [:x1, :x2, :x3, :x4]
        @test size(params.coefficients, 1) == 4
        @test length(params.x_mean) == 4
        close(fitresult.native)
    end

    @test MMI.input_scitype(PLSRegressor) == MMI.Table(MMI.Continuous)
    @test MMI.target_scitype(PLSRegressor) == AbstractVector{MMI.Continuous}
    @test MMI.load_path(PLSRegressor) == "Pls4all.PLSRegressor"
end
