# SPDX-License-Identifier: CECILL-2.1

using Pls4all
using Test

const JULIA_METHOD_SURFACE = Symbol[
    :pls_fit, :model_fit, :model_array, :model_predict,
    :sparse_simpls_fit, :di_pls_fit, :recursive_pls_run, :cppls_fit,
    :weighted_pls_fit, :robust_pls_fit, :ridge_pls_fit,
    :continuum_regression_fit, :n_pls_fit, :kernel_pls_fit, :o2pls_fit,
    :approximate_press_compute, :pls_diagnostics_compute,
    :sparse_pls_da_fit, :group_sparse_pls_fit, :fused_sparse_pls_fit,
    :pls_monitoring_run, :one_se_rule_compute, :so_pls_fit, :on_pls_fit,
    :rosa_fit, :bagging_pls_fit, :gpr_pls_fit, :boosting_pls_fit,
    :random_subspace_pls_fit, :pls_glm_fit, :pls_qda_fit, :pls_cox_fit,
    :pds_fit, :ds_fit, :mir_pls_fit, :missing_aware_nipals_fit,
    :mb_pls_fit, :lw_pls_fit, :pls_lda_fit, :pls_logistic_fit,
    :aom_preprocess_fit, :aom_global_select, :aom_per_component_select,
    :variable_select_rank, :interval_select, :stability_select, :uve_select,
    :spa_select, :cars_select, :random_frog_select, :scars_select,
    :ga_select, :pso_select, :vissa_select, :shaving_select, :bve_select,
    :t2_select, :wvc_select, :wvc_threshold_select, :emcuve_select,
    :randomization_select, :bipls_select, :sipls_select, :rep_select,
    :ipw_select, :st_select, :ecr_fit, :iriv_select, :irf_select,
    :vip_spa_select,
]

function _toy()
    X = [sin(i * j * 0.21) + 0.1i + 0.03j for i in 1:12, j in 1:6]
    Y = reshape([0.4X[i, 1] - 0.2X[i, 3] + 0.05i for i in 1:12], :, 1)
    return X, Y
end

@testset "Julia method binding surface" begin
    for name in JULIA_METHOD_SURFACE
        @test isdefined(Pls4all, name)
    end

    X, Y = _toy()
    ctx = Context()
    cfg = Config()
    set_algorithm!(cfg, 0)
    set_solver!(cfg, 1)
    set_deflation!(cfg, 0)
    set_n_components!(cfg, 2)
    set_store_scores!(cfg, true)
    try
        r = sparse_simpls_fit(ctx, cfg, X, Y, 0.01)
        try
            @test size(matrix(r, "predictions")) == size(Y)
        finally
            close(r)
        end

        m = model_fit(ctx, cfg, X, Y)
        try
            @test size(model_predict(ctx, m, X)) == size(Y)
        finally
            close(m)
        end

        r = vip_spa_select(ctx, cfg, X, Y, 3)
        try
            @test length(vector_int64(r, "selected_indices")) > 0
        finally
            close(r)
        end

        plan = ValidationPlan(size(X, 1))
        add_fold!(plan, 0:5, 6:11)
        add_fold!(plan, 6:11, 0:5)
        bank = OperatorBank()
        add_operator!(bank, 0)
        add_operator!(bank, 15, [1.0])
        try
            aom = aom_global_select(ctx, cfg, bank, X, Y, plan, 2)
            try
                @test size(predictions(aom)) == size(Y)
            finally
                close(aom)
            end
        finally
            close(bank)
            close(plan)
        end
    finally
        close(cfg)
        close(ctx)
    end
end
