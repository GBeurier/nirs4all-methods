% Comprehensive smoke test for the full MATLAB / Octave tier-1 +
% tier-2 surface (33 method-result fits + 24 selectors + 2 transformers
% + 4 diagnostics). Each call is checked for "didn't throw + shapes
% look right"; numerical correctness is covered by the parity tests.

rand("seed", 0); randn("seed", 0);
n = 60; p = 30;
X = randn(n, p);
y = 2.0 * X(:, 3) - X(:, 6) + 0.5 * X(:, 9) + 0.05 * randn(n, 1);

% Integer labels for classifier-style fits.
y_cls = (X(:, 1) > 0) + (X(:, 2) > 0) * 2;  % 0..3 → 4 classes
y_cls = int32(y_cls);

ok = 0; fail = 0; fail_names = {};

function check(name, fn)
    try
        fn();
        printf("  %s OK\n", name);
        evalin("caller", "ok = ok + 1;");
    catch err
        printf("  %s FAIL: %s\n", name, err.message);
        evalin("caller", "fail = fail + 1;");
        evalin("caller", sprintf("fail_names{end+1} = '%s';", name));
    end
end

printf("=== MethodResult fits ===\n");
check("sparse_simpls",        @() pls4all.sparse_simpls(X, y, 5));
check("cppls",                @() pls4all.cppls(X, y, 5));
check("ecr",                  @() pls4all.ecr(X, y, 5));
check("weighted_pls",         @() pls4all.weighted_pls(X, y, 5, ones(n, 1)));
check("robust_pls",           @() pls4all.robust_pls(X, y, 5));
check("ridge_pls",            @() pls4all.ridge_pls(X, y, 5));
check("continuum_regression", @() pls4all.continuum_regression(X, y, 5));
check("recursive_pls",        @() pls4all.recursive_pls(X, y, 5, 20));
check("n_pls",                @() pls4all.n_pls(X, y, 3, 5, 6));
check("kernel_pls",           @() pls4all.kernel_pls(X, y, 5));
check("o2pls",                @() pls4all.o2pls(X, y, 2));
check("group_sparse_pls",     @() pls4all.group_sparse_pls(X, y, 5, floor((0:p-1)/5)));
check("fused_sparse_pls",     @() pls4all.fused_sparse_pls(X, y, 5));
check("bagging_pls",          @() pls4all.bagging_pls(X, y, 5, 10));
check("boosting_pls",         @() pls4all.boosting_pls(X, y, 5, 10));
check("random_subspace_pls",  @() pls4all.random_subspace_pls(X, y, 5, 10, 15));
check("mb_pls",               @() pls4all.mb_pls(X, y, 3, [15, 15]));
check("mir_pls",              @() pls4all.mir_pls(X, y, 5));
check("missing_aware_nipals", @() pls4all.missing_aware_nipals(X, y, 5));
check("pls_glm gaussian",     @() pls4all.pls_glm(X, y, 5, "gaussian"));
check("lw_pls",               @() pls4all.lw_pls(X, y, 3, 5));
check("sparse_pls_da",        @() pls4all.sparse_pls_da(X, y_cls, 3));
check("pls_qda",              @() pls4all.pls_qda(X, y_cls, 3));
check("pls_lda",              @() pls4all.pls_lda(X, y_cls, 3));
check("pls_logistic",         @() pls4all.pls_logistic(X, y_cls, 3));
check("pls_cox",              @() pls4all.pls_cox(X, 5, abs(y) + 0.5, ones(n, 1, "int32")));
check("di_pls",               @() pls4all.di_pls(X, y, 3, X(1:30, :)));
check("so_pls",               @() pls4all.so_pls(X, y, [2, 2], [15, 15]));
check("rosa",                 @() pls4all.rosa(X, y, 3, [15, 15]));
check("pds",                  @() pls4all.pds(X, X + 0.01 * randn(n, p), 2));
check("ds",                   @() pls4all.ds(X, X + 0.01 * randn(n, p)));
check("gpr_pls",              @() pls4all.gpr_pls(X, y, 3));

printf("\n=== Selectors ===\n");
check("spa_select",           @() pls4all.spa_select(X, y, 3, 5));
check("cars_select",          @() pls4all.cars_select(X, y, 3, 20, 5));
check("interval_select",      @() pls4all.interval_select(X, y, 3, 5, 1));
check("stability_select",     @() pls4all.stability_select(X, y, 3, 5));
check("uve_select",           @() pls4all.uve_select(X, y, 3));
check("random_frog_select",   @() pls4all.random_frog_select(X, y, 3, 30, 10, 3, 20, 5));
check("scars_select",         @() pls4all.scars_select(X, y, 3, 20, 5));
check("ga_select",            @() pls4all.ga_select(X, y, 3, 10, 20, 3));
check("pso_select",           @() pls4all.pso_select(X, y, 3, 10, 10));
check("vissa_select",         @() pls4all.vissa_select(X, y, 3, 5, 20));
check("shaving_select",       @() pls4all.shaving_select(X, y, 3));
check("bve_select",           @() pls4all.bve_select(X, y, 3));
check("t2_select",            @() pls4all.t2_select(X, y, 3, [0.05, 0.1], 3));
check("wvc_select",           @() pls4all.wvc_select(X, y, 3, 5));
check("wvc_threshold_select", @() pls4all.wvc_threshold_select(X, y, 3, 1, 0.1));
check("emcuve_select",        @() pls4all.emcuve_select(X, y, 3));
check("randomization_select", @() pls4all.randomization_select(X, y, 3, 30));
check("bipls_select",         @() pls4all.bipls_select(X, y, 3, 5));
check("sipls_select",         @() pls4all.sipls_select(X, y, 3, 5, 2));
check("rep_select",           @() pls4all.rep_select(X, y, 3));
check("ipw_select",           @() pls4all.ipw_select(X, y, 3));
check("st_select",            @() pls4all.st_select(X, y, 3, [0.1, 0.2, 0.3], 3));
check("iriv_select",          @() pls4all.iriv_select(X, y, 3, 10));
check("irf_select",           @() pls4all.irf_select(X, y, 3, 20, 5, 5, 2));
check("vip_spa_select",       @() pls4all.vip_spa_select(X, y, 3));

printf("\n=== Diagnostics ===\n");
check("approximate_press",    @() pls4all.approximate_press(X, y, 8));
check("one_se_rule",          @() pls4all.one_se_rule(rand(10, 5)));

printf("\n=== Tier-2 classdefs via fit() factory ===\n");
check("fit pls",          @() predict(pls4all.fit("pls", X, y, "NumComponents", 5), X));
check("fit sparse_simpls", @() predict(pls4all.fit("sparse_simpls", X, y, "NumComponents", 5), X));
check("fit cppls",        @() predict(pls4all.fit("cppls", X, y, "NumComponents", 5), X));
check("fit ecr",          @() predict(pls4all.fit("ecr", X, y, "NumComponents", 5, "Alpha", 0.3), X));
check("fit weighted_pls", @() predict(pls4all.fit("weighted_pls", X, y, "NumComponents", 5, "Weights", ones(n, 1)), X));
check("fit robust_pls",   @() predict(pls4all.fit("robust_pls", X, y, "NumComponents", 5), X));
check("fit ridge_pls",    @() predict(pls4all.fit("ridge_pls", X, y, "NumComponents", 5, "Lambda", 0.1), X));
check("fit mir_pls",      @() predict(pls4all.fit("mir_pls", X, y, "NumComponents", 5), X));
check("fit n_pls",        @() predict(pls4all.fit("n_pls", X, y, "NumComponents", 3, "ModeJ", 5, "ModeK", 6), X));
check("fit o2pls",        @() predict(pls4all.fit("o2pls", X, y, "NPredictive", 2), X));
check("fit mb_pls",       @() predict(pls4all.fit("mb_pls", X, y, "NumComponents", 3, "BlockSizes", [15, 15]), X));
check("fit pls_glm",      @() predict(pls4all.fit("pls_glm", X, y, "NumComponents", 5), X));
check("fit bagging_pls",  @() predict(pls4all.fit("bagging_pls", X, y, "NumComponents", 5, "NEstimators", 10), X));
check("fit boosting_pls", @() predict(pls4all.fit("boosting_pls", X, y, "NumComponents", 5, "NEstimators", 10), X));

printf("\n=== TOTAL: %d passed, %d failed ===\n", ok, fail);
if fail > 0
    for k = 1:length(fail_names)
        printf("  FAILED: %s\n", fail_names{k});
    end
    error("pls4all:smoke", "Some smoke tests failed");
end
printf("=== ALL MATLAB FULL-SURFACE SMOKE TESTS PASSED ===\n");
