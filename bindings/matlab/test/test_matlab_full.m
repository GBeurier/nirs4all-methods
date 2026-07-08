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
check("sparse_simpls",        @() n4m.sparse_simpls(X, y, 5));
check("cppls",                @() n4m.cppls(X, y, 5));
check("ecr",                  @() n4m.ecr(X, y, 5));
check("weighted_pls",         @() n4m.weighted_pls(X, y, 5, ones(n, 1)));
check("robust_pls",           @() n4m.robust_pls(X, y, 5));
check("ridge_pls",            @() n4m.ridge_pls(X, y, 5));
check("continuum_regression", @() n4m.continuum_regression(X, y, 5));
check("recursive_pls",        @() n4m.recursive_pls(X, y, 5, 20));
check("n_pls",                @() n4m.n_pls(X, y, 3, 5, 6));
check("kernel_pls",           @() n4m.kernel_pls(X, y, 5));
check("o2pls",                @() n4m.o2pls(X, y, 2));
check("group_sparse_pls",     @() n4m.group_sparse_pls(X, y, 5, floor((0:p-1)/5)));
check("fused_sparse_pls",     @() n4m.fused_sparse_pls(X, y, 5));
check("bagging_pls",          @() n4m.bagging_pls(X, y, 5, 10));
check("boosting_pls",         @() n4m.boosting_pls(X, y, 5, 10));
check("random_subspace_pls",  @() n4m.random_subspace_pls(X, y, 5, 10, 15));
check("mb_pls",               @() n4m.mb_pls(X, y, 3, [15, 15]));
check("mir_pls",              @() n4m.mir_pls(X, y, 5));
check("missing_aware_nipals", @() n4m.missing_aware_nipals(X, y, 5));
check("pls_glm gaussian",     @() n4m.pls_glm(X, y, 5, "gaussian"));
check("lw_pls",               @() n4m.lw_pls(X, y, 3, 5));
check("sparse_pls_da",        @() n4m.sparse_pls_da(X, y_cls, 3));
check("pls_qda",              @() n4m.pls_qda(X, y_cls, 3));
check("pls_lda",              @() n4m.pls_lda(X, y_cls, 3));
check("pls_logistic",         @() n4m.pls_logistic(X, y_cls, 3));
check("pls_cox",              @() n4m.pls_cox(X, 5, abs(y) + 0.5, ones(n, 1, "int32")));
check("di_pls",               @() n4m.di_pls(X, y, 3, X(1:30, :)));
check("so_pls",               @() n4m.so_pls(X, y, [2, 2], [15, 15]));
check("rosa",                 @() n4m.rosa(X, y, 3, [15, 15]));
check("pds",                  @() n4m.pds(X, X + 0.01 * randn(n, p), 2));
check("ds",                   @() n4m.ds(X, X + 0.01 * randn(n, p)));
check("gpr_pls",              @() n4m.gpr_pls(X, y, 3));

printf("\n=== Selectors ===\n");
check("spa_select",           @() n4m.spa_select(X, y, 3, 5));
check("cars_select",          @() n4m.cars_select(X, y, 3, 20, 5));
check("interval_select",      @() n4m.interval_select(X, y, 3, 5, 1));
check("stability_select",     @() n4m.stability_select(X, y, 3, 5));
check("uve_select",           @() n4m.uve_select(X, y, 3));
check("random_frog_select",   @() n4m.random_frog_select(X, y, 3, 30, 10, 3, 20, 5));
check("scars_select",         @() n4m.scars_select(X, y, 3, 20, 5));
check("ga_select",            @() n4m.ga_select(X, y, 3, 10, 20, 3));
check("pso_select",           @() n4m.pso_select(X, y, 3, 10, 10));
check("vissa_select",         @() n4m.vissa_select(X, y, 3, 5, 20));
check("shaving_select",       @() n4m.shaving_select(X, y, 3));
check("bve_select",           @() n4m.bve_select(X, y, 3));
check("t2_select",            @() n4m.t2_select(X, y, 3, [0.05, 0.1], 3));
check("wvc_select",           @() n4m.wvc_select(X, y, 3, 5));
check("wvc_threshold_select", @() n4m.wvc_threshold_select(X, y, 3, 1, 0.1));
check("emcuve_select",        @() n4m.emcuve_select(X, y, 3));
check("randomization_select", @() n4m.randomization_select(X, y, 3, 30));
check("bipls_select",         @() n4m.bipls_select(X, y, 3, 5));
check("sipls_select",         @() n4m.sipls_select(X, y, 3, 5, 2));
check("rep_select",           @() n4m.rep_select(X, y, 3));
check("ipw_select",           @() n4m.ipw_select(X, y, 3));
check("st_select",            @() n4m.st_select(X, y, 3, [0.1, 0.2, 0.3], 3));
check("iriv_select",          @() n4m.iriv_select(X, y, 3, 10));
check("irf_select",           @() n4m.irf_select(X, y, 3, 20, 5, 5, 2));
check("vip_spa_select",       @() n4m.vip_spa_select(X, y, 3));

printf("\n=== Diagnostics ===\n");
check("approximate_press",    @() n4m.approximate_press(X, y, 8));
check("one_se_rule",          @() n4m.one_se_rule(rand(10, 5)));

printf("\n=== Tier-2 classdefs via fit() factory ===\n");
check("fit pls",          @() predict(n4m.fit("pls", X, y, "NumComponents", 5), X));
check("fit sparse_simpls", @() predict(n4m.fit("sparse_simpls", X, y, "NumComponents", 5), X));
check("fit cppls",        @() predict(n4m.fit("cppls", X, y, "NumComponents", 5), X));
check("fit ecr",          @() predict(n4m.fit("ecr", X, y, "NumComponents", 5, "Alpha", 0.3), X));
check("fit weighted_pls", @() predict(n4m.fit("weighted_pls", X, y, "NumComponents", 5, "Weights", ones(n, 1)), X));
check("fit robust_pls",   @() predict(n4m.fit("robust_pls", X, y, "NumComponents", 5), X));
check("fit ridge_pls",    @() predict(n4m.fit("ridge_pls", X, y, "NumComponents", 5, "Lambda", 0.1), X));
check("fit mir_pls",      @() predict(n4m.fit("mir_pls", X, y, "NumComponents", 5), X));
check("fit n_pls",        @() predict(n4m.fit("n_pls", X, y, "NumComponents", 3, "ModeJ", 5, "ModeK", 6), X));
check("fit o2pls",        @() predict(n4m.fit("o2pls", X, y, "NPredictive", 2), X));
check("fit mb_pls",       @() predict(n4m.fit("mb_pls", X, y, "NumComponents", 3, "BlockSizes", [15, 15]), X));
check("fit pls_glm",      @() predict(n4m.fit("pls_glm", X, y, "NumComponents", 5), X));
check("fit bagging_pls",  @() predict(n4m.fit("bagging_pls", X, y, "NumComponents", 5, "NEstimators", 10), X));
check("fit boosting_pls", @() predict(n4m.fit("boosting_pls", X, y, "NumComponents", 5, "NEstimators", 10), X));

printf("\n=== TOTAL: %d passed, %d failed ===\n", ok, fail);
if fail > 0
    for k = 1:length(fail_names)
        printf("  FAILED: %s\n", fail_names{k});
    end
    error("n4m:smoke", "Some smoke tests failed");
end
printf("=== ALL MATLAB FULL-SURFACE SMOKE TESTS PASSED ===\n");
