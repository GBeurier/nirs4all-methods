% Smoke test for the new MATLAB tier-1 dispatcher MEX + tier-2 classes.
% Run with Octave:
%   PATH=$(dirname $(which octave)):$PATH \
%     LD_LIBRARY_PATH=$REPO/build/dev-release/cpp/src:$LD_LIBRARY_PATH \
%     octave --eval "addpath(genpath('bindings/matlab')); test_matlab_tier2"

% Deterministic synthetic regression.
rand("seed", 0); randn("seed", 0);
n = 50; p = 20;
X = randn(n, p);
y = 2.0 * X(:, 3) - X(:, 6) + 0.5 * X(:, 9) + 0.05 * randn(n, 1);

% Tier-1 round-trip via dispatcher.
printf("--- pls4all.sparse_simpls ---\n");
res = pls4all.sparse_simpls(X, y, 5, 0.05);
if size(res.coefficients, 1) ~= p, error("sparse_simpls coef shape wrong"); end
printf("  coef shape = (%d, %d), rmse = %.4f\n", ...
       size(res.coefficients, 1), size(res.coefficients, 2), res.rmse);

printf("--- pls4all.cppls ---\n");
res = pls4all.cppls(X, y, 5, 0.5);
printf("  rmse = %.4f\n", res.rmse);

printf("--- pls4all.weighted_pls ---\n");
w = 0.5 + rand(n, 1);
res = pls4all.weighted_pls(X, y, 5, w);
printf("  rmse = %.4f\n", res.rmse);

printf("--- pls4all.mb_pls ---\n");
res = pls4all.mb_pls(X, y, 3, [10, 10]);
printf("  rmse = %.4f, block_weights = [%s]\n", ...
       res.rmse, num2str(res.block_weights));

printf("--- pls4all.pls_glm (gaussian) ---\n");
res = pls4all.pls_glm(X, y, 5, "gaussian");
printf("  rmse = %.4f\n", res.rmse);

printf("--- pls4all.mir_pls ---\n");
res = pls4all.mir_pls(X, y, 5);
printf("  rmse = %.4f\n", res.rmse);

printf("--- pls4all.spa_select ---\n");
res = pls4all.spa_select(X, y, 5, 5);
printf("  indices = %s, best_rmse = %.4f\n", ...
       num2str(res.selected_indices), res.best_rmse);

printf("--- pls4all.approximate_press ---\n");
res = pls4all.approximate_press(X, y, 10);
printf("  selected k = %d, rmse curve = %s\n", ...
       res.selected_n_components, sprintf("%.3f ", res.rmse_per_component));

% Tier-2 idiomatic classdef.
printf("\n--- Tier-2 classdef: pls4all.fit('sparse_simpls', ...) ---\n");
mdl = pls4all.fit("sparse_simpls", X, y, "NumComponents", 5, "Lambda", 0.05);
yhat = predict(mdl, X);
printf("  class = %s, NumPredictors = %d, NumResponses = %d\n", ...
       class(mdl), mdl.NumPredictors, mdl.NumResponses);
printf("  R² = %.4f\n", score(mdl, X, y));

mdl = pls4all.fit("cppls", X, y, "NumComponents", 5, "Gamma", 0.5);
printf("  cppls R² = %.4f\n", score(mdl, X, y));

mdl = pls4all.fit("mb_pls", X, y, "NumComponents", 3, "BlockSizes", [10, 10]);
printf("  mb_pls R² = %.4f (uses intercept = %d)\n", ...
       score(mdl, X, y), mdl.UseIntercept);

printf("\n=== ALL MATLAB TIER-1 + TIER-2 SMOKE TESTS PASSED ===\n");
