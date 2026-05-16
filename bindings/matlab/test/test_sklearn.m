function test_sklearn()
% Smoke + bit-exact parity test for the pls4all tier-2 MATLAB/Octave
% wrapper (Regression class + fitrpls factory).

    addpath(fullfile(fileparts(mfilename("fullpath")), ".."));

    rng_seed = 42;
    if exist("rng", "file")
        rng(rng_seed);
    else
        rand("state", rng_seed); randn("state", rng_seed);
    end

    n = 100; p = 12;
    X = randn(n, p);
    beta = zeros(p, 1);
    beta([2 5 8 10]) = [2.0; -1.5; 1.0; -0.8];
    y = X * beta + 0.1 * randn(n, 1);

    fprintf("=== fitrpls + predict + score ===\n");
    mdl = pls4all.fitrpls(X, y, "NumComponents", 5);
    yhat = predict(mdl, X);
    r2 = mdl.score(X, y);
    fprintf("  in-sample R² = %.4f\n", r2);
    fprintf("  Coefficients size: %dx%d\n", ...
            size(mdl.Coefficients, 1), size(mdl.Coefficients, 2));

    if r2 < 0.9
        error("expected R² >= 0.9 on a clean synthetic regression");
    end

    fprintf("=== bit-exact tier1 vs tier2 ===\n");
    [coefs, x_mean, y_mean, preds_t1] = ...
        pls4all.pls_fit(X, y, int32(5));
    preds_t2 = predict(mdl, X);
    max_diff = max(abs(preds_t1(:, 1) - preds_t2(:)));
    fprintf("  max abs diff (tier1 vs tier2): %.2e\n", max_diff);
    if max_diff > 1e-12
        error("tier1 vs tier2 prediction divergence: %.4e", max_diff);
    end

    fprintf("=== feature count rejection ===\n");
    ok = false;
    try
        predict(mdl, randn(10, p + 2));
    catch
        ok = true;
    end
    if ~ok
        error("predict accepted wrong feature count");
    end
    fprintf("  rejected wrong feature count: OK\n");

    fprintf("=== loss ===\n");
    l = mdl.loss(X, y);
    fprintf("  in-sample MSE = %.6f\n", l);

    fprintf("\n=== ALL PASSED ===\n");
end
