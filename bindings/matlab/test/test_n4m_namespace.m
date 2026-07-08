function test_n4m_namespace()
% test_n4m_namespace  Smoke the V1 +n4m namespace.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, "bindings", "matlab"));

rand("seed", 1); randn("seed", 1);
n = 24; p = 6;
X = randn(n, p);
y = X(:, 1) - 0.25 * X(:, 3) + 0.01 * randn(n, 1);

v = n4m.version();
if isempty(v)
    error("n4m_namespace:version", "n4m.version returned an empty string");
end

[coefs, x_mean, y_mean, preds] = n4m.pls_fit(X, y, 2);

tol = 1e-12;
if any(~isfinite(coefs(:))) || any(~isfinite(x_mean(:))) || ...
   any(~isfinite(y_mean(:))) || any(~isfinite(preds(:)))
    error("n4m_namespace:finite", "n4m.pls_fit returned non-finite values");
end
expected_preds = (X - x_mean) * coefs + y_mean;
if max(abs(preds(:) - expected_preds(:))) > tol
    error("n4m_namespace:prediction", "n4m.pls_fit predictions are inconsistent");
end

mdl = n4m.fitrpls(X, y, "NumComponents", 2);
if ~strcmp(class(mdl), "n4m.Regression")
    error("n4m_namespace:class", "n4m.fitrpls returned %s", class(mdl));
end
pred = predict(mdl, X);
if numel(pred) ~= n || any(~isfinite(pred(:)))
    error("n4m_namespace:predict", "n4m.Regression predict output is invalid");
end

factory_mdl = n4m.fit("pls", X, y, "NumComponents", 2);
if ~strcmp(class(factory_mdl), "n4m.Regression")
    error("n4m_namespace:factory", "n4m.fit returned %s", class(factory_mdl));
end

fprintf("n4m namespace smoke OK (%s)\n", v);
end
