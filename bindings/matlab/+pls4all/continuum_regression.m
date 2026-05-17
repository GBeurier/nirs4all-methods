function res = continuum_regression(X, Y, n_components, tau)
% pls4all.continuum_regression  Continuum regression (tau=0 → OLS, tau=1 → PLS).
if nargin < 4 || isempty(tau), tau = 0.5; end
params = struct("tau", double(tau));
res = pls4all.p4a_method_fit_mex("continuum_regression", double(X), double(Y), ...
                                  int32(n_components), params);
end
