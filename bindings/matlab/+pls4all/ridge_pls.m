function res = ridge_pls(X, Y, n_components, ridge_lambda)
% pls4all.ridge_pls  L2-augmented PLS regression.
if nargin < 4 || isempty(ridge_lambda), ridge_lambda = 1.0; end
params = struct("ridge_lambda", double(ridge_lambda));
res = pls4all.p4a_method_fit_mex("ridge_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
