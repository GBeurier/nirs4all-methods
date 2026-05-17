function res = bagging_pls(X, Y, n_components, n_estimators, seed)
% pls4all.bagging_pls  Bootstrap aggregation of n_estimators PLS regressors.
if nargin < 4 || isempty(n_estimators), n_estimators = 50; end
if nargin < 5 || isempty(seed),         seed = 0;          end
params = struct("n_estimators", int32(n_estimators), "seed", uint64(seed));
res = pls4all.p4a_method_fit_mex("bagging_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
