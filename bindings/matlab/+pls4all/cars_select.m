function res = cars_select(X, Y, n_components, n_iterations, min_features)
% pls4all.cars_select  Competitive Adaptive Reweighted Sampling.
%
%   res = pls4all.cars_select(X, Y, K, n_iter, min_feats)
%
% Uses the default (NULL) ValidationPlan on the C side (5-fold fallback).
if nargin < 4 || isempty(n_iterations), n_iterations = 50; end
if nargin < 5 || isempty(min_features),  min_features = 5;  end
params = struct("n_iterations", int32(n_iterations), ...
                "min_features", int32(min_features));
res = pls4all.n4m_method_fit_mex("cars_select", double(X), double(Y), ...
                                  int32(n_components), params);
% indices are already 1-based after the MEX dispatcher conversion.
end
