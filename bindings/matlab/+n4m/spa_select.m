function res = spa_select(X, Y, n_components, top_k)
% n4m.spa_select  Successive Projections Algorithm.
%
%   res = n4m.spa_select(X, Y, K, top_k)
%
% Output struct fields:
%   selected_indices : 1 × top_k row vector of 1-based feature indices.
%   best_rmse        : scalar best in-sample RMSE on the selected subset.
if nargin < 4
    error("n4m:nargin", "top_k is required");
end
params = struct("top_k", int32(top_k));
res = n4m.n4m_method_fit_mex("spa_select", double(X), double(Y), ...
                                  int32(n_components), params);
% The MEX dispatcher converts *_indices fields to 1-based MATLAB
% indices on the way out; no further adjustment needed here.
end
