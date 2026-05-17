function res = spa_select(X, Y, n_components, top_k)
% pls4all.spa_select  Successive Projections Algorithm.
%
%   res = pls4all.spa_select(X, Y, K, top_k)
%
% Output struct fields:
%   selected_indices : 1 × top_k row vector of 1-based feature indices.
%   best_rmse        : scalar best in-sample RMSE on the selected subset.
if nargin < 4
    error("pls4all:nargin", "top_k is required");
end
params = struct("top_k", int32(top_k));
res = pls4all.p4a_method_fit_mex("spa_select", double(X), double(Y), ...
                                  int32(n_components), params);
% C side returns 0-based; expose 1-based for MATLAB idiom.
if isfield(res, "selected_indices")
    res.selected_indices = res.selected_indices + 1;
end
end
