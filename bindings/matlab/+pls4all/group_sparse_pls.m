function res = group_sparse_pls(X, Y, n_components, group_assignment, group_lambda)
% pls4all.group_sparse_pls  Group-sparse PLS (group L1 over feature groups).
if nargin < 5 || isempty(group_lambda), group_lambda = 0.05; end
params = struct("group_assignment", int32(group_assignment(:)), ...
                "group_lambda",     double(group_lambda));
res = pls4all.n4m_method_fit_mex("group_sparse_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
