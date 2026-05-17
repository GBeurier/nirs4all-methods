function res = one_se_rule(fold_rmse_matrix)
% pls4all.one_se_rule  One-SE component selection from a fold RMSE matrix.
% fold_rmse_matrix: (max_components × n_folds) matrix of fold RMSE values.
params = struct("fold_rmse_matrix", double(fold_rmse_matrix));
% Dispatcher requires X / Y / n_components positional args; provide stubs.
% n_components is unused on the C side for this entry.
n_k = size(fold_rmse_matrix, 1);
res = pls4all.p4a_method_fit_mex("one_se_rule_compute", ...
                                  zeros(2, 1), zeros(2, 1), ...
                                  int32(n_k), params);
end
