function res = sparse_pls_da(X, y_labels, n_components, sparsity_lambda)
% pls4all.sparse_pls_da  Sparse PLS-DA classifier (Chun & Keles 2010 + DA).
% y_labels: integer class IDs in {0, …, n_classes-1}.
if nargin < 4 || isempty(sparsity_lambda), sparsity_lambda = 0.05; end
params = struct("y_labels", int32(y_labels(:)), ...
                "sparsity_lambda", double(sparsity_lambda));
% X is the only matrix here; Y is unused on the C side but the dispatcher
% still expects a placeholder.
res = pls4all.p4a_method_fit_mex("sparse_pls_da", double(X), zeros(size(X,1), 1), ...
                                  int32(n_components), params);
end
