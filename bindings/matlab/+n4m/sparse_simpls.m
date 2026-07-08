function res = sparse_simpls(X, Y, n_components, sparsity_lambda)
% n4m.sparse_simpls  Sparse SIMPLS PLS regression (Chun & Keles 2010).
%
%   res = n4m.sparse_simpls(X, Y, K)
%   res = n4m.sparse_simpls(X, Y, K, lambda)
%
% Inputs:
%   X            n × p double matrix.
%   Y            n × q double matrix or n × 1 vector.
%   n_components positive integer.
%   sparsity_lambda (optional, default 0.05) — soft-threshold magnitude.
%
% Output: struct with fields
%   coefficients (p × q)
%   predictions  (n × q)  in-sample
%   x_mean       (1 × p)
%   y_mean       (1 × q)
%   weights_w    (p × n_components)
%   rmse         scalar (in-sample RMSE)
if nargin < 3
    error("n4m:nargin", "Usage: n4m.sparse_simpls(X, Y, K, lambda)");
end
if nargin < 4 || isempty(sparsity_lambda)
    sparsity_lambda = 0.05;
end
params = struct("sparsity_lambda", double(sparsity_lambda));
res = n4m.n4m_method_fit_mex("sparse_simpls", double(X), double(Y), ...
                                  int32(n_components), params);
end
