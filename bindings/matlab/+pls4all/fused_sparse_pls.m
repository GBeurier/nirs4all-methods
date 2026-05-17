function res = fused_sparse_pls(X, Y, n_components, l1_lambda, fusion_lambda)
% pls4all.fused_sparse_pls  Fused-sparse PLS (L1 + adjacent-coef smoothing).
if nargin < 4 || isempty(l1_lambda),     l1_lambda = 0.05;     end
if nargin < 5 || isempty(fusion_lambda), fusion_lambda = 0.05; end
params = struct("l1_lambda",     double(l1_lambda), ...
                "fusion_lambda", double(fusion_lambda));
res = pls4all.p4a_method_fit_mex("fused_sparse_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
