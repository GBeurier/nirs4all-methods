function res = robust_pls(X, Y, n_components, huber_k, max_irls_iter)
% pls4all.robust_pls  Robust PLS via Huber IRLS over SIMPLS.
if nargin < 4 || isempty(huber_k), huber_k = 1.345; end
if nargin < 5 || isempty(max_irls_iter), max_irls_iter = 20; end
params = struct("huber_k", double(huber_k), ...
                "max_irls_iter", int32(max_irls_iter));
res = pls4all.n4m_method_fit_mex("robust_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
