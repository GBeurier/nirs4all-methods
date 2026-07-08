function res = ecr(X, Y, n_components, alpha)
% n4m.ecr  Elastic Component Regression (alpha=0 → PCR-like,
% alpha=1 → PLS-like, default 0.5).
if nargin < 4 || isempty(alpha), alpha = 0.5; end
params = struct("alpha", double(alpha));
res = n4m.n4m_method_fit_mex("ecr", double(X), double(Y), ...
                                  int32(n_components), params);
end
