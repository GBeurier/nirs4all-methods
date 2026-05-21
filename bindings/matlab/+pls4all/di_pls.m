function res = di_pls(X_source, Y_source, n_components, X_target, di_lambda)
% pls4all.di_pls  Domain-Invariant PLS (Nikzad-Langerodi 2018).
if nargin < 5 || isempty(di_lambda), di_lambda = 1.0; end
params = struct("X_target", double(X_target), "di_lambda", double(di_lambda));
res = pls4all.n4m_method_fit_mex("di_pls", double(X_source), double(Y_source), ...
                                  int32(n_components), params);
end
