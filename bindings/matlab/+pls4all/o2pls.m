function res = o2pls(X, Y, n_predictive, n_x_orthogonal, n_y_orthogonal)
% pls4all.o2pls  O2-PLS (bi-directional OPLS, Trygg & Wold 2003).
if nargin < 4 || isempty(n_x_orthogonal), n_x_orthogonal = 1; end
if nargin < 5 || isempty(n_y_orthogonal), n_y_orthogonal = 1; end
params = struct("n_predictive",   int32(n_predictive), ...
                "n_x_orthogonal", int32(n_x_orthogonal), ...
                "n_y_orthogonal", int32(n_y_orthogonal));
res = pls4all.n4m_method_fit_mex("o2pls", double(X), double(Y), ...
                                  int32(n_predictive), params);
end
