function res = lw_pls(X, Y, n_components, n_neighbors)
% pls4all.lw_pls  Locally-weighted PLS (Næs & Centner 1998).
if nargin < 4 || isempty(n_neighbors), n_neighbors = 30; end
params = struct("n_neighbors", int32(n_neighbors));
res = pls4all.n4m_method_fit_mex("lw_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
