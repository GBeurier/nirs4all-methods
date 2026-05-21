function res = n_pls(X_flat, Y, n_components, mode_j, mode_k)
% pls4all.n_pls  N-PLS (3-way tensor) regression.
% X_flat: n × (mode_j * mode_k) flattened row-major.
params = struct("mode_j", int32(mode_j), "mode_k", int32(mode_k));
res = pls4all.n4m_method_fit_mex("n_pls", double(X_flat), double(Y), ...
                                  int32(n_components), params);
end
