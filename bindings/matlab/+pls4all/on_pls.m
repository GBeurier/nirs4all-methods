function res = on_pls(X, Y, n_joint, n_unique_per_block, block_sizes)
% pls4all.on_pls  Orthogonal multi-block PLS (joint + unique loadings).
% n_components arg is unused by on_pls (it has its own block structure),
% but the dispatcher still requires it; we pass n_joint.
params = struct("block_sizes",          int64(block_sizes(:)), ...
                "n_joint",              int32(n_joint), ...
                "n_unique_per_block",   int32(n_unique_per_block(:)));
res = pls4all.n4m_method_fit_mex("on_pls", double(X), double(Y), ...
                                  int32(n_joint), params);
end
