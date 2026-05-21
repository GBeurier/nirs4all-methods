function res = rosa(X, Y, n_components, block_sizes)
% pls4all.rosa  Response-Oriented Sequential Alternation (Liland & Næs 2016).
params = struct("block_sizes", int64(block_sizes(:)));
res = pls4all.n4m_method_fit_mex("rosa", double(X), double(Y), ...
                                  int32(n_components), params);
end
