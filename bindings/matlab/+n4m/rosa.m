function res = rosa(X, Y, n_components, block_sizes)
% n4m.rosa  Response-Oriented Sequential Alternation (Liland & Næs 2016).
params = struct("block_sizes", int64(block_sizes(:)));
res = n4m.n4m_method_fit_mex("rosa", double(X), double(Y), ...
                                  int32(n_components), params);
end
