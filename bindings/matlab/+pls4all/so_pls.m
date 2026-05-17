function res = so_pls(X, Y, n_components_per_block, block_sizes)
% pls4all.so_pls  Sequential & Orthogonalised multi-block PLS (Næs 2011).
% X: n × sum(block_sizes) concatenated multi-block matrix.
% n_components_per_block: int32 vector of length n_blocks.
nc_total = int32(sum(n_components_per_block));
params = struct("block_sizes",            int64(block_sizes(:)), ...
                "n_components_per_block", int32(n_components_per_block(:)));
res = pls4all.p4a_method_fit_mex("so_pls", double(X), double(Y), ...
                                  nc_total, params);
end
