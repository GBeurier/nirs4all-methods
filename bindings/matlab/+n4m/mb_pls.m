function res = mb_pls(X, Y, n_components, block_sizes)
% n4m.mb_pls  Multi-block PLS (block-weighted NIPALS).
%
%   res = n4m.mb_pls(X, Y, K, block_sizes)
%
% block_sizes : integer vector summing to size(X, 2).
%
% The returned struct exposes `block_weights` per block plus the
% per-block intercept-folded coefficients (predict via
% `X @ coef + intercept`, no centering).
if nargin < 4
    error("n4m:nargin", "block_sizes is required");
end
params = struct("block_sizes", int32(block_sizes(:)));
res = n4m.n4m_method_fit_mex("mb_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
