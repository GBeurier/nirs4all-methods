function res = weighted_pls(X, Y, n_components, sample_weights)
% n4m.weighted_pls  sqrt(w)-prescaled SIMPLS.
%
%   res = n4m.weighted_pls(X, Y, K, w)
%
% w must be a length-size(X, 1) strictly positive vector.
if nargin < 4
    error("n4m:nargin", ...
          "Usage: n4m.weighted_pls(X, Y, K, sample_weights)");
end
params = struct("sample_weights", double(sample_weights(:)));
res = n4m.n4m_method_fit_mex("weighted_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
