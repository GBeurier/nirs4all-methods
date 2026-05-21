function res = wvc_select(X, Y, n_components, top_k, normalize)
% pls4all.wvc_select  Weighted-vector correlation top-k selector.
if nargin < 5 || isempty(normalize), normalize = 1; end
params = struct("top_k", int32(top_k), "normalize", int32(normalize));
res = pls4all.n4m_method_fit_mex("wvc_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
