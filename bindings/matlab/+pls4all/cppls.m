function res = cppls(X, Y, n_components, gamma)
% pls4all.cppls  Canonical Powered PLS (Indahl 2005).
%
%   res = pls4all.cppls(X, Y, K, gamma)
%
% gamma ∈ [0, 1] (default 0.5). 0 → SIMPLS, 1 → fully power-rescaled.
if nargin < 4 || isempty(gamma), gamma = 0.5; end
params = struct("gamma", double(gamma));
res = pls4all.n4m_method_fit_mex("cppls", double(X), double(Y), ...
                                  int32(n_components), params);
end
