function res = pls_glm(X, Y, n_components, family)
% pls4all.pls_glm  PLS + Generalised Linear Model head (Gaussian / Poisson).
%
%   res = pls4all.pls_glm(X, Y, K)              % gaussian (default)
%   res = pls4all.pls_glm(X, Y, K, "poisson")   % poisson IRLS
if nargin < 4 || isempty(family), family = "gaussian"; end
if isstring(family), family = char(family); end
poisson = strcmpi(family, "poisson");
params = struct("poisson", int32(poisson));
res = pls4all.n4m_method_fit_mex("pls_glm", double(X), double(Y), ...
                                  int32(n_components), params);
end
