function res = uve_select(X, Y, n_components, noise_features, noise_seed)
% pls4all.uve_select  Uninformative Variable Elimination (artificial-noise threshold).
if nargin < 4 || isempty(noise_features), noise_features = size(X, 2); end
if nargin < 5 || isempty(noise_seed),     noise_seed = 0;              end
params = struct("noise_features", int32(noise_features), ...
                "noise_seed",     uint64(noise_seed));
res = pls4all.n4m_method_fit_mex("uve_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
