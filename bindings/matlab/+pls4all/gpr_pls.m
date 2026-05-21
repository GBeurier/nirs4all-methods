function res = gpr_pls(X, Y, n_components, length_scale, noise_level, seed)
% pls4all.gpr_pls  GPR on PLS scores (RBF kernel, single-target Y).
if nargin < 4 || isempty(length_scale), length_scale = 1.0;  end
if nargin < 5 || isempty(noise_level),  noise_level  = 1e-4; end
if nargin < 6 || isempty(seed),         seed = 0;            end
params = struct("length_scale", double(length_scale), ...
                "noise_level",  double(noise_level), ...
                "seed",         uint64(seed));
res = pls4all.n4m_method_fit_mex("gpr_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
