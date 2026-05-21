function res = aom_preprocess(X, Y, n_operators, gating_mode)
% pls4all.aom_preprocess  AOM preprocessing fit/transform.
if nargin < 2 || isempty(Y), Y = zeros(size(X, 1), 1); end
if nargin < 3 || isempty(n_operators), n_operators = 3; end
if nargin < 4 || isempty(gating_mode), gating_mode = 0; end
params = struct("n_operators", int32(n_operators), ...
                "gating_mode", int32(gating_mode));
res = pls4all.n4m_method_fit_mex("aom_preprocess", double(X), double(Y), ...
                                  int32(1), params);
end
