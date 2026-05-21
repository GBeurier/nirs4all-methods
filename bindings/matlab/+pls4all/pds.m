function res = pds(X_source, X_target, window_half_width)
% pls4all.pds  Piecewise Direct Standardization (calibration transfer).
if nargin < 3 || isempty(window_half_width), window_half_width = 2; end
params = struct("X_target", double(X_target), ...
                "window_half_width", int32(window_half_width));
% Dispatcher's n_components is unused but required; pass 1.
res = pls4all.n4m_method_fit_mex("pds", double(X_source), zeros(size(X_source,1), 1), ...
                                  int32(1), params);
end
