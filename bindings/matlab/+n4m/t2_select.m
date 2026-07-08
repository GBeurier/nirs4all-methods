function res = t2_select(X, Y, n_components, alpha_thresholds, min_selected)
% n4m.t2_select  T²-based selector (sweep over alpha thresholds).
if nargin < 5 || isempty(min_selected), min_selected = 1; end
params = struct("alpha_thresholds", double(alpha_thresholds(:)), ...
                "min_selected",     int32(min_selected));
res = n4m.n4m_method_fit_mex("t2_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
