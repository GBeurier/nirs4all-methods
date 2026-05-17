function res = st_select(X, Y, n_components, thresholds, min_selected)
% pls4all.st_select  Score-threshold selector (sweep over thresholds).
if nargin < 5 || isempty(min_selected), min_selected = 1; end
params = struct("thresholds",   double(thresholds(:)), ...
                "min_selected", int32(min_selected));
res = pls4all.p4a_method_fit_mex("st_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
