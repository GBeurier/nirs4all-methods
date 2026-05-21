function res = bipls_select(X, Y, n_components, interval_width, min_intervals)
if nargin < 4 || isempty(interval_width), interval_width = 10; end
if nargin < 5 || isempty(min_intervals),  min_intervals = 1;   end
params = struct("interval_width", int32(interval_width), ...
                "min_intervals",  int32(min_intervals));
res = pls4all.n4m_method_fit_mex("bipls_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
