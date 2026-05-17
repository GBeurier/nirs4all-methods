function res = interval_select(X, Y, n_components, interval_width, step)
if nargin < 4 || isempty(interval_width), interval_width = 10; end
if nargin < 5 || isempty(step),           step = 1;            end
params = struct("interval_width", int32(interval_width), "step", int32(step));
res = pls4all.p4a_method_fit_mex("interval_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
