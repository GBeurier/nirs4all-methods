function res = sipls_select(X, Y, n_components, interval_width, combination_size)
if nargin < 4 || isempty(interval_width),   interval_width = 10;   end
if nargin < 5 || isempty(combination_size), combination_size = 2;  end
params = struct("interval_width",   int32(interval_width), ...
                "combination_size", int32(combination_size));
res = pls4all.p4a_method_fit_mex("sipls_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
