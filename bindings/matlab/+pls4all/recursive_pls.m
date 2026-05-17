function res = recursive_pls(X, Y, n_components, window_size)
% pls4all.recursive_pls  Moving-window recursive PLS.
if nargin < 4 || isempty(window_size), window_size = 50; end
params = struct("window_size", int32(window_size));
res = pls4all.p4a_method_fit_mex("recursive_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
