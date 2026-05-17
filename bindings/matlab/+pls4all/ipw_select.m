function res = ipw_select(X, Y, n_components, n_iterations, top_k, damping, weight_floor)
if nargin < 4 || isempty(n_iterations), n_iterations = 10; end
if nargin < 5 || isempty(top_k),        top_k = 10;        end
if nargin < 6 || isempty(damping),      damping = 0.5;     end
if nargin < 7 || isempty(weight_floor), weight_floor = 1e-6; end
params = struct("n_iterations", int32(n_iterations), "top_k", int32(top_k), ...
                "damping", double(damping), "weight_floor", double(weight_floor));
res = pls4all.p4a_method_fit_mex("ipw_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
