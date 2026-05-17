function res = kernel_pls(X, Y, n_components, kernel_type, gamma, coef0, degree)
% pls4all.kernel_pls  Non-linear kernel PLS (Rosipal & Trejo 2001).
%
% kernel_type: 0=linear, 1=RBF (default), 2=polynomial, 3=sigmoid.
if nargin < 4 || isempty(kernel_type), kernel_type = 1; end
if nargin < 5 || isempty(gamma),       gamma = 0.0;     end
if nargin < 6 || isempty(coef0),       coef0 = 1.0;     end
if nargin < 7 || isempty(degree),      degree = 3;      end
params = struct("kernel_type", int32(kernel_type), ...
                "gamma", double(gamma), ...
                "coef0", double(coef0), ...
                "degree", int32(degree));
res = pls4all.p4a_method_fit_mex("kernel_pls", double(X), double(Y), ...
                                  int32(n_components), params);
end
