function res = pop_pls(X, Y, max_components, n_operators, cv)
% pls4all.pop_pls  POP-PLS per-component operator selection.
if nargin < 3 || isempty(max_components), max_components = 3; end
if nargin < 4 || isempty(n_operators), n_operators = 9; end
if nargin < 5 || isempty(cv), cv = 3; end
params = struct("max_components", int32(max_components), ...
                "n_operators", int32(n_operators), ...
                "cv", int32(cv));
res = pls4all.n4m_method_fit_mex("pop_pls", double(X), double(Y), ...
                                  int32(max_components), params);
end
