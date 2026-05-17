function res = iriv_select(X, Y, n_components, max_rounds, seed)
if nargin < 4 || isempty(max_rounds), max_rounds = 20; end
if nargin < 5 || isempty(seed),       seed = 0;        end
params = struct("max_rounds", int32(max_rounds), "seed", uint64(seed));
res = pls4all.p4a_method_fit_mex("iriv_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
