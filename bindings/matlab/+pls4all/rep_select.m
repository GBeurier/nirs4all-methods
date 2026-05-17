function res = rep_select(X, Y, n_components, n_steps, min_features, remove_count)
if nargin < 4 || isempty(n_steps),      n_steps = 10;       end
if nargin < 5 || isempty(min_features), min_features = 5;   end
if nargin < 6 || isempty(remove_count), remove_count = 1;   end
params = struct("n_steps",       int32(n_steps), ...
                "min_features",  int32(min_features), ...
                "remove_count",  int32(remove_count));
res = pls4all.p4a_method_fit_mex("rep_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
