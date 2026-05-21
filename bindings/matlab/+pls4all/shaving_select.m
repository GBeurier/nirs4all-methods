function res = shaving_select(X, Y, n_components, n_steps, min_features, shave_fraction)
if nargin < 4 || isempty(n_steps),        n_steps = 10;          end
if nargin < 5 || isempty(min_features),   min_features = 5;      end
if nargin < 6 || isempty(shave_fraction), shave_fraction = 0.1;  end
params = struct("n_steps", int32(n_steps), "min_features", int32(min_features), ...
                "shave_fraction", double(shave_fraction));
res = pls4all.n4m_method_fit_mex("shaving_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
