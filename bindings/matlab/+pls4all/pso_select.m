function res = pso_select(X, Y, n_components, n_swarm, n_iterations, ...
                            w, c1, c2, v_max, seed)
if nargin < 4 || isempty(n_swarm),      n_swarm = 30;      end
if nargin < 5 || isempty(n_iterations), n_iterations = 50; end
if nargin < 6 || isempty(w),    w = 0.729;     end
if nargin < 7 || isempty(c1),   c1 = 1.494;    end
if nargin < 8 || isempty(c2),   c2 = 1.494;    end
if nargin < 9 || isempty(v_max), v_max = 4.0;  end
if nargin < 10 || isempty(seed), seed = 0;     end
params = struct("n_swarm", int32(n_swarm), "n_iterations", int32(n_iterations), ...
                "w", double(w), "c1", double(c1), "c2", double(c2), ...
                "v_max", double(v_max), "seed", uint64(seed));
res = pls4all.n4m_method_fit_mex("pso_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
