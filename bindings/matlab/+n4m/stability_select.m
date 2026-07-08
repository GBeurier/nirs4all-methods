function res = stability_select(X, Y, n_components, top_k)
% n4m.stability_select  Coefficient-stability selector (MCUVE-style).
params = struct("top_k", int32(top_k));
res = n4m.n4m_method_fit_mex("stability_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
