function res = stability_select(X, Y, n_components, top_k)
% pls4all.stability_select  Coefficient-stability selector (MCUVE-style).
params = struct("top_k", int32(top_k));
res = pls4all.p4a_method_fit_mex("stability_select", double(X), double(Y), ...
                                  int32(n_components), params);
end
