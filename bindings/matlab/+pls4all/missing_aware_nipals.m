function res = missing_aware_nipals(X, Y, n_components)
% pls4all.missing_aware_nipals  Nelson 1996 missing-aware NIPALS PLS.
res = pls4all.n4m_method_fit_mex("missing_aware_nipals", double(X), double(Y), ...
                                  int32(n_components), struct());
end
