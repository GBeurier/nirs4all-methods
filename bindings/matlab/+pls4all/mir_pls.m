function res = mir_pls(X, Y, n_components)
% pls4all.mir_pls  Multivariate Inverse Regression PLS (Sjöblom 1998).
%
% Returns coefficients in the ORIGINAL X/Y space.
res = pls4all.n4m_method_fit_mex("mir_pls", double(X), double(Y), ...
                                  int32(n_components), struct());
end
