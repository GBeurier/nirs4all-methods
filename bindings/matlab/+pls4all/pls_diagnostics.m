function res = pls_diagnostics(X, Y, n_components, X_reference)
% pls4all.pls_diagnostics  Hotelling T2, Q residuals, DModX from a SIMPLS fit.
%
%   res = pls4all.pls_diagnostics(X, Y, n_components)
%   res = pls4all.pls_diagnostics(X, Y, n_components, X_reference)
%
% Fits an internal SIMPLS model (store_scores=1) and evaluates row-wise
% Hotelling T^2, Q-residuals and Distance-to-Model X (DModX). When
% X_reference is omitted, the reference score distribution falls back to
% the model's stored training scores.
%
% Output struct fields:
%   t2     : 1 x n_samples Hotelling T^2 statistic per row of X.
%   q      : 1 x n_samples Q residual (squared reconstruction error).
%   dmodx  : 1 x n_samples Distance-to-Model X.
%   n_components, n_features.
params = struct();
if nargin >= 4 && ~isempty(X_reference)
    params.X_reference = double(X_reference);
end
res = pls4all.p4a_method_fit_mex("pls_diagnostics_compute", ...
                                  double(X), double(Y), ...
                                  int32(n_components), params);
end
