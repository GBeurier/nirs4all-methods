function [coefs, x_mean, y_mean, predictions] = pls_fit(X, Y, n_components)
% pls4all.pls_fit  Fit a SIMPLS PLS regression on (X, Y).
%
%   [COEFS, X_MEAN, Y_MEAN, PRED] = pls4all.pls_fit(X, Y, K)
%
% Inputs:
%   X            n × p double matrix.
%   Y            n × q double matrix.
%   n_components positive integer (latent component count).
%
% Outputs:
%   COEFS        (p × q) regression coefficients.
%   X_MEAN       (1 × p) per-feature mean (centring).
%   Y_MEAN       (1 × q) per-target mean.
%   PRED         (n × q) in-sample predictions.
%
% The fit uses SIMPLS with center_x = center_y = scale_x = scale_y = 1
% to align with the public C ABI Config defaults.

if nargin < 3
    error("pls4all:nargin", ...
          "Usage: pls4all.pls_fit(X, Y, n_components)");
end

[coefs, x_mean, y_mean, predictions] = ...
    pls4all.p4a_pls_fit_mex(double(X), double(Y), int32(n_components));
end
