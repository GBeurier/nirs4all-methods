function [coefs, x_mean, y_mean, predictions] = pcr(X, Y, n_components)
% n4m.pcr  Fit a Principal Component Regression model on (X, Y).
%
%   [COEFS, X_MEAN, Y_MEAN, PRED] = n4m.pcr(X, Y, K)
%
% Inputs:
%   X            n x p double matrix.
%   Y            n x q double matrix.
%   n_components positive integer (latent component count).
%
% Outputs:
%   COEFS        (p x q) regression coefficients.
%   X_MEAN       (1 x p) per-feature mean (centring).
%   Y_MEAN       (1 x q) per-target mean.
%   PRED         (n x q) in-sample predictions.
%
% Uses N4M_ALGO_PCR + N4M_SOLVER_SVD + N4M_DEFLATION_REGRESSION.

if nargin < 3
    error("n4m:nargin", ...
          "Usage: n4m.pcr(X, Y, n_components)");
end

[coefs, x_mean, y_mean, predictions] = ...
    n4m.n4m_model_fit_mex("pcr", double(X), double(Y), int32(n_components));
end
