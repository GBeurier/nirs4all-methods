function [coefs, x_mean, y_mean, predictions] = pcr(X, Y, n_components)
% pls4all.pcr  Fit a Principal Component Regression model on (X, Y).
%
%   [COEFS, X_MEAN, Y_MEAN, PRED] = pls4all.pcr(X, Y, K)
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
% Uses P4A_ALGO_PCR + P4A_SOLVER_SVD + P4A_DEFLATION_REGRESSION.

if nargin < 3
    error("pls4all:nargin", ...
          "Usage: pls4all.pcr(X, Y, n_components)");
end

[coefs, x_mean, y_mean, predictions] = ...
    pls4all.p4a_model_fit_mex("pcr", double(X), double(Y), int32(n_components));
end
