function [coefs, x_mean, y_mean, predictions] = opls(X, Y, n_components)
% pls4all.opls  Fit an Orthogonal Partial Least Squares model on (X, Y).
%
%   [COEFS, X_MEAN, Y_MEAN, PRED] = pls4all.opls(X, Y, K)
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
% Uses N4M_ALGO_OPLS + N4M_SOLVER_NIPALS + N4M_DEFLATION_ORTHOGONAL.

if nargin < 3
    error("pls4all:nargin", ...
          "Usage: pls4all.opls(X, Y, n_components)");
end

[coefs, x_mean, y_mean, predictions] = ...
    pls4all.n4m_model_fit_mex("opls", double(X), double(Y), int32(n_components));
end
