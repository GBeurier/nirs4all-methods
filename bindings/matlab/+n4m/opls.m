function [coefs, x_mean, y_mean, predictions] = opls(X, Y, n_components)
% n4m.opls  Fit an Orthogonal Partial Least Squares model on (X, Y).
%
%   [COEFS, X_MEAN, Y_MEAN, PRED] = n4m.opls(X, Y, K)
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
    error("n4m:nargin", ...
          "Usage: n4m.opls(X, Y, n_components)");
end

[coefs, x_mean, y_mean, predictions] = ...
    n4m.n4m_model_fit_mex("opls", double(X), double(Y), int32(n_components));
end
