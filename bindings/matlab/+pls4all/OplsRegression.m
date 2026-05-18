classdef OplsRegression
% pls4all.OplsRegression — Orthogonal Partial Least Squares Regression model.
%
% Example:
%
%     mdl = pls4all.OplsRegression(X, y, 5);
%     yhat = predict(mdl, Xnew);
%     mse  = loss(mdl, Xtest, ytest);
%
% Properties (read-only):
%   Coefficients   p x q regression coefficient matrix
%   XMean          1 x p per-feature mean (centring)
%   YMean          1 x q per-target mean
%   NumComponents  number of latent components used
%   NumPredictors  p
%   NumResponses   q
%
% Methods:
%   predict(mdl, Xnew)      — returns predictions on new X
%   loss(mdl, X, y)         — mean squared error on (X, y)
%   score(mdl, X, y)        — coefficient of determination R^2

    properties (SetAccess = private)
        Coefficients     % (p x q) double matrix
        XMean            % (1 x p) double row vector
        YMean            % (1 x q) double row vector
        NumComponents    % positive integer
        NumPredictors    % positive integer
        NumResponses     % positive integer
        Method           % algorithm name
    end

    methods
        function obj = OplsRegression(X, y, n_components)
            % Constructor
            if nargin < 3
                error("pls4all:nargin", ...
                      "Usage: pls4all.OplsRegression(X, y, n_components)");
            end
            if isvector(y)
                y = y(:);
            end
            if size(X, 1) ~= size(y, 1)
                error("pls4all:shape", ...
                      "X and y must have the same number of rows");
            end
            [coefs, x_mean, y_mean, ~] = pls4all.opls( ...
                X, y, double(n_components));
            obj.Coefficients  = coefs;
            obj.XMean         = x_mean;
            obj.YMean         = y_mean;
            obj.NumComponents = double(n_components);
            obj.NumPredictors = size(X, 2);
            obj.NumResponses  = size(y, 2);
            obj.Method        = "opls";
        end

        function yhat = predict(obj, X)
            % predict(MDL, X) — predictions on new X.
            if size(X, 2) ~= obj.NumPredictors
                error("pls4all:shape", ...
                      "X has %d features, but model was fitted with %d", ...
                      size(X, 2), obj.NumPredictors);
            end
            yhat = (double(X) - obj.XMean) * obj.Coefficients + obj.YMean;
            if obj.NumResponses == 1
                yhat = yhat(:, 1);
            end
        end

        function l = loss(obj, X, y)
            % loss(MDL, X, y) — mean squared error on (X, y).
            yhat = predict(obj, X);
            if isvector(y)
                y = y(:);
            end
            err = yhat - y;
            l = mean(err(:) .^ 2);
        end

        function s = score(obj, X, y)
            % score(MDL, X, y) — coefficient of determination R^2.
            yhat = predict(obj, X);
            if isvector(y)
                y = y(:);
            end
            ss_res = sum((yhat - y) .^ 2, 1);
            ss_tot = sum((y - mean(y, 1)) .^ 2, 1);
            s = 1 - ss_res ./ ss_tot;
            if numel(s) == 1
                s = s(1);
            end
        end
    end
end
