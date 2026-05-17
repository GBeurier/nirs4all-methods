classdef MethodResultRegression
% pls4all.MethodResultRegression — base classdef for tier-2 wrappers
% around MethodResult-returning fits (sparse SIMPLS, CPPLS, weighted
% PLS, MIR-PLS, MB-PLS, PLS-GLM, …). Subclasses set `Method` + any
% extra hyperparameters and inherit predict/loss/score.
%
% Prediction follows the C ABI convention
% (cpp/src/core/model.cpp::fill_prediction):
%   Y_pred = (X - XMean) @ Coefficients + YMean
% MB-PLS / GLM override `predict` to use the stored intercept directly
% (their coefficients are already in original X scale, intercept folds
% in y_mean - x_mean @ coef).

    properties (SetAccess = protected)
        Coefficients
        XMean
        YMean
        NumComponents
        NumPredictors
        NumResponses
        Rmse
        Method            % char string identifying the algorithm
        Extra = struct(); % algorithm-specific extras (lambda, gamma, …)
        UseIntercept = false
        Intercept = []
    end

    methods (Access = protected)
        function obj = absorb_result(obj, res, n_components, n_predictors)
            obj.Coefficients  = res.coefficients;
            if isfield(res, "x_mean") && ~isempty(res.x_mean)
                obj.XMean = res.x_mean;
            else
                obj.XMean = zeros(1, n_predictors);
            end
            if isfield(res, "y_mean") && ~isempty(res.y_mean)
                obj.YMean = res.y_mean;
            else
                obj.YMean = zeros(1, size(res.coefficients, 2));
            end
            if isfield(res, "rmse") && ~isempty(res.rmse)
                obj.Rmse = res.rmse;
            else
                obj.Rmse = NaN;
            end
            obj.NumComponents = double(n_components);
            obj.NumPredictors = n_predictors;
            obj.NumResponses  = size(res.coefficients, 2);
            if isfield(res, "intercept") && ~isempty(res.intercept)
                obj.Intercept    = res.intercept;
                obj.UseIntercept = true;
            end
        end
    end

    methods
        function yhat = predict(obj, X)
            if size(X, 2) ~= obj.NumPredictors
                error("pls4all:shape", ...
                      "X has %d features, but model was fitted with %d", ...
                      size(X, 2), obj.NumPredictors);
            end
            if obj.UseIntercept
                yhat = double(X) * obj.Coefficients + obj.Intercept;
            else
                yhat = (double(X) - obj.XMean) * obj.Coefficients + obj.YMean;
            end
            if obj.NumResponses == 1
                yhat = yhat(:, 1);
            end
        end

        function l = loss(obj, X, y)
            yhat = predict(obj, X);
            if isvector(y), y = y(:); end
            err = yhat - y;
            l = mean(err(:) .^ 2);
        end

        function s = score(obj, X, y)
            yhat = predict(obj, X);
            if isvector(y), y = y(:); end
            ss_res = sum((yhat - y) .^ 2, 1);
            ss_tot = sum((y - mean(y, 1)) .^ 2, 1);
            s = 1 - ss_res ./ ss_tot;
            if numel(s) == 1, s = s(1); end
        end
    end
end
