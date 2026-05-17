classdef RecursivePlsRegression < pls4all.MethodResultRegression
% pls4all.RecursivePlsRegression  Moving-window recursive PLS.
% In-sample only: result holds in-window predictions; no global coefficient
% matrix. `predict(X)` returns the stored predictions for the training X
% (length-preserved; warmup samples are 0).
    properties (SetAccess = private)
        WindowSize
        Predictions
        InWindow
    end
    methods
        function obj = RecursivePlsRegression(X, y, n_components, window_size)
            if nargin < 4 || isempty(window_size), window_size = 50; end
            if isvector(y), y = y(:); end
            res = pls4all.recursive_pls(X, y, n_components, window_size);
            obj.NumComponents = double(n_components);
            obj.NumPredictors = size(X, 2);
            obj.NumResponses  = size(y, 2);
            obj.Coefficients  = [];
            obj.XMean = zeros(1, size(X, 2));
            obj.YMean = zeros(1, size(y, 2));
            obj.WindowSize = int32(window_size);
            obj.Predictions = res.predictions;
            obj.InWindow = res.in_window;
            obj.Rmse = res.rmse_predictable;
            obj.Method = "recursive_pls";
            obj.Extra = struct("window_size", obj.WindowSize);
        end
        function yhat = predict(obj, X)
            % Predict-on-new-X is not defined for recursive PLS (no global
            % coefficient export). Returns the in-sample predictions when
            % size(X, 1) matches the training set.
            if size(X, 1) ~= size(obj.Predictions, 1)
                error("pls4all:nopredict", ...
                      "RecursivePlsRegression is in-sample only");
            end
            yhat = obj.Predictions;
            if obj.NumResponses == 1, yhat = yhat(:, 1); end
        end
    end
end
