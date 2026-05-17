classdef KernelPlsRegression < pls4all.MethodResultRegression
% pls4all.KernelPlsRegression  Non-linear kernel PLS (Rosipal & Trejo 2001).
%
% **In-sample only**: the C ABI exports `alpha` and `y_mean` but NOT the
% kernel-centering state needed to compute K(X_new, X_train) at predict
% time. predict(X) returns the stored predictions when X matches the
% training matrix (content equality), otherwise raises an error.
    properties (SetAccess = private)
        KernelType
        Gamma
        Coef0
        Degree
        Predictions
        Alpha
        XTrain
    end
    methods
        function obj = KernelPlsRegression(X, y, n_components, kernel_type, gamma, coef0, degree)
            if nargin < 4 || isempty(kernel_type), kernel_type = 1; end
            if nargin < 5 || isempty(gamma),       gamma = 0.0;     end
            if nargin < 6 || isempty(coef0),       coef0 = 1.0;     end
            if nargin < 7 || isempty(degree),      degree = 3;      end
            if isvector(y), y = y(:); end
            res = pls4all.kernel_pls(X, y, n_components, kernel_type, gamma, coef0, degree);
            obj.NumComponents = double(n_components);
            obj.NumPredictors = size(X, 2);
            obj.NumResponses  = size(y, 2);
            obj.Coefficients  = [];
            obj.XMean = zeros(1, size(X, 2));
            obj.YMean = res.y_mean;
            obj.KernelType = int32(kernel_type);
            obj.Gamma = double(gamma); obj.Coef0 = double(coef0);
            obj.Degree = int32(degree);
            obj.Predictions = res.predictions;
            obj.Alpha = res.alpha;
            obj.XTrain = double(X);
            obj.Rmse = res.rmse;
            obj.Method = "kernel_pls";
        end
        function yhat = predict(obj, X)
            if size(X, 1) ~= size(obj.XTrain, 1) || ~isequal(X, obj.XTrain)
                error("pls4all:nopredict", ...
                      ["KernelPlsRegression is in-sample only: the C ABI " ...
                       "does not export the kernel-centering state needed " ...
                       "to compute K_test consistently with the fit."]);
            end
            yhat = obj.Predictions;
            if obj.NumResponses == 1, yhat = yhat(:, 1); end
        end
    end
end
