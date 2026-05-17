classdef BoostingPlsRegression < pls4all.MethodResultRegression
    properties (SetAccess = private)
        NEstimators
        LearningRate
    end
    methods
        function obj = BoostingPlsRegression(X, y, n_components, n_estimators, learning_rate)
            if nargin < 4 || isempty(n_estimators),  n_estimators = 50;  end
            if nargin < 5 || isempty(learning_rate), learning_rate = 0.1; end
            if isvector(y), y = y(:); end
            res = pls4all.boosting_pls(X, y, n_components, n_estimators, learning_rate);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.NEstimators = int32(n_estimators);
            obj.LearningRate = double(learning_rate);
            obj.Method = "boosting_pls";
            obj.Extra = struct("n_estimators", obj.NEstimators, ...
                                "learning_rate", obj.LearningRate);
        end
    end
end
