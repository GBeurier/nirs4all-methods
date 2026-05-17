classdef BaggingPlsRegression < pls4all.MethodResultRegression
    properties (SetAccess = private)
        NEstimators
        Seed
    end
    methods
        function obj = BaggingPlsRegression(X, y, n_components, n_estimators, seed)
            if nargin < 4 || isempty(n_estimators), n_estimators = 50; end
            if nargin < 5 || isempty(seed),         seed = 0;          end
            if isvector(y), y = y(:); end
            res = pls4all.bagging_pls(X, y, n_components, n_estimators, seed);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.NEstimators = int32(n_estimators);
            obj.Seed = uint64(seed);
            obj.Method = "bagging_pls";
            obj.Extra = struct("n_estimators", obj.NEstimators);
        end
    end
end
