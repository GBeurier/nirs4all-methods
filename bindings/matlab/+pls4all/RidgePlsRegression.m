classdef RidgePlsRegression < pls4all.MethodResultRegression
% pls4all.RidgePlsRegression  L2-augmented PLS regression.
    properties (SetAccess = private)
        RidgeLambda = 1.0
    end
    methods
        function obj = RidgePlsRegression(X, y, n_components, ridge_lambda)
            if nargin < 4 || isempty(ridge_lambda), ridge_lambda = 1.0; end
            if isvector(y), y = y(:); end
            res = pls4all.ridge_pls(X, y, n_components, ridge_lambda);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.RidgeLambda = double(ridge_lambda);
            obj.Method = "ridge_pls";
            obj.Extra = struct("ridge_lambda", obj.RidgeLambda);
        end
    end
end
