classdef DiPlsRegression < pls4all.MethodResultRegression
% pls4all.DiPlsRegression  Domain-Invariant PLS regression.
    properties (SetAccess = private)
        DiLambda = 1.0
    end
    methods
        function obj = DiPlsRegression(X, y, n_components, X_target, di_lambda)
            if nargin < 5 || isempty(di_lambda), di_lambda = 1.0; end
            if nargin < 4 || isempty(X_target)
                error("pls4all:nargin", "di_pls requires XTarget");
            end
            if isvector(y), y = y(:); end
            res = pls4all.di_pls(X, y, n_components, X_target, di_lambda);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.DiLambda = double(di_lambda);
            obj.Method = "di_pls";
            obj.Extra = struct("di_lambda", obj.DiLambda);
        end
    end
end
