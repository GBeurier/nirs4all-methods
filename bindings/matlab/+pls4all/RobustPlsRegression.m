classdef RobustPlsRegression < pls4all.MethodResultRegression
% pls4all.RobustPlsRegression  Robust PLS via Huber IRLS.
    properties (SetAccess = private)
        HuberK = 1.345
        MaxIrlsIter = 20
    end
    methods
        function obj = RobustPlsRegression(X, y, n_components, huber_k, max_irls_iter)
            if nargin < 4 || isempty(huber_k),        huber_k = 1.345;       end
            if nargin < 5 || isempty(max_irls_iter),  max_irls_iter = 20;    end
            if isvector(y), y = y(:); end
            res = pls4all.robust_pls(X, y, n_components, huber_k, max_irls_iter);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.HuberK = double(huber_k);
            obj.MaxIrlsIter = int32(max_irls_iter);
            obj.Method = "robust_pls";
            obj.Extra = struct("huber_k", obj.HuberK, ...
                                "max_irls_iter", obj.MaxIrlsIter);
        end
    end
end
