classdef EcrRegression < pls4all.MethodResultRegression
% pls4all.EcrRegression  Elastic Component Regression (Liu 2009).
    properties (SetAccess = private)
        Alpha = 0.5
    end
    methods
        function obj = EcrRegression(X, y, n_components, alpha)
            if nargin < 4 || isempty(alpha), alpha = 0.5; end
            if isvector(y), y = y(:); end
            res = pls4all.ecr(X, y, n_components, alpha);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Alpha = double(alpha);
            obj.Method = "ecr";
            obj.Extra = struct("alpha", obj.Alpha);
        end
    end
end
