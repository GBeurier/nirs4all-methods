classdef O2plsRegression < pls4all.MethodResultRegression
% pls4all.O2plsRegression  O2-PLS (bi-directional OPLS).
    properties (SetAccess = private)
        NPredictive
        NXOrthogonal
        NYOrthogonal
    end
    methods
        function obj = O2plsRegression(X, y, n_predictive, n_x_orthogonal, n_y_orthogonal)
            if nargin < 4 || isempty(n_x_orthogonal), n_x_orthogonal = 1; end
            if nargin < 5 || isempty(n_y_orthogonal), n_y_orthogonal = 1; end
            if isvector(y), y = y(:); end
            res = pls4all.o2pls(X, y, n_predictive, n_x_orthogonal, n_y_orthogonal);
            obj = obj.absorb_result(res, n_predictive, size(X, 2));
            obj.NPredictive = int32(n_predictive);
            obj.NXOrthogonal = int32(n_x_orthogonal);
            obj.NYOrthogonal = int32(n_y_orthogonal);
            obj.Method = "o2pls";
            obj.Extra = struct("n_predictive", obj.NPredictive, ...
                                "n_x_orthogonal", obj.NXOrthogonal, ...
                                "n_y_orthogonal", obj.NYOrthogonal);
        end
    end
end
