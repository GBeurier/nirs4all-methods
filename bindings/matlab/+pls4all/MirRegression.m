classdef MirRegression < pls4all.MethodResultRegression
% pls4all.MirRegression — Multivariate Inverse Regression PLS.
    methods
        function obj = MirRegression(X, y, n_components)
            if isvector(y), y = y(:); end
            res = pls4all.mir_pls(X, y, n_components);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Method = "mir_pls";
        end
    end
end
