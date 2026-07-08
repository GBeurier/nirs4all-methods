classdef MissingAwareNipalsRegression < n4m.MethodResultRegression
% n4m.MissingAwareNipalsRegression  Nelson 1996 missing-aware NIPALS PLS.
    methods
        function obj = MissingAwareNipalsRegression(X, y, n_components)
            if isvector(y), y = y(:); end
            res = n4m.missing_aware_nipals(X, y, n_components);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Method = "missing_aware_nipals";
        end
    end
end
