classdef GlmRegression < n4m.MethodResultRegression
% n4m.GlmRegression — PLS-GLM (Gaussian / Poisson IRLS).
% Like MB-PLS, uses the stored intercept directly.
    properties (SetAccess = private)
        Family
    end
    methods
        function obj = GlmRegression(X, y, n_components, family)
            if nargin < 4 || isempty(family), family = "gaussian"; end
            if isstring(family), family = char(family); end
            if isvector(y), y = y(:); end
            res = n4m.pls_glm(X, y, n_components, family);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Family = family;
            obj.Method = sprintf("pls_glm_%s", family);
        end
    end
end
