classdef CpplsRegression < pls4all.MethodResultRegression
% pls4all.CpplsRegression — Canonical Powered PLS (Indahl 2005).
    properties (SetAccess = private)
        Gamma = 0.5
    end
    methods
        function obj = CpplsRegression(X, y, n_components, gamma)
            if nargin < 4 || isempty(gamma), gamma = 0.5; end
            if isvector(y), y = y(:); end
            res = pls4all.cppls(X, y, n_components, gamma);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.Gamma  = double(gamma);
            obj.Method = "cppls";
            obj.Extra  = struct("gamma", obj.Gamma);
        end
    end
end
