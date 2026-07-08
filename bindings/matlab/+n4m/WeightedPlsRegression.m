classdef WeightedPlsRegression < n4m.MethodResultRegression
% n4m.WeightedPlsRegression — sqrt(w)-prescaled SIMPLS.
    properties (SetAccess = private)
        SampleWeights
    end
    methods
        function obj = WeightedPlsRegression(X, y, n_components, sample_weights)
            if nargin < 4
                error("n4m:nargin", "sample_weights required");
            end
            if isvector(y), y = y(:); end
            res = n4m.weighted_pls(X, y, n_components, sample_weights);
            obj = obj.absorb_result(res, n_components, size(X, 2));
            obj.SampleWeights = double(sample_weights(:));
            obj.Method = "weighted_pls";
        end
    end
end
