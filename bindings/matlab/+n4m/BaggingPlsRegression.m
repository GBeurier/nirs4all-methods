classdef BaggingPlsRegression < pls4all.BaggingPlsRegression
% n4m.BaggingPlsRegression  Namespace alias for pls4all.BaggingPlsRegression.
    methods
        function obj = BaggingPlsRegression(varargin)
            obj@pls4all.BaggingPlsRegression(varargin{:});
        end
    end
end
