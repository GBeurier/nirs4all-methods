classdef MultimodalPipeline < handle
    % Complete native raw multimodal early-fusion predictor (ABI 2.16).
    % The native state contains training-only encoders, UTF-8 vocabulary and
    % Ridge. Raw tensors retain sample-first dimensions; metadata is an n-by-2
    % cell matrix whose declared numeric column alone is converted to numbers.
    properties (SetAccess = private)
        recipe
        sourceSchemas
    end
    properties (Access = private)
        nativeHandle = uint64(0)
    end
    methods
        function obj = MultimodalPipeline(recipe, sourceSchemas)
            if nargin == 0
                return;
            end
            obj.recipe = recipe;
            obj.sourceSchemas = sourceSchemas;
            obj.nativeHandle = n4m.n4m_multimodal_pipeline_mex('create', recipe, sourceSchemas);
        end
        function obj = fit(obj, blocks, y)
            obj.requireOpen();
            if ~isnumeric(y) || ~isreal(y) || ~isvector(y)
                error('n4m:multimodal', 'y must be one real numeric target per row');
            end
            n4m.n4m_multimodal_pipeline_mex('fit', obj.nativeHandle, blocks, obj.sourceSchemas, double(y));
        end
        function result = predict(obj, blocks, sourceSchemas)
            obj.requireOpen();
            if nargin < 3
                sourceSchemas = obj.sourceSchemas;
            end
            result = n4m.n4m_multimodal_pipeline_mex('predict', obj.nativeHandle, blocks, sourceSchemas);
        end
        function result = transform(obj, blocks, sourceSchemas)
            obj.requireOpen();
            if nargin < 3
                sourceSchemas = obj.sourceSchemas;
            end
            result = n4m.n4m_multimodal_pipeline_mex('transform', obj.nativeHandle, blocks, sourceSchemas);
        end
        function result = exportState(obj)
            obj.requireOpen();
            result = n4m.n4m_multimodal_pipeline_mex('export_state', obj.nativeHandle);
        end
        function close(obj)
            if obj.nativeHandle ~= 0
                n4m.n4m_multimodal_pipeline_mex('close', obj.nativeHandle);
                obj.nativeHandle = uint64(0);
            end
        end
        function delete(obj)
            obj.close();
        end
    end
    methods (Static)
        function obj = fromState(state, recipe, sourceSchemas)
            obj = n4m.MultimodalPipeline();
            obj.recipe = recipe;
            obj.sourceSchemas = sourceSchemas;
            obj.nativeHandle = n4m.n4m_multimodal_pipeline_mex('from_state', recipe, sourceSchemas, state);
        end
    end
    methods (Access = private)
        function requireOpen(obj)
            if obj.nativeHandle == 0
                error('n4m:multimodal', 'MultimodalPipeline is closed');
            end
        end
    end
end
