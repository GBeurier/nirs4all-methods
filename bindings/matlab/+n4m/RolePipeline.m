classdef RolePipeline < handle
    % n4m.RolePipeline  Native fitted recipe, portable as N4ME state bytes.
    %
    % model = n4m.RolePipeline(steps, featureNames, options)
    % steps is a cell or struct vector. Each scalar step has method_id and
    % optional params (one scalar struct). Native catalog defaults, parameter
    % types, role order and recipe validation are authoritative.
    % featureNames is an optional ordered cell vector of character strings.
    % options accepts num_threads and max_state_bytes (native import limit).
    %
    % fit(X,Y,aux) takes full real double matrices. Optional aux fields are
    % labels, sample_weight, groups, feature_groups, block_sizes, axis,
    % X_target and fold_ids. Integer inputs must fit int64 exactly. For a
    % classifier, pass Y=[] and integer class IDs in aux.labels.
    %
    % predict/transform/decisionFunction/predictProba/predictLabels take X
    % and optional ordered featureNames. Without names, arrays are positional;
    % native width checks always apply. predict returns int64 class IDs for
    % classifiers and an n-by-target double matrix for regressors.
    % exportStates returns one uint8 N4ME vector per stateful step. Training
    % rows are excluded by default; opt in explicitly with exportStates(true).
    % importStates validates bytes against this object's native recipe.
    properties (Access = private)
        NativeHandle = uint64(0)
    end

    methods
        function obj = RolePipeline(steps, featureNames, options)
            if nargin < 1, steps = {}; end
            if nargin < 2, featureNames = {}; end
            if nargin < 3, options = struct(); end
            obj.NativeHandle = n4m.n4m_role_pipeline_mex( ...
                'create', steps, featureNames, options);
        end

        function obj = fit(obj, X, Y, aux)
            if nargin < 3, Y = []; end
            if nargin < 4, aux = struct(); end
            % A failed native refit retains the previous fitted state.
            n4m.n4m_role_pipeline_mex('fit', obj.handle(), X, Y, aux);
        end

        function obj = importStates(obj, states)
            % Native recipe/method/parameter/width/state-count checks apply.
            % A failed import retains the previous fitted state.
            n4m.n4m_role_pipeline_mex('import_states', obj.handle(), states);
        end

        function states = exportStates(obj, allowTrainingRows)
            if nargin < 2, allowTrainingRows = false; end
            states = n4m.n4m_role_pipeline_mex( ...
                'export_states', obj.handle(), allowTrainingRows);
        end

        function obj = setFeatureNames(obj, featureNames)
            % Native code refuses changing column identities once fitted.
            n4m.n4m_role_pipeline_mex('set_feature_names', obj.handle(), featureNames);
        end

        function value = featureNames(obj)
            value = n4m.n4m_role_pipeline_mex('feature_names', obj.handle());
        end

        function value = stepsInfo(obj)
            % state_index is zero based; sample filters have index -1.
            value = n4m.n4m_role_pipeline_mex('steps_info', obj.handle());
        end

        function value = isFitted(obj)
            value = n4m.n4m_role_pipeline_mex('is_fitted', obj.handle());
        end

        function value = nFeaturesIn(obj)
            value = n4m.n4m_role_pipeline_mex('n_features_in', obj.handle());
        end

        function value = nOutputs(obj)
            value = n4m.n4m_role_pipeline_mex('n_outputs', obj.handle());
        end

        function value = transformCols(obj)
            value = n4m.n4m_role_pipeline_mex('transform_cols', obj.handle());
        end

        function value = predict(obj, X, featureNames)
            if nargin < 3, featureNames = {}; end
            value = n4m.n4m_role_pipeline_mex('predict', obj.handle(), X, featureNames);
        end

        function value = transform(obj, X, featureNames)
            if nargin < 3, featureNames = {}; end
            value = n4m.n4m_role_pipeline_mex('transform', obj.handle(), X, featureNames);
        end

        function value = decisionFunction(obj, X, featureNames)
            if nargin < 3, featureNames = {}; end
            value = n4m.n4m_role_pipeline_mex( ...
                'decision_function', obj.handle(), X, featureNames);
        end

        function value = predictProba(obj, X, featureNames)
            if nargin < 3, featureNames = {}; end
            value = n4m.n4m_role_pipeline_mex( ...
                'predict_proba', obj.handle(), X, featureNames);
        end

        function value = predictLabels(obj, X, featureNames)
            if nargin < 3, featureNames = {}; end
            value = n4m.n4m_role_pipeline_mex('predict_labels', obj.handle(), X, featureNames);
        end

        function value = classes(obj)
            value = n4m.n4m_role_pipeline_mex('classes', obj.handle());
        end

        function close(obj)
            if obj.NativeHandle ~= 0
                n4m.n4m_role_pipeline_mex('close', obj.NativeHandle);
                obj.NativeHandle = uint64(0);
            end
        end

        function delete(obj)
            obj.close();
        end

        function saveobj(~)
            error('n4m:role_pipeline', ...
                'Native handles cannot be saved; use exportStates and the current recipe');
        end
    end

    methods (Static)
        function loadobj(~)
            error('n4m:role_pipeline', ...
                'Native handles cannot be loaded; construct a recipe and call importStates');
        end
    end

    methods (Access = private)
        function value = handle(obj)
            if obj.NativeHandle == 0
                error('n4m:role_pipeline', 'RolePipeline is closed');
            end
            value = obj.NativeHandle;
        end
    end
end
