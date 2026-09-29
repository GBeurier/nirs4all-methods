classdef Optimizer < handle
    % n4m.Optimizer  Stateful native ask/tell optimizer for MATLAB and Octave.
    %
    % Space is an ordered struct array. Every axis has name and kind. Numeric
    % axes also have low/high and optional step; categorical/ordinal axes have
    % choices; sorted_tuple axes have length, low/high and optional integer.
    % Empty fields on other axis kinds are ignored. Choices for a categorical
    % axis may be a cell array of strings, double, int64 or logical values.
    %
    % Options is a scalar struct using C ABI names: sampler, pruner,
    % direction, metric, eval_mode, liar, n_startup_trials, seed,
    % timeout_seconds, max_resource and reduction_factor. Constraints is a
    % struct array with kind, refs (cellstr) and optional labels (cellstr).
    % The native engine owns proposal, pruning, trace and checkpoint logic.
    properties (Access = private)
        NativeHandle = uint64(0)
        Space
    end

    methods
        function obj = Optimizer(space, options, constraints)
            if nargin == 0
                return;
            end
            if nargin < 2, options = struct(); end
            if nargin < 3, constraints = []; end
            obj.NativeHandle = n4m.n4m_optimizer_mex('create', space, options, constraints);
            obj.Space = space;
        end

        function close(obj)
            if obj.NativeHandle ~= 0
                n4m.n4m_optimizer_mex('close', obj.NativeHandle);
                obj.NativeHandle = uint64(0);
            end
        end

        function delete(obj)
            obj.close();
        end

        function trial = ask(obj)
            trial = n4m.n4m_optimizer_mex('ask', obj.handle());
        end

        function [trials, native_status] = askBatch(obj, n)
            [trials, native_status] = n4m.n4m_optimizer_mex('ask_batch', obj.handle(), n);
        end

        function enqueue(obj, parameters, values)
            % Numeric warm starts use zero-based indices for categoricals.
            % Pass cell-string names and numeric values for DAG paths with dots.
            if nargin < 3
                if ~isstruct(parameters) || numel(parameters) ~= 1
                    error('n4m:optimizer', 'enqueue parameters must be one struct');
                end
                names = fieldnames(parameters);
                values = zeros(1, numel(names));
                for k = 1:numel(names)
                    values(k) = double(parameters.(names{k}));
                end
            else
                names = parameters;
            end
            n4m.n4m_optimizer_mex('enqueue', obj.handle(), names, values);
        end

        function tell(obj, trial, score, status, error_text)
            if nargin < 4 || isempty(status), status = 'completed'; end
            if nargin < 5, error_text = ''; end
            if isstruct(trial), trial = trial.id; end
            n4m.n4m_optimizer_mex('tell', obj.handle(), trial, status, score, error_text);
        end

        function should_prune = intermediate(obj, trial, step, score)
            if isstruct(trial), trial = trial.id; end
            should_prune = n4m.n4m_optimizer_mex('intermediate', obj.handle(), ...
                                                 trial, step, score);
        end

        function result = best(obj)
            result = n4m.n4m_optimizer_mex('best', obj.handle());
        end

        function result = trials(obj, since_id)
            if nargin < 2, since_id = int64(0); end
            result = n4m.n4m_optimizer_mex('trials', obj.handle(), since_id);
        end

        function bytes = save(obj)
            bytes = n4m.n4m_optimizer_mex('save', obj.handle());
        end
    end

    methods (Static)
        function value = getParameter(trial, name)
            % Exact lookup also works for native DAG parameter names with dots.
            if ~isstruct(trial) || ~iscell(trial.parameter_names)
                error('n4m:optimizer', 'expected a native optimizer trial');
            end
            index = find(strcmp(trial.parameter_names, name), 1);
            if isempty(index)
                error('n4m:optimizer', 'unknown trial parameter: %s', name);
            end
            value = trial.parameter_values{index};
        end

        function obj = load(bytes, space)
            % Restore a N4MOPT checkpoint; space must match the saved axis
            % declarations so typed trial values can be decoded faithfully.
            obj = n4m.Optimizer();
            obj.NativeHandle = n4m.n4m_optimizer_mex('load', bytes, space);
            obj.Space = space;
        end
    end

    methods (Access = private)
        function value = handle(obj)
            if obj.NativeHandle == 0
                error('n4m:optimizer', 'optimizer is closed');
            end
            value = obj.NativeHandle;
        end
    end
end
