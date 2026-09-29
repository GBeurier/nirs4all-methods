function test_optimizer_goldens(fixture_dir)
% Compare selected or exhaustive native optimizer traces from Octave with Python.
if nargin ~= 1 || exist(fullfile(fixture_dir, 'n4m_hpo_fixtures.m'), 'file') ~= 2
    error('n4m:optimizer', 'expected the generated HPO fixture directory');
end
addpath(fixture_dir);
[specs, goldens] = n4m_hpo_fixtures();
specs = elements(specs);
assert(numel(specs) == 14 || numel(specs) == 45);
for s = 1:numel(specs)
    spec = specs{s};
    golden = elements(goldens{s});
    declared = elements(spec.space);
    space = search_space(declared);
    options = struct('sampler', spec.sampler, 'pruner', spec.pruner, ...
                     'direction', spec.direction, 'seed', spec.seed, ...
                     'n_startup_trials', spec.n_startup_trials, ...
                     'max_resource', spec.max_resource, ...
                     'reduction_factor', spec.reduction_factor);
    optimizer = n4m.Optimizer(space, options);
    try
        for start = 1:spec.max_in_flight:spec.n_trials
            size_now = min(spec.max_in_flight, spec.n_trials - start + 1);
            batch = cell(1, size_now);
            for i = 1:size_now
                batch{i} = optimizer.ask();
                expected = golden{double(batch{i}.id) + 1};
                compare_parameters(batch{i}.parameters, expected.params, ...
                                   spec.id, batch{i}.id);
            end
            if isempty(spec.tell_order)
                order = 1:size_now;
            else
                order = double(spec.tell_order(:)') + 1;
                order = order(order <= size_now);
            end
            pruned = false(1, size_now);
            if ~isempty(spec.intermediate)
                for step = 0:spec.intermediate_steps-1
                    for position = order
                        if pruned(position), continue; end
                        trial = batch{position};
                        expected = golden{double(trial.id) + 1};
                        reports = elements(expected.intermediates);
                        event = reports{step + 1};
                        assert(event.step == step);
                        decision = optimizer.intermediate(trial, step, event.score);
                        assert(decision == event.should_prune);
                        if decision, pruned(position) = true; end
                    end
                end
            end
            for position = order
                if pruned(position), continue; end
                trial = batch{position};
                expected = golden{double(trial.id) + 1};
                if any(double(spec.failed_trial_ids) == double(trial.id))
                    optimizer.tell(trial, [], 'failed', ...
                        ['n4m.error.v1|' expected.error.code '|0|' expected.error.message]);
                else
                    optimizer.tell(trial, expected.score);
                end
            end
        end
        verify_trace(optimizer.trials(), space, spec, golden);
        fprintf('HPO MATLAB/Octave golden: %s (%d trials)\n', spec.id, numel(golden));
    catch err
        optimizer.close();
        rethrow(err);
    end
    optimizer.close();
end
fprintf('All %d HPO Octave golden traces match Python/native\n', numel(specs));
end

function items = elements(value)
if iscell(value)
    items = value(:)';
elseif isstruct(value)
    items = num2cell(value(:)');
elseif isempty(value)
    items = {};
else
    error('n4m:optimizer', 'expected a JSON array of objects');
end
end

function space = search_space(declared)
blank = struct('name', '', 'kind', '', 'low', [], 'high', [], ...
               'step', [], 'choices', [], 'length', [], 'integer', []);
space = repmat(blank, 1, numel(declared));
for i = 1:numel(declared)
    item = declared{i};
    space(i).name = item.name;
    space(i).kind = item.kind;
    if strcmp(item.kind, 'categorical') || strcmp(item.kind, 'ordinal')
        space(i).choices = item.args;
    else
        space(i).low = item.args(1);
        space(i).high = item.args(2);
        if numel(item.args) > 2, space(i).step = item.args(3); end
        if strcmp(item.kind, 'sorted_tuple')
            space(i).length = item.length;
        end
    end
end
end

function compare_parameters(actual, expected, spec_id, trial_id)
names = sort(fieldnames(expected));
assert(isequal(sort(fieldnames(actual)), names), ...
       '%s trial %d: parameter names differ', spec_id, trial_id);
for i = 1:numel(names)
    name = names{i};
    a = actual.(name);
    b = expected.(name);
    if isnumeric(a) && isnumeric(b)
        assert(isequal(double(a), double(b)), ...
               '%s trial %d: parameter %s differs', spec_id, trial_id, name);
    else
        assert(isequal(a, b), '%s trial %d: parameter %s differs', ...
               spec_id, trial_id, name);
    end
end
end

function value = pool_string(bytes, offsets, index)
first = double(offsets(index)) + 1;
last = double(offsets(index + 1));
if last < first
    value = '';
else
    value = native2unicode(uint8(bytes(first:last)), 'UTF-8');
end
end

function verify_trace(trace, space, spec, golden)
assert(trace.trace_format_version == 1);
assert(trace.n_trials == numel(golden));
assert(trace.n_params == numel(space));
reported = 0;
last_event = -1;
for i = 1:numel(golden)
    expected = golden{i};
    last_event = max(last_event, expected.terminal_at);
    assert(double(trace.trial_ids_i64(i)) == expected.id);
    assert(double(trace.trial_ask_sequence(i)) == expected.asked_at);
    assert(trace.trial_status(i) == expected.status);
    assert(double(trace.trial_terminal_sequence(i)) == expected.terminal_at);
    if expected.status == 1
        assert(isequal(trace.trial_scores(i), expected.score));
    else
        assert(isnan(trace.trial_scores(i)));
    end
    for j = 1:numel(space)
        name = space(j).name;
        cell_index = (i - 1) * numel(space) + j;
        assert(trace.trial_param_active(cell_index) == 1);
        if strcmp(space(j).kind, 'categorical')
            index = double(trace.trial_param_category_index(cell_index)) + 1;
            assert(index >= 1);
            value = space(j).choices{index};
        else
            value = trace.trial_param_values(i, j);
        end
        if isnumeric(value)
            assert(isequal(double(value), double(expected.params.(name))));
        else
            assert(isequal(value, expected.params.(name)));
        end
    end
    first = double(trace.trial_intermediate_offsets(i)) + 1;
    last = double(trace.trial_intermediate_offsets(i + 1));
    if isempty(spec.intermediate)
        assert(last < first);
    else
        reports = elements(expected.intermediates);
        reported = reported + numel(reports);
        assert(last - first + 1 == numel(reports));
        for k = 1:numel(reports)
            index = first + k - 1;
            event = reports{k};
            assert(double(trace.trial_intermediate_sequence(index)) == event.sequence);
            assert(trace.trial_intermediate_steps(index) == event.step);
            assert(isequal(trace.trial_intermediate_scores(index), event.score));
            assert(logical(trace.trial_intermediate_should_prune(index)) == ...
                   event.should_prune);
        end
        pruned = false;
        for k = 1:numel(reports)
            if reports{k}.should_prune
                assert(reports{k}.step == expected.pruned_at);
                pruned = true;
                break;
            end
        end
        if ~pruned, assert(expected.pruned_at == -1); end
    end
    code = pool_string(trace.trial_error_code_utf8, ...
                       trace.trial_error_code_offsets, i);
    message = pool_string(trace.trial_error_message_utf8, ...
                          trace.trial_error_message_offsets, i);
    if isfield(expected, 'error')
        assert(strcmp(code, expected.error.code));
        assert(strcmp(message, expected.error.message));
        assert(logical(trace.trial_error_retryable(i)) == expected.error.retryable);
    else
        assert(isempty(code) && isempty(message));
    end
end
assert(trace.n_intermediates == reported);
assert(trace.n_events == last_event + 1);
end
