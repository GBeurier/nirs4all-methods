function test_optimizer()
% Native optimizer lifecycle through the same MEX source on Octave and MATLAB.
blank = struct('name', '', 'kind', '', 'low', [], 'high', [], ...
               'step', [], 'choices', [], 'length', [], 'integer', []);
space = repmat(blank, 1, 5);
space(1).name = 'mode';
space(1).kind = 'categorical';
space(1).choices = {'plain', 'sparse'};
space(2).name = 'components';
space(2).kind = 'int';
space(2).low = 1;
space(2).high = 4;
space(3).name = 'penalty';
space(3).kind = 'float';
space(3).low = 0.05;
space(3).high = 0.5;
space(4).name = 'weight';
space(4).kind = 'ordinal';
space(4).choices = [0.2, 0.8];
space(5).name = 'knots';
space(5).kind = 'sorted_tuple';
space(5).length = 2;
space(5).low = 0;
space(5).high = 1;
condition = struct('kind', 'condition_in', 'refs', {{'penalty', 'mode'}}, ...
                   'labels', {{'', 'sparse'}});
options = struct('seed', uint64(42), 'sampler', 'random');
reference = n4m.Optimizer(space, options, condition);
resumable = n4m.Optimizer(space, options, condition);
assert(isempty(reference.best()));
for i = 1:4
    a = reference.ask();
    b = resumable.ask();
    assert(isequal(a, b));
    assert(isa(a.id, 'int64'));
    assert(a.status == 0);
    if strcmp(a.parameters.mode, 'plain')
        assert(isempty(a.parameters.penalty));
    else
        assert(~isempty(a.parameters.penalty));
    end
    score = objective(a);
    assert(~reference.intermediate(a, 0, score));
    assert(~resumable.intermediate(b, 0, score));
    reference.tell(a, score);
    resumable.tell(b, score);
end
blob = resumable.save();
assert(isa(blob, 'uint8') && numel(blob) > 100);
resumable.close();
resumable = n4m.Optimizer.load(blob, space);
for i = 1:4
    a = reference.ask();
    b = resumable.ask();
    assert(isequal(a, b));
    reference.tell(a, objective(a));
    resumable.tell(b, objective(b));
end
assert(isequal(reference.best(), resumable.best()));
trace = resumable.trials();
assert(trace.trace_format_version == 1 && trace.n_trials == 8);
assert(isequal(double(trace.trial_ids_i64), 0:7));
assert(all(trace.trial_status == 1));
assert(numel(trace.trial_intermediate_steps) == 4);
assert(resumable.trials(int64(7)).n_trials == 1);

single = blank;
single.name = 'components';
single.kind = 'int';
single.low = 1;
single.high = 4;
queued = n4m.Optimizer(single, struct('seed', 7));
queued.enqueue(struct('components', 3));
[batch, status] = queued.askBatch(3);
assert(status == 0 && numel(batch) == 3);
assert(batch{1}.parameters.components == 3);
for i = 1:numel(batch)
    queued.tell(batch{i}, double(batch{i}.parameters.components));
end
assert(~isempty(queued.best()));
queued.close();
queued.close();
caught = false;
try
    queued.ask();
catch
    caught = true;
end
assert(caught);

typed_space = repmat(blank, 1, 5);
typed_space(1).name = 'flag';
typed_space(1).kind = 'categorical';
typed_space(1).choices = logical([false, true]);
typed_space(2).name = 'code';
typed_space(2).kind = 'categorical';
typed_space(2).choices = int64([2147483648, 2147483649]);
typed_space(3).name = 'ratio';
typed_space(3).kind = 'categorical';
typed_space(3).choices = [0.25, 0.75];
typed_space(4).name = 'log_k';
typed_space(4).kind = 'log_int';
typed_space(4).low = 1;
typed_space(4).high = 16;
typed_space(5).name = 'log_rate';
typed_space(5).kind = 'log_float';
typed_space(5).low = 0.01;
typed_space(5).high = 1;
typed = n4m.Optimizer(typed_space, struct('seed', 3));
typed_trial = typed.ask();
assert(islogical(typed_trial.parameters.flag));
assert(isa(typed_trial.parameters.code, 'int64'));
assert(any(typed_trial.parameters.code == int64([2147483648, 2147483649])));
assert(any(typed_trial.parameters.ratio == [0.25, 0.75]));
assert(typed_trial.parameters.log_k >= 1 && typed_trial.parameters.log_k <= 16);
assert(typed_trial.parameters.log_rate >= 0.01 && typed_trial.parameters.log_rate <= 1);
typed.tell(typed_trial, 0.5);
pending = typed.ask();
caught = false;
try
    typed.tell(pending, []);
catch
    caught = true;
end
running_trace = typed.trials();
assert(caught && running_trace.trial_status(end) == 0);
typed.tell(pending, 1.5);
typed.close();

sparse_axis = blank;
sparse_axis.name = 'flag';
sparse_axis.kind = 'categorical';
sparse_axis.choices = sparse(logical([false, true]));
caught = false;
try
    n4m.Optimizer(sparse_axis);
catch
    caught = true;
end
assert(caught);

dotted = blank;
dotted.name = 'branches.nir.model.n_components';
dotted.kind = 'int';
dotted.low = 1;
dotted.high = 4;
dag_study = n4m.Optimizer(dotted, struct('seed', 5));
dag_study.enqueue({dotted.name}, 3);
dag_trial = dag_study.ask();
assert(isempty(dag_trial.parameters));
assert(strcmp(dag_trial.parameter_names{1}, dotted.name));
assert(n4m.Optimizer.getParameter(dag_trial, dotted.name) == 3);
dag_study.tell(dag_trial, 1.5);
dag_study.close();

bad = blob;
bad(1) = uint8(0);
caught = false;
try
    n4m.Optimizer.load(bad, space);
catch
    caught = true;
end
assert(caught);
reference.close();
resumable.close();
fprintf('MATLAB/Octave native optimizer lifecycle and checkpoint OK\n');
end

function score = objective(trial)
p = trial.parameters;
if isempty(p.penalty), penalty = 0.2; else, penalty = p.penalty; end
score = (double(p.components) - 2)^2 + penalty + abs(p.weight - 0.2) + sum(p.knots);
end
