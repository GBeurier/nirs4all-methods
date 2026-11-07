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
log_domains = [1e-5, 1e-2; 5, 5000; 7, 7000; 50, 50000; ...
    1000, 1e6; 1e18, 9e18; 2e19, 9e20; 1e21, 9e21; 1e15, 1e16; ...
    0.5, 2.5; 1, 1000];
for i = 1:size(log_domains, 1)
    log_space = blank;
    log_space.name = 'scale';
    log_space.kind = 'log_float';
    log_space.low = log_domains(i, 1);
    log_space.high = log_domains(i, 2);
    log_opt = n4m.Optimizer(log_space, struct('sampler', 'sobol', 'seed', 19));
    log_trial = log_opt.ask();
    value = log_trial.parameters.scale;
    assert(isfinite(value) && value >= log_space.low && value <= log_space.high);
    if (log_space.low == 0.5 && log_space.high == 2.5) || ...
            (log_space.low == 1 && log_space.high == 1000)
        assert(isequal(typecast(double(value), 'uint64'), typecast(double(log_space.low), 'uint64')));
    end
    log_running = n4m.Optimizer.load(log_opt.save(), log_space);
    assert(log_running.trials().trial_status(end) == 0);
    log_opt.tell(log_trial, 0.5);
    log_running.tell(log_trial, 0.5);
    log_completed = n4m.Optimizer.load(log_opt.save(), log_space);
    assert(all(log_completed.trials().trial_status == 1));
    expected = log_opt.ask();
    for restored = {log_running, log_completed}
        actual = restored{1}.ask();
        assert(actual.id == expected.id);
        assert(isequal(typecast(double(actual.parameters.scale), 'uint64'), ...
                       typecast(double(expected.parameters.scale), 'uint64')));
    end
    log_running.close();
    log_completed.close();
    log_opt.close();
end
for sampler = {'cmaes', 'tpe'}
    for domain = {[1e15, 1e16], [2e19, 9e20]}
        log_space = repmat(blank, 1, 2);
        log_space(1).name = 'scale';
        log_space(1).kind = 'log_float';
        log_space(1).low = domain{1}(1);
        log_space(1).high = domain{1}(2);
        log_space(2).name = 'aux';
        log_space(2).kind = 'float';
        log_space(2).low = 0;
        log_space(2).high = 1;
        log_opt = n4m.Optimizer(log_space, struct('sampler', sampler{1}, 'seed', 19, 'direction', 'minimize'));
        for i = 1:60
            log_trial = log_opt.ask();
            value = log_trial.parameters.scale;
            assert(isfinite(value) && value >= domain{1}(1) && value <= domain{1}(2));
            log_opt.tell(log_trial, -log(value));
        end
        log_restored = n4m.Optimizer.load(log_opt.save(), log_space);
        assert(isequal(log_opt.ask(), log_restored.ask()));
        log_restored.close();
        log_opt.close();
    end
end
log_space = blank;
log_space.name = 'scale';
log_space.kind = 'log_int';
log_space.low = 1e15;
log_space.high = 2^53 - 1;
log_opt = n4m.Optimizer(log_space, struct('sampler', 'sobol', 'seed', 19));
log_trial = log_opt.ask();
assert(log_trial.parameters.scale >= log_space.low && log_trial.parameters.scale <= log_space.high);
log_restored = n4m.Optimizer.load(log_opt.save(), log_space);
log_restored.close();
log_opt.close();
% The domain correction exercises all samplers and both tuple paths.
q = 2^53; p = 2^52;
tuple_domains = [0,10;-5,5;1000,1002;-1000,-998;2^40,2^40+2; ...
    2^50,2^50+2;2^51,2^51+2;-2^51-2,-2^51;p,p;p,p+1;p,p+2; ...
    p+1000,p+1003;q-3,q-1;q-2,q;-q,-q+2;-q,-q;-p-2,-p;-q,q;-q+1,q-1];
samplers = {'random','sobol','lhs','ternary','ga','pso','cmaes','tpe','gp_ei'};
for d = 1:size(tuple_domains, 1)
    for sampler = samplers
        for len = [1, 3]
            ds = repmat(blank, 1, 2);
            ds(1).name = 'x'; ds(1).kind = 'sorted_tuple'; ds(1).length = len;
            ds(1).low = tuple_domains(d, 1); ds(1).high = tuple_domains(d, 2); ds(1).integer = true;
            ds(2).name = 'aux'; ds(2).kind = 'float'; ds(2).low = 0; ds(2).high = 1;
            domain_stages_matlab(ds, sampler{1}, 19, true);
        end
    end
end
log_cases = { 'log_float', 0.1, 0.1+2^-56; 'log_float', 1e300, 1e300*(1+1e-14); ...
    'log_float', 1e-5, 1e-5+2^-68; 'log_int', 1e15, 1e15+1; ...
    'log_int', 1e15, 1e15+2; 'log_int', q-2, q; 'log_int', 1e15, 1e15+3; ...
    'log_float', 1, 1+2^-52; 'log_float', 5, 5; 'log_float', 0.5, 2.5; 'log_float', 1, 1000 };
for d = 1:size(log_cases, 1)
    for sampler = {'gp_ei','tpe'}
        for seed = [19, 3]
            ds = repmat(blank, 1, 2);
            ds(1).name = 'x'; ds(1).kind = log_cases{d, 1};
            ds(1).low = log_cases{d, 2}; ds(1).high = log_cases{d, 3};
            ds(2).name = 'aux'; ds(2).kind = 'float'; ds(2).low = 0; ds(2).high = 1;
            domain_stages_matlab(ds, sampler{1}, seed, false);
        end
    end
end
span_max = realmax; span_power = 2^1023;
bad_spans = [-1.7e308,1.7e308; -span_max,span_max; -span_power,span_max/2; -span_power,span_power];
good_spans = [-span_max/2,span_max/2; 0,span_max; -8e307,8e307; -span_power,span_max/2-2^970];
for d = 1:size(bad_spans, 1)
    for tuple_axis = [false, true]
        for aux = [false, true]
            ds = repmat(blank, 1, 1 + aux);
            ds(1).name = 'x'; ds(1).low = bad_spans(d,1); ds(1).high = bad_spans(d,2);
            if tuple_axis, ds(1).kind = 'sorted_tuple'; ds(1).length = 2; else, ds(1).kind = 'float'; end
            if aux, ds(2).name = 'aux'; ds(2).kind = 'float'; ds(2).low = 0; ds(2).high = 1; end
            rejected = false;
            try, invalid = n4m.Optimizer(ds, struct('seed', 19)); invalid.close(); catch, rejected = true; end
            assert(rejected);
        end
    end
end
for d = 1:size(good_spans, 1)
    for tuple_axis = [false, true]
        for sampler = samplers
            ds = repmat(blank, 1, 2);
            ds(1).name = 'x'; ds(1).low = good_spans(d,1); ds(1).high = good_spans(d,2);
            if tuple_axis, ds(1).kind = 'sorted_tuple'; ds(1).length = 2; else, ds(1).kind = 'float'; end
            ds(2).name = 'aux'; ds(2).kind = 'float'; ds(2).low = 0; ds(2).high = 1;
            domain_stages_matlab(ds, sampler{1}, 19, tuple_axis);
        end
    end
end
for sampler = {'random','tpe','gp_ei'}
    for integer_axis = [false, true]
        ds = repmat(blank, 1, 2); ds(1).name = 'x';
        if integer_axis
            ds(1).kind = 'log_int'; ds(1).low = 1e15; ds(1).high = 1e15+2;
        else
            ds(1).kind = 'log_float'; ds(1).low = 1e300; ds(1).high = 1e300*(1+1e-14);
        end
        ds(2).name = 'aux'; ds(2).kind = 'float'; ds(2).low = 0; ds(2).high = 1;
        opt = n4m.Optimizer(ds, struct('sampler',sampler{1},'seed',19));
        middle = ds(1).low + (ds(1).high-ds(1).low)/2;
        values = [ds(1).low, ds(1).high, middle];
        if strcmp(sampler{1}, 'gp_ei')
            rejected = false;
            try, opt.enqueue(struct('x',values(1),'aux',0.25)); catch, rejected = true; end
            assert(rejected);
        else
            for value = values, opt.enqueue(struct('x',value,'aux',0.25)); end
            for value = values
                t = opt.ask();
                assert(isequal(typecast(double(t.parameters.x),'uint64'),typecast(double(value),'uint64')));
                opt.tell(t,0.5);
            end
        end
        opt.close();
    end
end
% RANDOM scalar draw stream is an oracle on the non-phantom-cell range.
oracle_low = 2^52; oracle_high = oracle_low + 1; oracle_count = 6000;
oracle_space = blank; oracle_space.name = 'x'; oracle_space.kind = 'int';
oracle_space.low = oracle_low; oracle_space.high = oracle_high;
scalar = n4m.Optimizer(oracle_space, struct('sampler','random','seed',19));
expected = zeros(1,oracle_count);
for j = 1:oracle_count
    t = scalar.ask(); expected(j) = double(t.parameters.x); scalar.tell(t,0);
end
scalar.close();
for len = [1,3]
    oracle_space.kind = 'sorted_tuple'; oracle_space.length = len; oracle_space.integer = true;
    tuple = n4m.Optimizer(oracle_space, struct('sampler','random','seed',19));
    upper = 0;
    for j = 1:oracle_count/len
        t = tuple.ask(); values = double(t.parameters.x);
        assert(isequal(values(:)', sort(expected((j-1)*len+1:j*len))));
        if len == 1 && values == oracle_high, upper = upper + 1; end
        tuple.tell(t,0);
    end
    if len == 1, assert(abs(upper/oracle_count-0.5) <= 0.03); end
    tuple.close();
end
% Scalar grid stages through each public sampler, including restored continuations.
scalar_domains = [2^53-3,2^53,2; -2^53,2^53,3];
for row = 1:size(scalar_domains,1)
    for axis_kind = {'int','float'}
        axis = repmat(blank,1,2); axis(1).name = 'x'; axis(1).kind = axis_kind{1};
        axis(1).low = scalar_domains(row,1); axis(1).high = scalar_domains(row,2); axis(1).step = scalar_domains(row,3);
        axis(2).name = 'aux'; axis(2).kind = 'float'; axis(2).low = 0; axis(2).high = 1;
        for sampler = {'random','sobol','lhs','ternary','ga','pso','cmaes','tpe','gp_ei'}
            domain_stages_matlab(axis,sampler{1},19,false);
        end
    end
end
% Near-ULP FLOAT grids must expose their intended equally likely cells.
near_grids = [1-2^-52,1,2^-53,3; 0.5-2^-53,0.5,2^-54,3; ...
    2^60-256,2^60,128,3; 1-3*2^-53,1,2^-53,4; 2^52,2^52+4,1.5,3];
for row = 1:size(near_grids,1)
    axis = blank; axis.name='x'; axis.kind='float'; axis.low=near_grids(row,1);
    axis.high=near_grids(row,2); axis.step=near_grids(row,3);
    opt = n4m.Optimizer(axis,struct('sampler','random','seed',19));
    values = zeros(1,6000);
    for j = 1:6000
        t=opt.ask(); values(j)=double(t.parameters.x);
        assert(values(j)>=axis.low && values(j)<=axis.high); opt.tell(t,0);
    end
    [~,~,groups]=unique(values); counts=accumarray(groups(:),1);
    assert(numel(counts)==near_grids(row,4));
    assert(all(abs(counts/6000-1/near_grids(row,4))<=0.03)); opt.close();
end
% T4f public FLOAT bounds, self-enqueue and rational-vector checks.
t4f_c=2^-53; t4f_r1=[-1,1-2^-52,3*t4f_c,400;-.5,.5-t4f_c,1.5*t4f_c,400; ...
  1.3094115832905704,9.007642868376131e21,3.020767460720702,240; ...
  1.3552258541085935,3.1816648682723785e21,9.932851196532727,240];
for row=1:size(t4f_r1,1)
 axis=repmat(blank,1,2);axis(1).name='x';axis(1).kind='float';axis(1).low=t4f_r1(row,1);axis(1).high=t4f_r1(row,2);axis(1).step=t4f_r1(row,3);axis(2).name='aux';axis(2).kind='float';axis(2).low=0;axis(2).high=1;
 for sampler={'cmaes','pso','ga','tpe'}
  opt=n4m.Optimizer(axis,struct('sampler',sampler{1},'seed',19,'direction','minimize'));outside=0;
  for j=1:t4f_r1(row,4),t=opt.ask();x=double(t.parameters.x);outside=outside+(x<axis(1).low||x>axis(1).high);opt.tell(t,-x);end
  assert(outside==0);restored=n4m.Optimizer.load(opt.save(),axis);restored.close();opt.close();
 end
end
t4f_r3=[-1,1,3*t4f_c;-2^43,2^43,3/1024;-2^54,2^54,6;-1,1,1.25*t4f_c;-1,1,1.5*t4f_c];
for row=1:size(t4f_r3,1)
 axis=blank;axis.name='x';axis.kind='float';axis.low=t4f_r3(row,1);axis.high=t4f_r3(row,2);axis.step=t4f_r3(row,3);
 a=n4m.Optimizer(axis,struct('sampler','random','seed',19));b=n4m.Optimizer(axis,struct('sampler','random','seed',3));
 for j=1:300,t=a.ask();x=double(t.parameters.x);b.enqueue(struct('x',x));u=b.ask();assert(isequal(typecast(x,'uint64'),typecast(double(u.parameters.x),'uint64')));a.tell(t,0);b.tell(u,0);end
 a.close();b.close();
end
% First 16 values from exact-rational, one-round Python oracle.
t4f_vectors={
 {'3fde203af977a18e','bfe8892c0c8a0e4a','bfd17566ede808aa','bfeab52861bdac95','3fe22dcef65374eb','3fed55f756a6f43b','bfa7b8182af81b30','3fc1d319af2e62bc','bfe018d371fddb18','3fa9b59bc17ac420','3fe9e10761ac89a6','3feb455755894456','3fe406f0891be619','3fe7ddeec22369f7','3fe1e0764bf9f33b','3fdfa660853c677c'}, ...
 {'428e203af977a18e','c298892c0c8a0e4a','c2817566ede808aa','c29ab52861bdac95','42922dcef65374eb','429d55f756a6f43b','c257b8182af81b30','4271d319af2e62bc','c29018d371fddb18','4259b59bc17ac420','4299e10761ac89a6','429b455755894456','429406f0891be619','4297ddeec22369f7','4291e0764bf9f33b','428fa660853c677c'}, ...
 {'433e203af977a18e','c348892c0c8a0e4a','c3317566ede808aa','c34ab52861bdac95','43422dcef65374eb','434d55f756a6f43b','c307b8182af81b30','4321d319af2e62bc','c34018d371fddb18','4309b59bc17ac420','4349e10761ac89a6','434b455755894456','434406f0891be619','4347ddeec22369f7','4341e0764bf9f33b','433fa660853c677c'}};
for row=1:3
 axis=blank;axis.name='x';axis.kind='float';axis.low=t4f_r3(row,1);axis.high=t4f_r3(row,2);axis.step=t4f_r3(row,3);o=n4m.Optimizer(axis,struct('sampler','random','seed',19));
 for j=1:16,t=o.ask();expected=hex2num(t4f_vectors{row}{j});assert(isequal(typecast(double(t.parameters.x),'uint64'),typecast(expected,'uint64')));o.tell(t,0);end;o.close();
end
% T4f generated and queued checkpoint stages for all public samplers.
t4f_stage_domains=[t4f_r1(1:2,1:3);t4f_r3];
for row=1:size(t4f_stage_domains,1)
 axis=repmat(blank,1,2);axis(1).name='x';axis(1).kind='float';
 axis(1).low=t4f_stage_domains(row,1);axis(1).high=t4f_stage_domains(row,2);axis(1).step=t4f_stage_domains(row,3);
 axis(2).name='aux';axis(2).kind='float';axis(2).low=0;axis(2).high=1;
 for sampler={'random','sobol','lhs','ternary','ga','pso','cmaes','tpe','gp_ei'}
  t4f_stages_matlab(axis,sampler{1});
  opt=n4m.Optimizer(axis,struct('sampler',sampler{1},'seed',19));
  if any(strcmp(sampler{1},{'ga','pso','cmaes','gp_ei'}))
   refused=false;try,opt.enqueue(struct('x',axis(1).low,'aux',0.25));catch,refused=true;end;assert(refused);
  else
   opt.enqueue(struct('x',axis(1).low,'aux',0.25));copy=n4m.Optimizer.load(opt.save(),axis);
   for j=1:6,a=opt.ask();b=copy.ask();assert(a.id==b.id);assert(isequal(typecast(double(a.parameters.x),'uint64'),typecast(double(b.parameters.x),'uint64')));assert(isequal(typecast(double(a.parameters.aux),'uint64'),typecast(double(b.parameters.aux),'uint64')));opt.tell(a,.5);copy.tell(b,.5);end
   copy.close();
  end
  opt.close();
 end
end
% Seed-19 vector pinned from the Python bigint raw-unit oracle.
h = 2^53;
expected_grid = [h,h-2,h-1,h-2,h,h,h-1,h-1,h-2,h-1,h,h,h,h,h,h];
for axis_kind = {'int','float'}
    axis = blank; axis.name = 'x'; axis.kind = axis_kind{1};
    axis.low = h-2; axis.high = h; axis.step = 1;
    scalar = n4m.Optimizer(axis, struct('sampler','random','seed',19));
    counts = zeros(1,3);
    for j = 1:6000
        trial = scalar.ask(); x = double(trial.parameters.x);
        assert(x >= h-2 && x <= h);
        if j <= numel(expected_grid), assert(x == expected_grid(j)); end
        counts(x-(h-2)+1) = counts(x-(h-2)+1)+1;
        scalar.tell(trial,0);
    end
    assert(isequal(counts,[2015,2005,1980])); scalar.close();
end
reference.close();
resumable.close();
% Genuine trial-record mutations, source-only unless MATLAB/Octave is available.
loader_grid_bad = { ...
    '4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000db5912a55f50163f010000000000000002000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff000000000101000000000000000000d03f01f44d02eab9c4f23e00000000000000000000000000000100000000000000f87c41c9ec45f73e00000000000000000000000000000000000000000000000000000079f206fca54cbe86'; ...
    '4e344d4f50540d0a010000002000000050010000000000002201000000000000fefd47e44137aed63c000000000000000100000001000000780000000000000000000000000000000000002440000000000000084001000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000049d81abe4359f13e010000000000000001000000000000000100000000000000000000000100000001000000780000000000001c40ffffffff00000000010000000000000000000000000000000000000000000000000000000000000000000000ffffffffffffffff2dd5cd116e3cdc3e00000000000000000000000000000000000000000000000000000055db2ba5873406ea'; ...
    '4e344d4f50540d0a0100000020000000500100000000000022010000000000007279c6e12b96c67b3c00000000000000010000000100000078010000000000000000000000000000000000f03f9a9999999999b93f00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000086c613419c87f33e01000000000000000200000000000000010000000000000000000000010000000100000078333333333333d33fffffffff000000000101000000000000000000d03f0155a485752ab3dd3e00000000000000000000000000000100000000000000f008a9065ab4e23e0000000000000000000000000000000000000000000000000000008453efa0f3c528c4'; ...
    '4e344d4f50540d0a01000000200000005001000000000000220100000000000086412335e4f789433c0000000000000001000000010000007801000000000000000000f0bf000000000000f03f000000000000b83c00000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a00000013000000000000000000000000000000000000000000000000000000130000000000000060062128d08aec3e01000000000000000200000000000000010000000000000000000000010000000100000078ffffffffffffefbfffffffff000000000101000000000000000000d03f01aff75f0ab6e1d53e0000000000000000000000000000010000000000000068c24a93bf6cdc3e000000000000000000000000000000000000000000000000000000b80d6507cf98d4c0'; ...
    '4e344d4f50540d0a0100000020000000e001000000000000b1010000000000009e900a20946c3bd2af0000000000000002000000040000006d6f64650400000000000000000000000000000000000000000000000000000000000000000002000000030000006f6666020000006f6e020000000000000000000000000000000000f03f0000000000010000007801000000000000000000f03f0000000000001440000000000000004000000000000000000000000000000000000000010000000300000002000000010000007800000000040000006d6f6465020000006f6e0d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000ddfd5cfab8b2fb3e0100000000000000020000000000000001000000000000000000000002000000040000006d6f6465000000000000000000000000030000006f66660101000000780000000000001040ffffffff000000000001000000000000000000d03f0101153dd995e2df3e000000000000000000000000000001000000000000003f384ef356a9e53e000000000000000000000000000000000000000000000000000000006c2ea3ac43e1c8c2'; ...
    '4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000bd0d7ae11a381dfd'; ...
    '4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000840ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90'; ...
    };
loader_grid_spaces = cell(1,7);
for i=1:7
    a=blank; a.name='x'; a.kind='float';
    if i<=2, a.kind='int'; a.low=0; a.high=10; a.step=3;
    elseif i==3, a.low=0; a.high=1; a.step=0.1;
    elseif i==4, a.low=-1; a.high=1; a.step=3*2^-53;
    elseif i==5
        mode=blank; mode.name='mode'; mode.kind='categorical'; mode.choices={'off','on'};
        a.low=1; a.high=5; a.step=2; loader_grid_spaces{i}=[mode,a]; continue;
    else, a.low=0; a.high=4; a.step=2; end
    loader_grid_spaces{i}=a;
end
for i=1:7
    damaged=uint8(sscanf(loader_grid_bad{i},'%2x').'); caught=false;
    try, invalid=n4m.Optimizer.load(damaged,loader_grid_spaces{i}); invalid.close();
    catch, caught=true; end
    assert(caught);
end
loader_grid_good=uint8(sscanf('4e344d4f50540d0a01000000200000006001000000000000330100000000000022b6e27c4643864c3c000000000000000100000001000000780100000000000000000000000000000000001040000000000000004000000000000000000000000000000000000000000000000d9a23bd0a81229034000000000000000000000000000000000000000100000000000000000000000a000000130000000000000000000000000000000000000000000000000000001300000000000000748090c7341c343f010000000000000002000000000000000100000000000000000000000100000001000000780000000000000040ffffffff000000000101000000000000000000d03f01d3f96cf49cbaf33e00000000000000000000000000000100000000000000854533b992c92b3f00000000000000000000000000010000000100000001000000780000000000001040000000000000000000750c03c702595a90','%2x').');
a=n4m.Optimizer.load(loader_grid_good,loader_grid_spaces{6});
b=n4m.Optimizer.load(loader_grid_good,loader_grid_spaces{6});
for i=1:6
    x=a.ask(); y=b.ask(); assert(x.id==y.id);
    assert(isequal(typecast(double(x.parameters.x),'uint64'),typecast(double(y.parameters.x),'uint64')));
    a.tell(x,0.5); b.tell(y,0.5);
end
a.close(); b.close();
fprintf('MATLAB/Octave native optimizer lifecycle and checkpoint OK\n');
end

function score = objective(trial)
p = trial.parameters;
if isempty(p.penalty), penalty = 0.2; else, penalty = p.penalty; end
score = (double(p.components) - 2)^2 + penalty + abs(p.weight - 0.2) + sum(p.knots);
end

function domain_stages_matlab(space, sampler, seed, tuple_axis)
stages = [0,0;1,1;1,0;12,1;12,0;13,1];
for stage = 1:size(stages, 1)
    original = n4m.Optimizer(space, struct('sampler', sampler, 'seed', seed, 'direction', 'minimize'));
    pending = [];
    for j = 1:stages(stage, 1)
        pending = original.ask();
        domain_validate_matlab(pending, space, tuple_axis);
        if ~(stages(stage, 2) && j == stages(stage, 1))
            original.tell(pending, mod(j-1, 7)/7);
        end
    end
    captured = original.trials();
    if isfield(captured, 'trial_duration'), captured = rmfield(captured, 'trial_duration'); end
    restored = n4m.Optimizer.load(original.save(), space);
    history = restored.trials();
    if isfield(history, 'trial_duration'), history = rmfield(history, 'trial_duration'); end
    assert(isequaln(captured, history));
    if stages(stage, 2)
        original.tell(pending, 0.5); restored.tell(pending, 0.5);
    end
    for j = 1:6
        a = original.ask(); b = restored.ask();
        domain_validate_matlab(a, space, tuple_axis); domain_validate_matlab(b, space, tuple_axis);
        assert(a.id == b.id);
        assert(isequal(typecast(double(a.parameters.x), 'uint64'), typecast(double(b.parameters.x), 'uint64')));
        assert(isequal(typecast(double(a.parameters.aux), 'uint64'), typecast(double(b.parameters.aux), 'uint64')));
        original.tell(a, mod(j-1, 7)/7); restored.tell(b, mod(j-1, 7)/7);
    end
    restored.close(); original.close();
end
end

function domain_validate_matlab(trial, space, tuple_axis)
values = double(trial.parameters.x);
assert(all(isfinite(values)) && all(values >= space(1).low) && all(values <= space(1).high));
if ~tuple_axis && ~isempty(space(1).step) && space(1).step > 0
    % These scalar test grids have integral bounds/steps and |x| <= 2^53.
    % Their nonnegative offsets are <= 2^54, so int64 subtraction is exact.
    assert(all(values == fix(values)));
    assert(all(mod(int64(values) - int64(space(1).low), int64(space(1).step)) == 0));
end
if tuple_axis
    assert(issorted(values));
    if isequal(space(1).integer, true), assert(all(values == fix(values))); end
end
end

function t4f_stages_matlab(space,sampler)
stages=[0,0;1,1;1,0;12,1;12,0;13,1];
for stage=1:size(stages,1)
 o=n4m.Optimizer(space,struct('sampler',sampler,'seed',19));pending=[];
 for j=1:stages(stage,1),pending=o.ask();x=double(pending.parameters.x);assert(x>=space(1).low && x<=space(1).high);if ~(stages(stage,2)&&j==stages(stage,1)),o.tell(pending,mod(j-1,7)/7);end,end
 copy=n4m.Optimizer.load(o.save(),space);
 if stages(stage,2),o.tell(pending,.5);copy.tell(pending,.5);end
 for j=1:6,a=o.ask();b=copy.ask();assert(a.id==b.id);assert(isequal(typecast(double(a.parameters.x),'uint64'),typecast(double(b.parameters.x),'uint64')));assert(isequal(typecast(double(a.parameters.aux),'uint64'),typecast(double(b.parameters.aux),'uint64')));o.tell(a,mod(j-1,7)/7);copy.tell(b,mod(j-1,7)/7);end
 copy.close();o.close();
end
end
