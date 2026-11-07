function test_snv_parameter_contract(mode)
% Actual preprocessing/role MEX contract; no numerical implementation here.
if nargin < 1, mode = 'all'; end
fixture = getenv('N4M_SNV_FIXTURE_PATH');
if isempty(fixture)
    root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    fixture = fullfile(root, 'parity', 'fixtures', 'snv_parameter_contract_v1.json');
end
doc = jsondecode(fileread(fixture));
assert(strcmp(doc.schema, 'n4m.snv-parameter-contract.v1'));
X = doc.matrix;
% Mandatory real positive anchor, before any refusal can be credited.
a = n4m.snv_transform(X);
assert_close(a, item(doc.references, 1).values);
fprintf('SNV_NATIVE_POSITIVE %s\n', which('n4m.n4m_preprocess_mex'));
results = {};
for i = 1:numel(doc.references)
    row = item(doc.references, i);
    check(['reference-' row.id], @() reference(row));
end
for name = {'with_sd', 'dof'}
    check(['unknown-' name{1}], @() expect_refusal(@() n4m.snv_transform(X, name{1}, 1)));
end
if ~strcmp(mode, 'names')
    for i = 1:numel(doc.typed_values)
        row = item(doc.typed_values, i);
        if isempty(row.admit.octave), continue; end
        check(['public-' row.id], @() typed(row, false));
        check(['raw-' row.id], @() typed(row, true));
    end
    for degree = [4 5 double(intmax('int32'))]
        check(sprintf('extent-%d', degree), @() expect_refusal(@() n4m.snv_transform(X, 'ddof', degree)));
    end
end
unavailable = {};
if strcmp(mode, 'all')
    if isempty(which('n4m.n4m_role_pipeline_mex'))
        unavailable = {'candidate generic role MEX is required (full build_mex)'};
    else
        check('role-forwarding', @() generic_positive());
        if strcmp(results{end}.status, 'PASS')
            for i = 1:numel(doc.typed_values)
                row = item(doc.typed_values, i);
                if isempty(row.admit.octave), continue; end
                check(['role-' row.id], @() generic_typed(row));
            end
        else
            unavailable{end+1} = 'generic role positive anchor failed; no refusal credited';
        end
    end
end
failed = 0;
for i = 1:numel(results)
    fprintf('%s %s %s\n', results{i}.status, results{i}.id, results{i}.error);
    if ~strcmp(results{i}.status, 'PASS'), failed = failed + 1; end
end
fprintf('SNV_CONTRACT_TOTAL %d FAILURES %d UNAVAILABLE %d MODE %s\n', numel(results), failed, numel(unavailable), mode);
if exist('/proc/self/maps', 'file')
    maps = strsplit(fileread('/proc/self/maps'), '\n');
    for i = 1:numel(maps)
        if ~isempty(strfind(maps{i}, 'libn4m')) || ~isempty(strfind(maps{i}, 'preprocess_mex'))
            fprintf('MAP %s\n', maps{i});
        end
    end
end
for i = 1:numel(unavailable), fprintf('UNAVAILABLE %s\n', unavailable{i}); end
assert(failed == 0 && isempty(unavailable), 'SNV contract has failed or unavailable rows');
    function check(id, fn)
        try
            fn(); results{end+1} = struct('id', id, 'status', 'PASS', 'error', '');
        catch err
            results{end+1} = struct('id', id, 'status', 'FAIL', 'error', err.message);
        end
    end
    function reference(row)
        p = row.params;
        assert_close(n4m.snv_transform(X, 'with_mean', p.with_mean, 'with_std', p.with_std, 'ddof', p.ddof), row.values);
    end
    function typed(row, raw)
        p = struct('with_mean', true, 'with_std', true, 'ddof', 0);
        p.(row.parameter) = host_value(row);
        if raw
            fn = @() n4m.n4m_preprocess_mex('snv', X, p.with_mean, p.with_std, p.ddof);
        else
            fn = @() n4m.snv_transform(X, 'with_mean', p.with_mean, 'with_std', p.with_std, 'ddof', p.ddof);
        end
        if row.admit.octave, got = fn(); assert(isequal(size(got), size(X)));
        else, expect_refusal(fn); end
    end
    function generic_typed(row)
        params = struct('with_mean', true);
        params.(row.parameter) = host_value(row);
        steps = {struct('method_id', 'preprocessing.scatter.snv', 'params', params), ...
                 struct('method_id', 'models.regularized.ridge', 'params', struct())};
        if row.admit.octave
            model = n4m.RolePipeline(steps); cleanup = onCleanup(@() model.close());
            model.fit(X, X(:,1)); model.predict(X);
        else
            % BC2: earlier valid setters are allowed. Above-i32 reaches its
            % setter and native status refusal; no SNV transform may occur.
            expect_refusal(@() generic_fit(steps));
        end
    end
    function generic_fit(steps)
        model = n4m.RolePipeline(steps); cleanup = onCleanup(@() model.close());
        model.fit(X, X(:,1));
    end
    function generic_positive()
        steps = {struct('method_id', 'preprocessing.scatter.snv', 'params', struct('with_std', false)), ...
                 struct('method_id', 'models.regularized.ridge', 'params', struct())};
        model = n4m.RolePipeline(steps); cleanup = onCleanup(@() model.close());
        model.fit(X, X(:,1)); model.predict(X);
        states = model.exportStates();
        replay = n4m.RolePipeline(steps); cleanup2 = onCleanup(@() replay.close());
        replay.importStates(states); assert(isequal(replay.predict(X), model.predict(X)));
    end
end
function row = item(rows, i)
if iscell(rows), row = rows{i}; else, row = rows(i); end
end
function v = host_value(row)
switch row.tag
    case 'literal', v = row.value;
    case 'nan', v = NaN;
    case 'infinity', v = Inf;
    otherwise, error('SNV:fixture', 'Unconsumed Octave tag %s', row.tag);
end
end
function expect_refusal(fn)
refused = false;
try, fn(); catch, refused = true; end
assert(refused, 'Expected SNV refusal');
end
function assert_close(actual, expected)
assert(isequal(size(actual), size(expected)));
assert(all(abs(actual(:)-expected(:)) <= 1e-12));
end
