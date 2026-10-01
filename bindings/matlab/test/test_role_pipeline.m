function test_role_pipeline(fixturePath)
% Real MEX/native lifecycle, layout, typed inputs and N4ME refusal checks.
% No mock MEX, numerical host estimator, or skipped native checks.
X = [1 0 2; 0 1 -1; -1 0 1; 0 -1 1; 1 1 1; -1 -1 -2];
Y = [2 1; 3 -1; -2 -1; -3 1; 5 0; -5 0];
featureNames = {'source:nir/first.column', 'source:nir/second', 'source:nir/third'};
steps = {step('models.regularized.ridge', struct('alpha', 0, ...
    'center_x', true, 'scale_x', false, 'center_y', true))};
model = n4m.RolePipeline(steps, featureNames, struct('num_threads', 1));
assert(~model.isFitted());
expect_refusal(@() model.predict(X, featureNames), 'not fitted');
model.fit(X, Y);
assert(model.isFitted() && model.nFeaturesIn() == 3 && model.nOutputs() == 2);
assert(isequal(model.featureNames(), featureNames));
% Independent exact affine targets check both strides and target-column order.
assert_close(model.predict(X, featureNames), Y);
assert_close(model.transform(X, featureNames), X);
assert(isequal(size(model.predict(zeros(0, 3), featureNames)), [0 2]));
info = model.stepsInfo();
assert(strcmp(info.method_id, 'models.regularized.ridge'));
assert(strcmp(info.role, 'regressor') && info.state_index == 0);
assert(~info.contains_training_rows && info.n_features_in == 3);
states = model.exportStates();
assert(iscell(states) && numel(states) == 1 && isa(states{1}, 'uint8'));
assert(isequal(char(states{1}(1:4)), 'N4ME'));
replay = n4m.RolePipeline(steps, featureNames);
replay.importStates(states);
assert_close(replay.predict(X, featureNames), Y);
assert(isequal(replay.exportStates(), states));

expect_refusal(@() model.predict(X, featureNames([2 1 3])), 'reordered');
expect_refusal(@() model.predict(X(:, 1:2)), 'columns');
expect_refusal(@() model.setFeatureNames(featureNames), 'fitted');
expect_refusal(@() model.fit(X, Y, struct('groups', int64(1:6))), 'groups');
assert_close(model.predict(X, featureNames), Y); % failed refit did not erase state
expect_refusal(@() model.fit(X, Y(1:5, :)), 'rows');
assert_close(model.predict(X, featureNames), Y);
expect_refusal(@() model.fit(sparse(X), Y), 'full real');
expect_refusal(@() model.fit(complex(X, ones(size(X))), Y), 'full real');
expect_refusal(@() model.fit(reshape(1:24, [2 3 4]), Y), 'two-dimensional');
expect_refusal(@() model.fit(X, Y, struct('unused', 1)), 'unknown');
expect_refusal(@() model.predict(X, {'one'}), 'count');
expect_refusal(@() model.predict(X, {'one', 'one', 'third'}), 'column');
expect_refusal(@() model.decisionFunction(X), 'unsupported');
expect_refusal(@() model.predictLabels(X), 'unsupported');
expect_refusal(@() model.saveobj(), 'cannot be saved');
expect_refusal(@() n4m.RolePipeline.loadobj(struct()), 'cannot be loaded');

bad = states;
bad{1}(1) = uint8(0);
expect_refusal(@() replay.importStates(bad), 'state');
assert_close(replay.predict(X, featureNames), Y); % failed import preserved state
expect_refusal(@() replay.importStates({}), 'state');
expect_refusal(@() replay.importStates({double(states{1})}), 'uint8');
expect_refusal(@() replay.importStates({[states{1}; states{1}]}), 'vector');
other = n4m.RolePipeline({step('models.regularized.ridge', struct('alpha', 1))}, featureNames);
expect_refusal(@() other.importStates(states), 'alpha');
limited = n4m.RolePipeline(steps, featureNames, struct('max_state_bytes', 16));
expect_refusal(@() limited.importStates(states), 'state');
assert(~limited.isFitted());
limited.close();
other.close();

% Native defaults agree with explicitly supplied defaults; no host default table.
defaults = n4m.RolePipeline({step('models.regularized.ridge', struct())});
explicit = n4m.RolePipeline({step('models.regularized.ridge', ...
    struct('alpha', 1, 'center_x', true, 'scale_x', true, 'center_y', true))});
defaults.fit(X, Y);
explicit.fit(X, Y);
assert_close(defaults.predict(X), explicit.predict(X));
assert(isequal(defaults.exportStates(), explicit.exportStates()));
defaults.close();
explicit.close();

% DOUBLE_ARRAY, ENUM, BOOL and INT parameter marshaling plus auxiliary axis.
resampled = n4m.RolePipeline({ ...
    step('preprocessing.resampling.resampler', struct('target_wavelengths', [1000 1002], ...
        'method', 'nearest', 'bounds_error', true, 'tgt_n', int32(0))), ...
    step('models.regularized.ridge', struct('alpha', 0))});
resampled.fit(X, Y(:, 1), struct('axis', [1000 1001 1002]));
assert_close(resampled.transform(X), X(:, [1 3]));
assert(resampled.transformCols() == 2);
roundtrip(resampled, { ...
    step('preprocessing.resampling.resampler', struct('target_wavelengths', [1000 1002], ...
        'method', 'nearest', 'bounds_error', true, 'tgt_n', int32(0))), ...
    step('models.regularized.ridge', struct('alpha', 0))}, X);
resampled.close();

% INT_ARRAY parameters and block_sizes reach the native multiblock engine.
blocks = {step('models.multiblock.so_pls', struct('n_components_per_block', int64([1 1])))};
multiblock = n4m.RolePipeline(blocks);
multiblock.fit(X, Y(:, 1), struct('block_sizes', int64([1 2])));
roundtrip(multiblock, blocks, X);
multiblock.close();

% A transformer inside the same native recipe receives all target columns.
multi = {step('models.pls.pls_regression', struct('n_components', 2)), ...
         step('models.pls.pls_regression', struct('n_components', 2))};
supervised = n4m.RolePipeline(multi);
supervised.fit(X, Y);
assert(isequal(size(supervised.predict(X)), [6 2]));
roundtrip(supervised, multi, X);
supervised.close();

% Class IDs stay exact int64 values; native role decides predict's operation.
classSteps = {step('models.classification.pls_lda', struct('n_components', 1))};
labels = int64([10; 20; 10; 20; 10; 20]);
classifier = n4m.RolePipeline(classSteps, featureNames);
classifier.fit(X, [], struct('labels', labels));
assert(isequal(classifier.classes(), int64([10 20])));
assert(isequal(classifier.predict(X, featureNames), classifier.predictLabels(X, featureNames)));
assert(isa(classifier.predict(X), 'int64'));
assert(isequal(size(classifier.decisionFunction(X)), [6 2]));
assert(isequal(size(classifier.predict(zeros(0, 3))), [0 1]));
expect_refusal(@() classifier.predictProba(X), 'unsupported');
classReplay = n4m.RolePipeline(classSteps, featureNames);
classReplay.importStates(classifier.exportStates());
assert(isequal(classifier.predict(X), classReplay.predict(X)));
expect_refusal(@() classifier.fit(X, [], struct('labels', ...
    repmat(bitshift(uint64(1), 63), 6, 1))), 'int64');
classReplay.close();
classifier.close();

% Probability output uses the native classifier's width and column-major view.
probabilitySteps = {step('models.classification.pls_logistic', struct('n_components', 1))};
probabilityModel = n4m.RolePipeline(probabilitySteps);
probabilityModel.fit(X, [], struct('labels', labels));
probability = probabilityModel.predictProba(X);
assert(isequal(size(probability), [6 2]));
assert(all(isfinite(probability(:))) && all(probability(:) >= 0));
assert_close(sum(probability, 2), ones(6, 1));
assert(isequal(size(probabilityModel.predictProba(zeros(0, 3))), [0 2]));
probabilityReplay = n4m.RolePipeline(probabilitySteps);
probabilityReplay.importStates(probabilityModel.exportStates());
assert_close(probabilityReplay.predictProba(X), probability);
assert_close(probabilityReplay.decisionFunction(X), probabilityModel.decisionFunction(X));
assert(isequal(probabilityReplay.predictLabels(X), probabilityModel.predictLabels(X)));
probabilityReplay.close();
probabilityModel.close();

% Filters keep no N4ME state; state_index describes the native recipe.
filteredSteps = {step('filters.y_outlier', struct()), ...
    step('preprocessing.scatter.snv', struct()), step('models.regularized.ridge', struct())};
filtered = n4m.RolePipeline(filteredSteps);
filtered.fit(X, Y(:, 1));
filteredInfo = filtered.stepsInfo();
assert(isequal([filteredInfo.state_index], int64([-1 0 1])));
assert(numel(filtered.exportStates()) == 2);
roundtrip(filtered, filteredSteps, X);
filtered.close();

% Export permission is enforced by native per-state training-row metadata.
kernelSteps = {step('preprocessing.scatter.snv', struct()), ...
    step('models.pls.kernel', struct('n_components', 1, 'kernel', 'rbf'))};
kernel = n4m.RolePipeline(kernelSteps);
kernel.fit(X, Y(:, 1));
kernelInfo = kernel.stepsInfo();
assert(~kernelInfo(1).contains_training_rows && kernelInfo(2).contains_training_rows);
expect_refusal(@() kernel.exportStates(), 'training rows');
kernelStates = kernel.exportStates(true);
kernelReplay = n4m.RolePipeline(kernelSteps);
kernelReplay.importStates(kernelStates);
assert_close(kernelReplay.predict(X), kernel.predict(X));
expect_refusal(@() kernelReplay.exportStates(), 'training rows');
kernelReplay.close();
kernel.close();

expect_refusal(@() n4m.RolePipeline({}), 'step');
expect_refusal(@() n4m.RolePipeline({step('unknown.method', struct())}), 'method');
expect_refusal(@() n4m.RolePipeline({step('preprocessing.scatter.snv', struct())}), 'regressor');
expect_refusal(@() n4m.RolePipeline({step('models.regularized.ridge', struct()), ...
    step('models.regularized.ridge', struct())}), 'last');
expect_refusal(@() n4m.RolePipeline({step('models.regularized.ridge', struct('extra', 1))}), 'parameter');
expect_refusal(@() n4m.RolePipeline({step('models.regularized.ridge', struct('center_x', 2))}), 'boolean');
expect_refusal(@() n4m.RolePipeline({step('models.pls.pls_regression', ...
    struct('n_components', 1.5))}), 'integer');
expect_refusal(@() n4m.RolePipeline({step('models.pls.kernel', struct('kernel', 'invented'))}), 'kernel');
expect_refusal(@() n4m.RolePipeline(steps, {'same', 'same'}), 'duplicate');
expect_refusal(@() n4m.RolePipeline(steps, {['name' char(0)]}), 'NUL');
expect_refusal(@() n4m.RolePipeline(steps, {}, struct('unknown_option', 1)), 'unknown');
expect_refusal(@() n4m.RolePipeline(steps, {}, struct('num_threads', 0)), 'positive');

% The internal entry point never interprets a user-supplied pointer as a model.
expect_refusal(@() mex_result('is_fitted', double(1)), 'uint64');
expect_refusal(@() mex_result('is_fitted', intmax('uint64')), 'unknown');
expect_refusal(@() mex_result('predict', intmax('uint64'), X, {}), 'unknown');
raw = n4m.n4m_role_pipeline_mex('create', steps, featureNames, struct());
n4m.n4m_role_pipeline_mex('close', raw);
n4m.n4m_role_pipeline_mex('close', raw);
expect_refusal(@() mex_result('is_fitted', raw), 'closed');

replay.close();
model.close();
model.close();
expect_refusal(@() model.predict(X), 'closed');
destructed = n4m.RolePipeline(steps);
delete(destructed);
if nargin < 1
    fixturePath = getenv('N4M_ROLE_PIPELINE_FIXTURE');
end
if isempty(fixturePath)
    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    fixturePath = fullfile(repoRoot, 'parity', 'fixtures', 'role_pipeline_negative.json');
end
test_fixture(fixturePath);
fprintf('MATLAB/Octave native RolePipeline lifecycle, operations and N4ME OK\n');
end

function value = step(method, params)
value = struct('method_id', method, 'params', params);
end

function roundtrip(model, steps, X)
restored = n4m.RolePipeline(steps);
restored.importStates(model.exportStates());
assert_close(restored.predict(X), model.predict(X));
assert_close(restored.transform(X), model.transform(X));
restored.close();
end

function assert_close(actual, expected)
assert(isequal(size(actual), size(expected)));
assert(all(abs(double(actual(:)) - double(expected(:))) <= 1e-9));
end

function expect_refusal(call, fragment)
caught = false;
try
    call();
catch failure
    caught = true;
    assert(~isempty(strfind(lower(failure.message), lower(fragment)))); %#ok<STREMP>
end
assert(caught, 'Expected native/binding refusal did not occur');
end

function value = mex_result(command, varargin)
% Preserve one-output arity when an anonymous negative test discards output.
value = n4m.n4m_role_pipeline_mex(command, varargin{:});
end

function test_fixture(fixturePath)
% Published Python-produced bytes and outputs are an independent native witness.
[~, ~, extension] = fileparts(fixturePath);
if strcmp(extension, '.m')
    run(fixturePath); % trusted generated test data, never a product import path
    assert(exist('role_pipeline_fixture', 'var') == 1);
    fixture = role_pipeline_fixture;
else
    assert(exist('jsondecode', 'builtin') || exist('jsondecode', 'file'), ...
        ['jsondecode unavailable: use export_role_pipeline_fixture.py and pass ' ...
         'its generated .m path to test_role_pipeline']);
    fixture = jsondecode(fileread(fixturePath));
end
runtime = n4m.version();
runtimeAbi = regexp(runtime, '\+abi\.([0-9]+\.[0-9]+\.[0-9]+)$', 'tokens', 'once');
assert(~isempty(runtimeAbi) && strcmp(runtimeAbi{1}, fixture.abi), ...
    'RolePipeline fixture ABI must exactly match the loaded libn4m');
X = fixture_matrix(fixture.x_train);
Y = fixture_matrix(fixture.y_train);
Xt = fixture_matrix(fixture.x_test);
features = fixture.feature_names;
regressionSteps = fixture_steps(fixture.regression.steps);
stored = n4m.RolePipeline(regressionSteps, features, struct('num_threads', 1));
stored.importStates(fixture_states(fixture.regression.states));
assert_close(stored.predict(Xt, features), fixture_matrix(fixture.regression.predict));
assert_close(stored.transform(Xt, features), fixture_matrix(fixture.regression.transform));
assert(isequal(stored.exportStates(), fixture_states(fixture.regression.states)));
trained = n4m.RolePipeline(regressionSteps, features, struct('num_threads', 1));
trained.fit(X, Y);
assert_close(trained.predict(Xt, features), fixture_matrix(fixture.regression.predict));
assert_close(trained.transform(Xt, features), fixture_matrix(fixture.regression.transform));
trained.close();

classificationSteps = fixture_steps(fixture.classification.steps);
classNames = fixture.classification.class_names;
classStates = fixture_states(fixture.classification.states);
labels = fixture_label_ids(fixture.labels_train, classNames);
expectedLabels = fixture_label_ids(fixture.classification.predict, classNames);
classifier = n4m.RolePipeline(classificationSteps, features, struct('num_threads', 1));
classifier.importStates(classStates);
assert(isequal(classifier.predict(Xt, features), expectedLabels));
assert_close(classifier.decisionFunction(Xt), ...
    fixture_matrix(fixture.classification.decision_function));
assert(isequal(classifier.classes(), int64(0:numel(classNames) - 1)));
assert(isequal(classifier.exportStates(), classStates));
trained = n4m.RolePipeline(classificationSteps, features, struct('num_threads', 1));
trained.fit(X, [], struct('labels', labels));
assert(isequal(trained.predict(Xt), expectedLabels));
assert_close(trained.decisionFunction(Xt), ...
    fixture_matrix(fixture.classification.decision_function));
trained.close();
classifier.close();

cases = fixture_cells(fixture.cases);
for k = 1:numel(cases)
    witness = cases{k};
    if strcmp(witness.stage, 'create')
        expect_refusal(@() n4m.RolePipeline(fixture_steps(witness.steps)), witness.message);
    elseif strcmp(witness.stage, 'import')
        candidate = n4m.RolePipeline(fixture_steps(witness.steps), features);
        if isfield(witness, 'state_bytes')
            payloads = witness.state_bytes;
        else
            payloads = fixture_cells(witness.states);
            for j = 1:numel(payloads)
                payloads{j} = fixture_base64(payloads{j});
            end
        end
        expect_refusal(@() candidate.importStates(payloads), witness.message);
        assert(~candidate.isFitted());
        candidate.close();
    elseif strcmp(witness.stage, 'predict')
        if isfield(witness, 'drop_last_column') && witness.drop_last_column
            expect_refusal(@() stored.predict(Xt(:, 1:end - 1)), witness.message);
        else
            expect_refusal(@() stored.predict(Xt, witness.feature_names), witness.message);
        end
    elseif strcmp(witness.stage, 'export')
        candidate = n4m.RolePipeline(fixture_steps(witness.steps), features);
        candidate.fit(X, fixture_matrix(fixture.(witness.y)));
        expect_refusal(@() candidate.exportStates(), witness.message);
        allowed = candidate.exportStates(true);
        assert(~isempty(allowed));
        candidate.close();
    elseif strcmp(witness.stage, 'fit')
        candidate = n4m.RolePipeline(fixture_steps(witness.steps), features);
        candidate.fit(X, fixture_matrix(fixture.(witness.y)));
        assert_close(candidate.predict(Xt), fixture_matrix(witness.predict));
        candidate.close();
    else
        error('Unhandled native RolePipeline witness stage: %s', witness.stage);
    end
end
nameCases = fixture_cells(fixture.name_cases);
for k = 1:numel(nameCases)
    witness = nameCases{k};
    if strcmp(witness.stage, 'predict')
        expect_refusal(@() stored.predict(Xt, witness.feature_names), 'NUL');
    elseif strcmp(witness.stage, 'fit') || strcmp(witness.stage, 'import')
        expect_refusal(@() n4m.RolePipeline(regressionSteps, witness.feature_names), 'NUL');
    else
        error('Unhandled native feature-name witness stage');
    end
end
stored.close();
fprintf('Published RolePipeline ABI %s stored-state parity and native negatives OK\n', fixture.abi);
end

function values = fixture_cells(value)
if iscell(value)
    values = reshape(value, 1, []);
elseif isstruct(value)
    values = arrayfun(@(v) v, value, 'UniformOutput', false);
    values = reshape(values, 1, []);
elseif isempty(value)
    values = {};
else
    error('Expected fixture cell or struct array');
end
end

function result = fixture_matrix(value)
if isnumeric(value)
    result = double(value);
elseif isempty(value)
    result = [];
elseif all(cellfun(@(v) isnumeric(v) && isscalar(v), value))
    result = reshape(cellfun(@double, value), [], 1);
else
    rows = cell(1, numel(value));
    for k = 1:numel(value)
        rows{k} = reshape(fixture_matrix(value{k}), 1, []);
    end
    result = vertcat(rows{:});
end
end

function result = fixture_steps(value)
values = fixture_cells(value);
result = cell(1, numel(values));
for k = 1:numel(values)
    nativeId = values{k}.class;
    assert(strncmp(nativeId, 'n4m:', 4));
    result{k} = step(nativeId(5:end), values{k}.params);
end
end

function result = fixture_states(value)
values = fixture_cells(value);
result = cell(1, numel(values));
for k = 1:numel(values)
    if isfield(values{k}, 'n4me_bytes')
        result{k} = values{k}.n4me_bytes;
    else
        result{k} = fixture_base64(values{k}.n4me_base64);
    end
end
end

function result = fixture_label_ids(labels, classNames)
result = zeros(numel(labels), 1, 'int64');
for k = 1:numel(labels)
    index = find(strcmp(labels{k}, classNames));
    assert(numel(index) == 1);
    result(k) = int64(index - 1);
end
end

function bytes = fixture_base64(encoded)
% Test-data transport only. Scientific outputs and N4ME bytes stay unchanged.
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
assert(ischar(encoded) && mod(numel(encoded), 4) == 0);
bytes = zeros(1, numel(encoded) / 4 * 3, 'uint8');
written = 0;
for k = 1:4:numel(encoded)
    digits = zeros(1, 4, 'uint32');
    padding = 0;
    for j = 1:4
        if encoded(k + j - 1) == '='
            assert(j >= 3 && k + 3 == numel(encoded));
            padding = padding + 1;
        else
            assert(padding == 0);
            index = find(alphabet == encoded(k + j - 1));
            assert(numel(index) == 1);
            digits(j) = uint32(index - 1);
        end
    end
    bits = bitor(bitor(bitshift(digits(1), 18), bitshift(digits(2), 12)), ...
        bitor(bitshift(digits(3), 6), digits(4)));
    decoded = uint8([bitand(bitshift(bits, -16), 255), ...
        bitand(bitshift(bits, -8), 255), bitand(bits, 255)]);
    bytes(written + (1:3 - padding)) = decoded(1:3 - padding);
    written = written + 3 - padding;
end
bytes = bytes(1:written);
end
