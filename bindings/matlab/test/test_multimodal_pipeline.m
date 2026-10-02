function test_multimodal_pipeline(fixturePath)
% Actual native fit and cross-host import, using the independent sklearn oracle.
if nargin ~= 1
    error('test:multimodal', 'raw diagnostic fixture JSON path required');
end
fixture = jsondecode(fileread(fixturePath));
recipe = fixture.recipe;
schemas = fixture.source_schemas;
train = raw_blocks(fixture.train);
heldout = raw_blocks(fixture.heldout);
if exist('OCTAVE_VERSION', 'builtin')
    % Octave exposes base64_decode for its numeric binary format, not a raw
    % byte stream. Decode the transport string explicitly, without touching
    % learned states or numeric source values.
    state = decode_bytes(fixture.state);
else
    state = uint8(matlab.net.base64decode(fixture.state));
end
replay = n4m.MultimodalPipeline.fromState(state, recipe, schemas);
cleanupReplay = onCleanup(@() replay.close()); %#ok<NASGU>
check_prediction(replay.predict(heldout), fixture.expected);
z = replay.transform(heldout);
assert(size(z, 2) == 14);
assert(isequal(z(1, 12:14), [0 0 0]));
wrong = schemas;
wrong.image.identity = [wrong.image.identity ':wrong-axis'];
must_fail(@() replay.predict(heldout, wrong));
broken = state;
broken(61) = bitxor(broken(61), uint8(1));
must_fail(@() n4m.MultimodalPipeline.fromState(broken, recipe, schemas));
must_fail(@() n4m.MultimodalPipeline.fromState(state, recipe, wrong));
fresh = n4m.MultimodalPipeline(recipe, schemas);
cleanupFresh = onCleanup(@() fresh.close()); %#ok<NASGU>
fresh.fit(train, fixture.y);
check_prediction(fresh.predict(heldout), fixture.expected);
hydrated = n4m.MultimodalPipeline.fromState(fresh.exportState(), recipe, schemas);
cleanupHydrated = onCleanup(@() hydrated.close()); %#ok<NASGU>
assert(isequal(hydrated.predict(heldout), fresh.predict(heldout)));
before = fresh.predict(heldout);
check_nul_categories(train, fixture.y, recipe, schemas);
train.series(1) = NaN;
must_fail(@() fresh.fit(train, fixture.y));
assert(isequal(fresh.predict(heldout), before));
fresh.close();
fresh.close();
disp('native Octave raw multimodal fit/import/schema/unknown-category/cleanup PASS');
end

function check_nul_categories(train, y, recipe, schemas)
% Distinguish known categories with a shared prefix, NUL and non-ASCII suffix.
nulLabel = ['A' char(0) 'é'];
labels = {'A', nulLabel, '猫'};
for row = 1:size(train.metadata, 1)
    train.metadata{row, 2} = labels{mod(row - 1, 3) + 1};
end
model = n4m.MultimodalPipeline(recipe, schemas);
cleanupModel = onCleanup(@() model.close()); %#ok<NASGU>
model.fit(train, y);
known = model.transform(train);
weight = recipe.source_weights.metadata;
assert(isequal(known(1:3, end-2:end), eye(3) * weight));
query = train;
query.metadata{1, 2} = [nulLabel 'x'];
query.metadata{2, 2} = nulLabel;
query.metadata{3, 2} = 'A';
z = model.transform(query);
assert(isequal(z(1:3, end-2:end), [0 0 0; 0 1 0; 1 0 0] * weight));
bytes = model.exportState();
replay = n4m.MultimodalPipeline.fromState(bytes, recipe, schemas);
cleanupReplay = onCleanup(@() replay.close()); %#ok<NASGU>
assert(isequal(replay.transform(query), z));
assert(isequal(replay.predict(query), model.predict(query)));
assert(isequal(replay.exportState(), bytes));
wrong = schemas;
wrong.nir.identity = [wrong.nir.identity char(0) 'suffix'];
must_fail(@() n4m.MultimodalPipeline(recipe, wrong));
end

function result = raw_blocks(raw)
result = struct();
names = {'nir', 'image', 'series', 'metadata'};
for index = 1:4
    name = names{index};
    if strcmp(name, 'metadata')
        result.(name) = raw.(name);
        if iscell(result.(name)) && isvector(result.(name)) && iscell(result.(name){1})
            result.(name) = vertcat(result.(name){:});
        end
    else
        shape = double(raw.(name).shape(:)');
        % Raw layout conversion only: preserve every sample-first dimension.
        result.(name) = permute(reshape(double(raw.(name).data), fliplr(shape)), numel(shape):-1:1);
    end
end
end

function check_prediction(actual, expected)
assert(numel(actual) == numel(expected));
assert(max(abs(actual(:) - expected(:))) <= 1e-8 * (1 + max(abs(expected(:)))));
end

function must_fail(action)
failed = false;
try
    action();
catch
    failed = true;
end
assert(failed);
end

function result = decode_bytes(value)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
value = value(value ~= '=');
result = zeros(1, floor(numel(value) * 6 / 8), 'uint8');
bits = uint32(0);
count = 0;
position = 1;
for i = 1:numel(value)
    index = find(alphabet == value(i), 1);
    assert(~isempty(index));
    bits = bitor(bitshift(bits, 6), uint32(index - 1));
    count = count + 6;
    if count >= 8
        count = count - 8;
        result(position) = uint8(bitand(bitshift(bits, -count), uint32(255)));
        position = position + 1;
    end
end
end
