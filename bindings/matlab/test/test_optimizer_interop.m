function test_optimizer_interop(directory)
% Continue Python N4MOPT in MATLAB/Octave, then export the reciprocal state.
if nargin ~= 1 || exist(directory, 'dir') ~= 7
    error('n4m:optimizer', 'expected the Python interoperability directory');
end
space(1) = struct('name', 'x', 'kind', 'int', 'low', 1, 'high', 5, ...
                  'step', [], 'choices', [], 'length', [], 'integer', []);
space(2) = struct('name', 'a', 'kind', 'float', 'low', 0.1, 'high', 0.9, ...
                  'step', [], 'choices', [], 'length', [], 'integer', []);
expected = sscanf(fileread(fullfile(directory, 'expected.txt')), '%f');
assert(numel(expected) == 3);

file = fopen(fullfile(directory, 'python.n4mopt'), 'rb');
assert(file >= 0);
python_bytes = fread(file, Inf, '*uint8')';
fclose(file);
from_python = n4m.Optimizer.load(python_bytes, space);
try
    trial = from_python.ask();
    assert(isequal([double(trial.id), double(trial.parameters.x), ...
                    trial.parameters.a], expected'));
catch err
    from_python.close();
    rethrow(err);
end
from_python.close();

from_octave = n4m.Optimizer(space, struct('seed', uint64(42)));
try
    for i = 1:3
        trial = from_octave.ask();
        from_octave.tell(trial, double(trial.parameters.x) + trial.parameters.a);
    end
    bytes = from_octave.save();
    file = fopen(fullfile(directory, 'octave.n4mopt'), 'wb');
    assert(file >= 0);
    written = fwrite(file, bytes, 'uint8');
    fclose(file);
    assert(written == numel(bytes));
    trial = from_octave.ask();
    assert(isequal([double(trial.id), double(trial.parameters.x), ...
                    trial.parameters.a], expected'));
catch err
    from_octave.close();
    rethrow(err);
end
from_octave.close();
fprintf('MATLAB/Octave resumed Python N4MOPT checkpoint exactly\n');
end
