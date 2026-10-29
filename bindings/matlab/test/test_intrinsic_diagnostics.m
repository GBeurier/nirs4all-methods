function test_intrinsic_diagnostics(fixture_path)
% Generic C ABI transport parity: no numerical method implemented in MATLAB.
fx = jsondecode(fileread(fixture_path));
for k = 1:numel(fx.cases)
    c = fx.cases(k);
    out = n4m.procedure(c.method_id, c.X, c.parameters, c.X_target);
    expected_fields = fieldnames(c.expected);
    assert(isequal(sort(fieldnames(out)), sort(expected_fields)));
    for j = 1:numel(expected_fields)
        key = expected_fields{j};
        actual = double(out.(key));
        expected = double(c.expected.(key));
        assert(numel(actual) == numel(expected));
        assert(all(isfinite(actual(:))));
        assert(all(abs(actual(:) - expected(:)) <= 1e-9 * (1 + abs(expected(:)))));
    end
end
fprintf('intrinsic diagnostics: %d native/Octave cross-binding cases PASS\n', numel(fx.cases));
end
