function test_spectral_encoding()
% Runtime conformance when MATLAB or Octave with compiled MEX is available.
rng(27);
X = randn(40, 19);
V = randn(7, 19);
encoder = n4m.LVSE(8, 2);
encoder.fit(X);
Z = encoder.transform(V);
[A, offset] = encoder.export_linear_operator();
expected = bsxfun(@plus, V * A', offset);
assert(max(abs(Z(:) - expected(:))) < 1e-10);
gcu = n4m.GCU(3);
gcu.fit(X);
H = gcu.transform(V);
assert(all(isfinite(H(:))) && all(H(:) >= 0));
delete(encoder);
delete(gcu);
disp('LVSE/GCU MATLAB/Octave tests passed');
end
