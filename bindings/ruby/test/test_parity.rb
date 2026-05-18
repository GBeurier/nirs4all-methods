# SPDX-License-Identifier: CECILL-2.1
#
# Cross-binding parity gate for the Ruby binding. Builds the same
# deterministic (X, Y) the Python generator writes, fits the model
# through Fiddle, and compares against
# bindings/js/test/parity_fixture.json. Exits non-zero on failure.

require "json"
require "pathname"

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "pls4all"

repo_root = Pathname.new(ENV["REPO_ROOT"] || Dir.pwd).expand_path
unless (repo_root + "bindings/js/test/parity_fixture.json").file?
  cur = Pathname.new(__FILE__).realpath.dirname
  loop do
    break if (cur + "bindings/js/test/parity_fixture.json").file?
    parent = cur.parent
    if parent == cur
      warn "could not locate repo root containing the shared fixture"
      exit 1
    end
    cur = parent
  end
  repo_root = cur
end

fixture_path = repo_root + "bindings/js/test/parity_fixture.json"
fixture = JSON.parse(fixture_path.read)

n = fixture["n"]
p = fixture["p"]
q = fixture["q"]
n_components = fixture["n_components"]

# Build the same deterministic input the Python generator wrote.
x = Array.new(n * p, 0.0)
y = Array.new(n * q, 0.0)
n.times do |i|
  p.times do |j|
    x[i * p + j] = Math.sin((i + 1) * (j + 1) * 0.3)
  end
  y[i * q + 0] = x[i * p + 0] + 0.5 * x[i * p + 1] - 0.3 * x[i * p + 2]
end

fit = Pls4all.pls_fit(x, y, n, p, q, n_components)

def rms_rel(a, b)
  raise "length mismatch #{a.length} vs #{b.length}" if a.length != b.length
  diff = 0.0
  ref = 0.0
  a.each_with_index do |av, i|
    d = av - b[i]
    diff += d * d
    ref += b[i] * b[i]
  end
  denom = [Math.sqrt(ref), Float::MIN].max
  Math.sqrt(diff) / denom
end

err_c = rms_rel(fit.coefficients, fixture["coefficients"])
err_x = rms_rel(fit.x_mean,        fixture["x_mean"])
err_y = rms_rel(fit.y_mean,        fixture["y_mean"])
err_p = rms_rel(fit.predictions,   fixture["predictions"])

major, minor, patch = Pls4all.abi_version
puts "Pls4all.version           = #{Pls4all.version}"
puts "Pls4all.abi_version       = #{major}.#{minor}.#{patch}"
puts "fixture pls4all_version   = #{fixture['pls4all_version']}"
puts format("rmse_rel coefficients    = %.3e", err_c)
puts format("rmse_rel x_mean          = %.3e", err_x)
puts format("rmse_rel y_mean          = %.3e", err_y)
puts format("rmse_rel predictions     = %.3e", err_p)

tol = 1e-12
if [err_c, err_x, err_y, err_p].any? { |e| e > tol }
  warn format("PARITY GATE FAIL (tol = %.1e)", tol)
  exit 1
end
puts "PARITY GATE PASS"
