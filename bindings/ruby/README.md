# pls4all — Ruby binding

Ruby wrapper around libp4a's `p4a_pls_fit_simple` C ABI helper.
Uses the **Ruby stdlib `Fiddle`** library — zero external gems
required. Parity-gated against the shared cross-binding fixture
(`bindings/js/test/parity_fixture.json`) at machine epsilon
(`rmse_rel = 0.0` — bit-exact).

## Layout

```
bindings/ruby/
├── lib/pls4all.rb             Public API (Fiddle DLL imports)
└── test/test_parity.rb        Parity gate vs shared fixture
```

## Usage

```ruby
$LOAD_PATH.unshift "bindings/ruby/lib"
require "pls4all"

puts Pls4all.version           # "0.91.0+abi.1.13.0"
puts Pls4all.abi_version.inspect # [1, 13, 0]

n = 50; p = 5; q = 1
x = Array.new(n * p, 0.0)      # row-major
y = Array.new(n * q, 0.0)
# fill x, y ...

fit = Pls4all.pls_fit(x, y, n, p, q, 3)
puts "coefficients: #{fit.coefficients.inspect}"
puts "predictions length: #{fit.predictions.length}"
```

## Parity gate

```bash
PLS4ALL_LIB=$(pwd)/build/dev-release/cpp/src/libp4a.so \
LD_LIBRARY_PATH=$(pwd)/build/dev-release/cpp/src \
REPO_ROOT=$(pwd) \
ruby bindings/ruby/test/test_parity.rb
```

Expected output:

```
Pls4all.version           = 0.94.0+abi.1.13.0
Pls4all.abi_version       = 1.13.0
fixture pls4all_version   = 0.94.0+abi.1.13.0
rmse_rel coefficients    = 0.000e+00
rmse_rel x_mean          = 0.000e+00
rmse_rel y_mean          = 0.000e+00
rmse_rel predictions     = 0.000e+00
PARITY GATE PASS
```

## Library lookup

`Pls4all` searches for libp4a in this order:
1. `ENV["PLS4ALL_LIB"]` (absolute path).
2. The standard names (`libp4a.so`, `libp4a.dylib`, `p4a.dll`) via the
   system loader.
3. The repo-local CMake `dev-release` artifact, located by walking up
   from the binding source.

Pure-stdlib: nothing to install. The binding works on the system Ruby
(>= 2.7 — Fiddle has been in core since 1.9; the buffer-pack API used
here works in 2.7+).
