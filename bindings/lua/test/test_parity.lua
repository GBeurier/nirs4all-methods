-- SPDX-License-Identifier: CeCILL-2.1
--
-- Cross-binding parity gate for the Lua / LuaJIT binding. Builds the
-- same deterministic (X, Y) the Python generator writes, fits the
-- model through FFI, and compares against
-- bindings/js/test/parity_fixture.json. Exits non-zero on failure.

local function script_dir()
  local source = debug.getinfo(1, "S").source:sub(2)
  return source:match("^(.*)/[^/]+$") or "."
end

-- Locate the repo root by walking up from the test script.
local function locate_repo_root()
  local override = os.getenv("REPO_ROOT")
  if override and override ~= "" then return override end
  local cur = script_dir()
  while cur and cur ~= "/" and cur ~= "" do
    local f = io.open(cur .. "/bindings/js/test/parity_fixture.json", "r")
    if f then f:close(); return cur end
    local parent = cur:match("^(.*)/[^/]+$")
    if not parent or parent == cur then break end
    cur = parent
  end
  error("could not locate repo root containing the shared fixture")
end

local repo_root = locate_repo_root()
-- Make the binding directory available on the Lua path.
package.path = repo_root .. "/bindings/lua/?.lua;" .. package.path

local pls4all = require("pls4all")

local fixture_path = repo_root .. "/bindings/js/test/parity_fixture.json"
local fh = assert(io.open(fixture_path, "r"))
local raw = fh:read("*a")
fh:close()

-- Tiny hand-rolled JSON reader covering the fixture's known schema.
local function parse_int(s, key)
  return assert(tonumber(s:match('"' .. key .. '"%s*:%s*(%-?%d+)')))
end

local function parse_string(s, key)
  return assert(s:match('"' .. key .. '"%s*:%s*"([^"]*)"'))
end

local function parse_array(s, key)
  local body = assert(s:match('"' .. key .. '"%s*:%s*%[([^%]]*)%]'))
  local t = {}
  for tok in body:gmatch("[^,%s]+") do
    table.insert(t, assert(tonumber(tok)))
  end
  return t
end

local n  = parse_int(raw, "n")
local p  = parse_int(raw, "p")
local q  = parse_int(raw, "q")
local nc = parse_int(raw, "n_components")
local fixture_version = parse_string(raw, "pls4all_version")
local fx_coefs   = parse_array(raw, "coefficients")
local fx_x_mean  = parse_array(raw, "x_mean")
local fx_y_mean  = parse_array(raw, "y_mean")
local fx_preds   = parse_array(raw, "predictions")

local x = {}
local y = {}
for i = 0, n - 1 do
  for j = 0, p - 1 do
    x[i * p + j + 1] = math.sin((i + 1) * (j + 1) * 0.3)
  end
  y[i * q + 1] =
    x[i * p + 0 + 1] + 0.5 * x[i * p + 1 + 1] - 0.3 * x[i * p + 2 + 1]
end

local fit = pls4all.pls_fit(x, y, n, p, q, nc)

local function rms_rel(a, b)
  assert(#a == #b, "length mismatch")
  local diff, ref = 0.0, 0.0
  for i = 1, #a do
    local d = a[i] - b[i]
    diff = diff + d * d
    ref  = ref  + b[i] * b[i]
  end
  local denom = math.max(math.sqrt(ref), 2.2250738585072014e-308)
  return math.sqrt(diff) / denom
end

local err_c = rms_rel(fit.coefficients, fx_coefs)
local err_x = rms_rel(fit.x_mean,        fx_x_mean)
local err_y = rms_rel(fit.y_mean,        fx_y_mean)
local err_p = rms_rel(fit.predictions,   fx_preds)

local abi = pls4all.abi_version()
io.write(string.format("pls4all.version           = %s\n",
                        pls4all.version()))
io.write(string.format("pls4all.abi_version       = %d.%d.%d\n",
                        abi[1], abi[2], abi[3]))
io.write(string.format("fixture pls4all_version   = %s\n", fixture_version))
io.write(string.format("rmse_rel coefficients    = %.3e\n", err_c))
io.write(string.format("rmse_rel x_mean          = %.3e\n", err_x))
io.write(string.format("rmse_rel y_mean          = %.3e\n", err_y))
io.write(string.format("rmse_rel predictions     = %.3e\n", err_p))

local tol = 1e-12
if err_c > tol or err_x > tol or err_y > tol or err_p > tol then
  io.stderr:write(string.format("PARITY GATE FAIL (tol = %.1e)\n", tol))
  os.exit(1)
end
io.write("PARITY GATE PASS\n")
