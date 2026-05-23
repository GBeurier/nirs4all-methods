-- SPDX-License-Identifier: CECILL-2.1
--
-- Lua / LuaJIT binding around libn4m's `n4m_pls_fit_simple` C ABI
-- helper. Uses LuaJIT's built-in FFI library — no external rocks
-- required. Parity-gated against the shared cross-binding fixture
-- `bindings/js/test/parity_fixture.json` at machine epsilon.
--
-- Requires LuaJIT (the FFI module is LuaJIT-specific, not standard
-- Lua 5.x). The plain-Lua port would use `lua-ffi-libffi` or the
-- `alien` rock; we ship the canonical zero-dep LuaJIT path here.

local ffi = require("ffi")

ffi.cdef[[
typedef int n4m_status_t;
const char* n4m_get_version_string(void);
unsigned int n4m_get_abi_version_major(void);
unsigned int n4m_get_abi_version_minor(void);
unsigned int n4m_get_abi_version_patch(void);
n4m_status_t n4m_pls_fit_simple(
    const double* x, const double* y,
    int n, int p, int q, int n_components,
    double* coefficients_out,
    double* x_mean_out,
    double* y_mean_out,
    double* predictions_out);
]]

local function load_libp4a()
  local override = os.getenv("PLS4ALL_LIB")
  local candidates = {}
  if override and override ~= "" then table.insert(candidates, override) end
  table.insert(candidates, "p4a")          -- standard loader
  table.insert(candidates, "libn4m.so")
  table.insert(candidates, "libn4m.dylib")

  -- Walk up from this file to find a CMake dev-release build.
  local source = debug.getinfo(1, "S").source:sub(2)
  local cur = source:match("^(.*)/[^/]+$") or "."
  while cur and cur ~= "/" and cur ~= "" do
    for _, name in ipairs({ "libn4m.so", "libn4m.dylib" }) do
      local path = cur .. "/build/dev-release/cpp/src/" .. name
      table.insert(candidates, path)
    end
    local parent = cur:match("^(.*)/[^/]+$")
    if not parent or parent == cur then break end
    cur = parent
  end

  for _, path in ipairs(candidates) do
    local ok, lib = pcall(ffi.load, path)
    if ok then return lib end
  end
  error("could not load libn4m (set PLS4ALL_LIB to its absolute path)")
end

local lib = load_libp4a()

local M = {}

--- Returns the libn4m runtime version, e.g. "0.92.0+abi.1.13.0".
function M.version()
  local raw = lib.n4m_get_version_string()
  if raw == nil then return "" end
  return ffi.string(raw)
end

--- Returns {major, minor, patch} from the libn4m ABI.
function M.abi_version()
  return {
    tonumber(lib.n4m_get_abi_version_major()),
    tonumber(lib.n4m_get_abi_version_minor()),
    tonumber(lib.n4m_get_abi_version_patch()),
  }
end

--- Fit a SIMPLS PLS regression on row-major matrices x and y.
--
-- `x` and `y` are Lua sequences (1-indexed tables) of floats with
-- length n*p and n*q respectively. Returns a table with fields
-- `coefficients`, `x_mean`, `y_mean`, `predictions`, each as a Lua
-- sequence. The C ABI is row-major; the order of `x` is
-- (sample_0_feat_0, sample_0_feat_1, ..., sample_1_feat_0, ...).
function M.pls_fit(x, y, n, p, q, n_components)
  if #x ~= n * p then
    error(string.format("x length %d != n*p (%d)", #x, n * p))
  end
  if #y ~= n * q then
    error(string.format("y length %d != n*q (%d)", #y, n * q))
  end
  if n_components < 1 then
    error("n_components must be >= 1")
  end

  local x_buf = ffi.new("double[?]", n * p, x)
  local y_buf = ffi.new("double[?]", n * q, y)
  local coefs = ffi.new("double[?]", p * q)
  local xm    = ffi.new("double[?]", p)
  local ym    = ffi.new("double[?]", q)
  local preds = ffi.new("double[?]", n * q)

  local status = lib.n4m_pls_fit_simple(
    x_buf, y_buf, n, p, q, n_components, coefs, xm, ym, preds)
  if status ~= 0 then
    error(string.format("n4m_pls_fit_simple failed with status %d", status))
  end

  local function to_table(buf, len)
    local t = {}
    for i = 0, len - 1 do t[i + 1] = buf[i] end
    return t
  end

  return {
    n = n, p = p, q = q, n_components = n_components,
    coefficients = to_table(coefs, p * q),
    x_mean       = to_table(xm,    p),
    y_mean       = to_table(ym,    q),
    predictions  = to_table(preds, n * q),
  }
end

return M
