// Smoke for the broad-model-pack additions: ECR, O2PLS (generic fitModel path),
// the AOM-Ridge blender + AOM operator-PLS stack bridges, and the DataTwinning /
// SystematicCircular splitters. Asserts finite, signal-correlated predictions
// and well-formed split masks. Run: node --experimental-vm-modules run_new_pack.mjs
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const n4m = await import(resolve(here, '..', 'dist', 'index.js'))
await n4m.loadModule()

let failed = 0
const ok = (c, m) => { if (c) console.log('  ✓ ' + m); else { console.error('  ✗ ' + m); failed++ } }

// structured regression: y depends on a few wavelengths
const n = 60, p = 16
let s = 9876 >>> 0
const rng = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
const beta = Array.from({ length: p }, (_, j) => (j % 4 === 0 ? 1.3 : j % 5 === 0 ? -0.6 : 0))
const Xd = new Float64Array(n * p)
const Yd = new Float64Array(n)
for (let i = 0; i < n; i++) {
  let yi = 3
  for (let j = 0; j < p; j++) {
    const v = rng() * 2 - 1 + Math.sin((i + j) * 0.05)
    Xd[i * p + j] = v
    yi += v * beta[j]
  }
  Yd[i] = yi + (rng() - 0.5) * 0.05
}
const X = { data: Xd, rows: n, cols: p }
const Y = { data: Yd, rows: n, cols: 1 }

const finite = (a) => a.length > 0 && Array.from(a).every((v) => Number.isFinite(v))
function corr(predData) {
  const a = Array.from(predData), b = Array.from(Yd)
  const ma = a.reduce((x, y) => x + y, 0) / n, mb = b.reduce((x, y) => x + y, 0) / n
  let c = 0, va = 0, vb = 0
  for (let i = 0; i < n; i++) { c += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2 }
  return va > 0 && vb > 0 ? c / Math.sqrt(va * vb) : 0
}

// ---- ECR + O2PLS via the generic dispatcher ----
const ecr = n4m.fitModel('ECR', X, Y, 6, [0.5])
ok(finite(ecr.coefficients), 'ECR coefficients finite')
const ecrPred = n4m.predictModel(ecr, X)
ok(finite(ecrPred.data) && corr(ecrPred.data) > 0.8, `ECR predictions correlate (r=${corr(ecrPred.data).toFixed(3)})`)

const o2 = n4m.fitModel('O2PLS', X, Y, 6, [2, 1, 1])
ok(finite(o2.coefficients), 'O2PLS coefficients finite')
const o2Pred = n4m.predictModel(o2, X)
// O2PLS removes orthogonal variation, so with few predictive components it
// legitimately tracks the signal less tightly than plain PLS — assert positive.
ok(finite(o2Pred.data) && corr(o2Pred.data) > 0.5, `O2PLS predictions correlate (r=${corr(o2Pred.data).toFixed(3)})`)

// ---- AOM-Ridge blender + AOM operator-PLS stack ----
const ridge = n4m.fitAomRidge(X, Y, { cv: 4 })
ok(finite(ridge.coefficients) && finite(ridge.intercept), 'AOM-Ridge coeffs + intercept finite')
const ridgePred = n4m.predictModel(ridge, X)
ok(finite(ridgePred.data) && corr(ridgePred.data) > 0.7, `AOM-Ridge predictions correlate (r=${corr(ridgePred.data).toFixed(3)})`)

const stack = n4m.fitAomStack(X, Y, { cv: 4, maxComponents: 8 })
ok(finite(stack.coefficients) && finite(stack.intercept), 'AOM-Stack coeffs + intercept finite')
const stackPred = n4m.predictModel(stack, X)
ok(finite(stackPred.data) && corr(stackPred.data) > 0.7, `AOM-Stack predictions correlate (r=${corr(stackPred.data).toFixed(3)})`)

// ---- new splitters ----
const dt = n4m.computeSplit('DataTwinning', X, null, { testSize: 0.25 })
const nDt = Array.from(dt).filter((v) => v === 1).length
ok(dt.length === n && nDt > 0 && nDt < n, `DataTwinning mask has ${nDt}/${n} test rows`)

const sc = n4m.computeSplit('SystematicCircular', X, Y, { testSize: 0.25 })
const nSc = Array.from(sc).filter((v) => v === 1).length
ok(sc.length === n && nSc > 0 && nSc < n, `SystematicCircular mask has ${nSc}/${n} test rows`)

if (failed) { console.error(`NEW PACK SMOKE FAILED (${failed})`); process.exit(1) }
console.log('NEW PACK SMOKE PASSED')
