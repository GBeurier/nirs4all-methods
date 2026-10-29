// SPDX-License-Identifier: CECILL-2.1
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import * as n4m from '../dist/index.js';
await n4m.loadModule();
const fixture=JSON.parse(readFileSync(new URL('../../../parity/fixtures/intrinsic_diagnostics.v1.json',import.meta.url),'utf8'));
const matrix=x=>({data:Float64Array.from(x.flat()),rows:x.length,cols:x[0].length});
const flatten=x=>Array.isArray(x)?x.flat(Infinity):[x];
for(const c of fixture.cases){
 const cls=n4m.methodClass(c.method_id);
 const procedure=new cls(c.parameters);
 const actual=procedure.run(matrix(c.X),undefined,c.X_target?{XTarget:matrix(c.X_target)}:{});
 assert.deepEqual(Object.keys(actual).sort(),Object.keys(c.expected).sort(),c.method_id);
 for(const [key,value] of Object.entries(c.expected)){
  const raw=actual[key];
  const values=typeof raw==='number'?[raw]:raw.data?Array.from(raw.data):Array.from(raw);
  const expected=flatten(value);
  assert.equal(values.length,expected.length,`${c.method_id}/${key} shape`);
  for(let i=0;i<values.length;i++)assert.ok(Number.isFinite(values[i])&&Math.abs(values[i]-expected[i])<=1e-9*(1+Math.abs(expected[i])),`${c.method_id}/${key}/${i}: ${values[i]} != ${expected[i]}`);
 }
}
console.log(`intrinsic diagnostics: ${fixture.cases.length} real native/WASM cross-binding cases PASS`);
