// SPDX-License-Identifier: CECILL-2.1 OR AGPL-3.0-or-later
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
const moduleUrl=process.argv[2] ?? new URL('../dist/index.js',import.meta.url).href;
const mode=process.argv[3] ?? 'all';
const doc=JSON.parse(readFileSync(process.env.N4M_SNV_FIXTURE_PATH ?? new URL('../../../parity/fixtures/snv_parameter_contract_v1.json',import.meta.url),'utf8'));
assert.deepEqual(Object.keys(doc).sort(),['default_bits','domains','matrix','provenance','references','schema','typed_values']);
assert.equal(doc.schema,'n4m.snv-parameter-contract.v1');
for(const r of [...doc.references,...doc.domains])assert.deepEqual(Object.keys(r.params).sort(),['ddof','with_mean','with_std']);
for(const r of doc.references)assert.deepEqual(Object.keys(r).sort(),['id','params','values']);
for(const r of doc.domains)assert.deepEqual(Object.keys(r).sort(),['create_status','id','params','transform_status']);
for(const r of doc.typed_values)assert.deepEqual(Object.keys(r).sort(),['admit','id','parameter','tag','value']);
const n4m=await import(moduleUrl);const rows=[];
async function check(id,fn){try{await fn();rows.push({id,status:'pass'});}catch(e){rows.push({id,status:String(e).includes('Aborted(')?'abort':'fail',error:String(e)});}}
const badVectors=[null,new Set([1]),{}, {0:1},new Float64Array(),new Float64Array([1]),'x',new Number(1),[1,0,0],[0,1,0],[1,1,1],[0,0,0]];
if(mode==='all'||mode==='host')for(let i=0;i<badVectors.length;i++)await check(`unloaded-${i}`,()=>assert.throws(()=>n4m.ppCreate('StandardNormalVariate',badVectors[i]),/SNV.*RolePipeline|RolePipeline.*SNV/));
const M=await n4m.loadModule();let calls=[];const oldCall=M.ccall,oldMalloc=M._malloc;
M.ccall=function(name,...args){calls.push(name);return oldCall.call(this,name,...args);};
M._malloc=function(size){calls.push('_malloc');return oldMalloc.call(this,size);};
const X={data:Float64Array.from(doc.matrix.flat()),rows:4,cols:4};
const noNative=()=>assert.deepEqual(calls.filter(n=>n==='n4m_method_find'||n==='n4m_params_create'||n==='n4m_params_validate'||n.startsWith('n4m_params_set_')||['n4m_estimator_create','n4m_estimator_fit','n4m_estimator_transform'].includes(n)),[]);
const close=(got,want)=>{assert.equal(got.length,want.length);for(let i=0;i<got.length;i++)assert.ok(Math.abs(got[i]-want[i])<=1e-12,`${i}: ${got[i]} != ${want[i]}`);};
const val=(r)=>{switch(r.tag){case'literal':return r.value;case'nan':return NaN;case'infinity':return Infinity;case'bigint':return BigInt(r.value);case'boxed-number':return new Number(r.value);case'boxed-boolean':return new Boolean(r.value);default:throw Error(`unconsumed JS tag ${r.tag}`);}};
async function typed(r){calls=[];const a=new n4m.SNV({[r.parameter]:val(r)});try{if(r.admit.javascript){a.fit(X);assert.equal(a.transform(X).data.length,16);}else{let failure;try{a.fit(X);}catch(e){failure=e;}assert.ok(failure,'Expected typed SNV refusal');if(String(failure).includes('Aborted('))throw failure;noNative();}}finally{a.dispose();}}
if(mode==='one'){const row=doc.typed_values.find(r=>r.id===process.argv[4]);assert.ok(row);await check(row.id,()=>typed(row));}
if(mode==='all'||mode==='host'){
 for(const r of doc.references)await check(`reference-${r.id}`,()=>{const a=new n4m.SNV(r.params);try{a.fit(X);const y=a.transform(X).data;close(y,r.values.flat());if(r.id==='mean1-std1-ddof0')assert.equal(Buffer.from(y.buffer,y.byteOffset,y.byteLength).toString('hex'),doc.default_bits.wasm_8b0);}finally{a.dispose();}});
 await check('mixed-malformed-fields',()=>{calls=[];const a=new n4m.SNV({with_mean:true,ddof:1.9});try{assert.throws(()=>a.fit(X));noNative();}finally{a.dispose();}});
 await check('counter-positive',()=>{calls=[];const a=new n4m.SNV();try{a.fit(X);a.transform(X);assert.ok(calls.includes('n4m_estimator_fit'));assert.ok(calls.includes('n4m_estimator_transform'));}finally{a.dispose();}});
 for(const r of doc.typed_values.filter(r=>r.admit.javascript!==null)){
  if(['overflow-i32','wrap-i32'].includes(r.id)){
   await check(`isolated-${r.id}`,()=>{const argv=[fileURLToPath(import.meta.url),moduleUrl,'one',r.id];const child=spawnSync(process.execPath,argv,{encoding:'utf8',env:process.env});console.log('ISOLATED',JSON.stringify({argv:[process.execPath,...argv],pid:child.pid,exit:child.status,signal:child.signal,stdout:child.stdout,stderr:child.stderr}));assert.equal(child.status,0);});
  }else await check(`typed-${r.id}`,()=>typed(r));
 }
 for(const last of [false,true])await check(`pipeline-malformed-${last?'last':'first'}`,()=>{calls=[];const bad={methodId:'preprocessing.scatter.snv',params:{ddof:1.9}};const good={methodId:'preprocessing.scaling.standard_scale'};const steps=last?[good,bad]:[bad,{methodId:'models.regularized.ridge'}];assert.throws(()=>n4m.RolePipeline.fromSteps(steps));assert.equal(calls.filter(n=>n==='n4m_role_pipeline_create').length,0);assert.equal(calls.filter(n=>n==='n4m_params_create').length,calls.filter(n=>n==='n4m_params_destroy').length);assert.equal(calls.filter(n=>n==='n4m_params_set_int').length,0);});
 await check('valid-role-pipeline',()=>{const y=Float64Array.from(doc.matrix.map(r=>r[0]+r[2]));const steps=[{methodId:'preprocessing.scatter.snv',params:{with_std:false}},{methodId:'models.regularized.ridge'}];const p=n4m.RolePipeline.fromSteps(steps).fit(X,y);const a=new n4m.SNV({with_std:false}).fit(X);const z=a.transform(X);const ridge=new n4m.Ridge().fit(z,y);try{close(p.predict(X).data,ridge.predict(z).data);}finally{p.dispose();a.dispose();ridge.dispose();}});
 for(const args of [[],undefined])await check(`default-pp-${args===undefined?'undefined':'empty'}`,()=>{const h=n4m.ppCreate('StandardNormalVariate',args);try{n4m.ppFit(h,X.data,4,4);close(n4m.ppTransform(h,X.data,4,4),doc.references[0].values.flat());}finally{n4m.ppDestroy(h);}});
}
if(mode==='all'||mode==='raw')for(const params of [[1,0,0],[0,1,0],[1,1,1],[0,0,0]])await check(`raw-${params}`,()=>{const ptr=M._malloc(24);M.HEAPF64.set(params,ptr/8);let h=0;try{h=M.ccall('n4m_wasm_pp_create','number',['string','number','number'],['StandardNormalVariate',ptr,params.length]);assert.equal(h,0);}finally{if(h)M.ccall('n4m_wasm_pp_destroy',null,['number'],[h]);M._free(ptr);}});
if(mode==='all'||mode==='extent')for(const r of doc.domains.filter(r=>r.create_status===0&&r.transform_status===1))await check(`extent-${r.id}`,()=>{const a=new n4m.SNV(r.params);try{assert.throws(()=>a.fit(X).transform(X));}finally{a.dispose();}});
let wasmSha=null;try{wasmSha=crypto.createHash('sha256').update(readFileSync(new URL('n4m.wasm',moduleUrl))).digest('hex');}catch(e){throw Error(`Cannot pin loaded WASM input: ${e}`);}
console.log(JSON.stringify({pid:process.pid,node:process.version,moduleUrl,wasmSha,mode,total:rows.length,failures:rows.filter(r=>r.status!=='pass').length,rows},null,2));
M.ccall=oldCall;M._malloc=oldMalloc;if(rows.some(r=>r.status!=='pass'))process.exitCode=1;
