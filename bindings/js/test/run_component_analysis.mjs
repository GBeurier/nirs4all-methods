import assert from 'node:assert/strict';
import {loadModule,mixedEffects,fanova,normalizeLoss} from '../dist/index.js';
await loadModule();
assert.deepEqual([...normalizeLoss(new Float64Array([0,3,8]),new Float64Array([2,2,4]))],[0,1.5,2]);
assert.throws(()=>normalizeLoss(new Float64Array([1]),new Float64Array([0])));
const data=[],groups=[],y=[];
for(let g=0;g<5;g++) for(let i=0;i<4;i++){
 const x=i-1.5;data.push(1,x);groups.push(BigInt(g));y.push(3+.75*x+g-2+[.1,-.1,-.1,.1][i]);
}
const matrix=(values,rows,cols)=>({data:new Float64Array(values),rows,cols});
const lmm=mixedEffects(matrix(data,20,2),matrix(y,20,1),{data:new BigInt64Array(groups),rows:20,cols:1});
if(lmm.reason==='mixed_effects_backend_not_compiled') assert.equal(lmm.state,'unavailable');
else{
 assert.equal(lmm.state,'ready',JSON.stringify(lmm));
 assert.ok(Math.abs(lmm.beta[0]-3)<1e-7);assert.ok(Math.abs(lmm.beta[1]-.75)<1e-7);
 assert.ok(lmm.standard_errors[1]>0);assert.equal(lmm.variances.length,2);
}
const xx=[],yy=[];for(let repeat=0;repeat<16;repeat++) for(const [a,b] of [[0,0],[0,1],[1,0],[1,1]]){xx.push(a,b);yy.push(a+2*b);}
const bounds=matrix([0,1,0,1],2,2),X=matrix(xx,64,2),types=new Uint32Array([2,2]);
const f=fanova(X,matrix(yy,64,1),types,bounds,{n_trees:12,min_samples_leaf:1,max_depth:4});
if(f.reason==='fanova_backend_not_compiled') assert.equal(f.state,'unavailable');
else{
 assert.equal(f.state,'ready',JSON.stringify(f));
 assert.ok(Math.abs(f.feature_importance[0].fraction-.2)<.025,JSON.stringify(f));
 assert.ok(Math.abs(f.feature_importance[1].fraction-.8)<.025);
 const interaction=fanova(X,matrix(Array.from({length:64},(_,i)=>(2*xx[2*i]-1)*(2*xx[2*i+1]-1)),64,1),types,bounds,{n_trees:12,min_samples_leaf:1,max_depth:4});
 assert.equal(interaction.state,'ready',JSON.stringify(interaction));assert.ok(interaction.interactions[0].fraction>.95);
 const constant=fanova(X,matrix(Array(64).fill(1),64,1),types,bounds);
 assert.equal(constant.state,'unavailable');assert.equal(constant.reason,'zero_surrogate_variance');
 // Unset max_depth/seed follow the native n4m_fanova_options_init_v1 defaults; explicit values are preserved.
 assert.equal(constant.options.max_depth,8);assert.equal(constant.options.seed,0);
 assert.equal(f.options.max_depth,4);
 assert.equal(JSON.stringify(constant).includes('NaN'),false);
}
console.log('component analysis: native normalization, mixed effects and fANOVA bindings passed');
