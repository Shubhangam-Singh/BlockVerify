import json, subprocess, sys, numpy as np
sys.path.insert(0, '.')
import eval_lib as E

src = open('../frontend/index.html').read()
def extract(name):
    i = src.index(f'function {name}('); depth=0; started=False; j=i
    while j < len(src):
        c=src[j]
        if c=='{': depth+=1; started=True
        elif c=='}':
            depth-=1
            if started and depth==0: return src[i:j+1]
        j+=1
js = "\n".join(extract(n) for n in ["bvFlatten","bvNumericStats","bvFindOutliers","bvLayerHealth"])

tensors = E.load_safetensors('checkpoints/google_bert_uncased_L-2_H-128_A-2.safetensors')
layers = E.weight_layers(tensors)
# a real 2-D attention/FFN weight matrix, capped to a manageable real slice
name = sorted([k for k in layers if 'weight' in k and layers[k].ndim==2],
              key=lambda k: layers[k].size)[len(layers)//2]
W = layers[name][:256, :128].copy()          # real weights, ~32k elements
flat = W.ravel()
flat[123] = 87.5; flat[999] = -142.0          # inject known backdoor + extreme
nested = W.tolist()
json.dump(nested, open('/tmp/xv_tensor.json','w'))

po = E.bv_find_outliers(W); ph = E.bv_layer_health(W)

open('/tmp/xv.mjs','w').write(js + """
import fs from 'fs';
const arr = JSON.parse(fs.readFileSync('/tmp/xv_tensor.json','utf8'));
const o = bvFindOutliers(arr), h = bvLayerHealth(arr);
const fl = bvFlatten(arr); let jz=0; for(const x of fl){const z=Math.abs(x-o.median)/o.scale; if(z>jz)jz=z;}
console.log(JSON.stringify({n:o.n,count:o.count,jz,scale:o.scale,median:o.median,
  extremes:h.extremes,entropy:h.entropy,nan:h.nan,inf:h.inf}));
""")
jr = json.loads(subprocess.run(['node','/tmp/xv.mjs'],capture_output=True,text=True,check=True).stdout)

def close(a,b,tol=1e-9): return abs(a-b) <= tol*max(1,abs(a),abs(b))
checks = {
 "n": (po['n'], jr['n'], po['n']==jr['n']),
 "outlier count (z>8)": (po['count'], jr['count'], po['count']==jr['count']),
 "max robust-z": (po['max_z'], jr['jz'], close(po['max_z'],jr['jz'])),
 "scale (1.4826·MAD)": (po['scale'], jr['scale'], close(po['scale'],jr['scale'])),
 "median": (po['median'], jr['median'], close(po['median'],jr['median'])),
 "extremes |w|>100": (ph['extremes'], jr['extremes'], ph['extremes']==jr['extremes']),
 "entropy (32-bin/log2 32)": (ph['entropy'], jr['entropy'], close(ph['entropy'],jr['entropy'])),
}
print(f"real layer: {name}  slice={tuple(W.shape)}  n={po['n']}  (2 backdoor weights injected)")
allok=True
for k,(p,j,ok) in checks.items():
    allok&=ok; print(f"  {'OK  ' if ok else 'FAIL'} {k:26s} py={p:<22} js={j}")
print("\nCROSS-VALIDATION:", "PASS — Python evaluation == deployed detector" if allok else "FAIL")
sys.exit(0 if allok else 1)
