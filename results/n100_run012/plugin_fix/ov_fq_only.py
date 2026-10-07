# FakeQuantize alone on GPU: 5-D tensor [C,1,1,3,KW] with ranges [C,1,1,1,KW], as model input (not constant) and as constant.
import numpy as np, openvino as ov, openvino.opset13 as ops
C,KW=16,3; rng=np.random.default_rng(0)
w=rng.standard_normal((C,1,1,3,KW)).astype(np.float32)
core=ov.Core()
def run(name, rshape, as_const, rank5=True):
    shape=(C,1,1,3,KW) if rank5 else (C,1,3,KW)
    hi=np.abs(w.reshape(shape)).max(axis=tuple(i for i in range(len(shape)) if rshape[i]==1),keepdims=True)
    hi=np.broadcast_to(hi,rshape).astype(np.float32).copy(); lo=-hi
    if as_const:
        src=ops.constant(w.reshape(shape)); params=[]; feed={}
        dummy=ops.parameter([1],np.float32); params=[dummy]; feed={0:np.zeros(1,np.float32)}
        y=ops.add(ops.fake_quantize(src,*[ops.constant(v) for v in (lo,hi,lo,hi)],255), ops.constant(np.float32(0)))
        y=ops.add(y, ops.reduce_sum(dummy,ops.constant(np.array([0],np.int64)),False))
    else:
        p=ops.parameter(list(shape),np.float32); params=[p]; feed={0:w.reshape(shape)}
        y=ops.fake_quantize(p,*[ops.constant(v) for v in (lo,hi,lo,hi)],255)
    m=ov.Model([y],params); r={}
    for d in ("CPU","GPU"):
        r[d]=list(core.compile_model(m,d,{"INFERENCE_PRECISION_HINT":"f32"}).create_infer_request().infer(feed).values())[0]
    g,c=r["GPU"].reshape(C,-1),r["CPU"].reshape(C,-1)
    bad=[k for k in range(C) if not np.allclose(g[k],c[k],atol=1e-5) or not np.isfinite(g[k]).all()]
    print(f"{name:44s} wrong channels {bad}  NaN {int(np.isnan(r['GPU']).sum())}")
run("5-D input,    ranges [C,1,1,1,KW]",(C,1,1,1,KW),False)
run("5-D input,    ranges [C,1,1,1,1]",(C,1,1,1,1),False)
run("4-D input,    ranges [C,1,1,KW]",(C,1,1,KW),False,rank5=False)
run("5-D constant, ranges [C,1,1,1,KW]",(C,1,1,1,KW),True)
run("5-D constant, ranges [C,1,1,1,1]",(C,1,1,1,1),True)
