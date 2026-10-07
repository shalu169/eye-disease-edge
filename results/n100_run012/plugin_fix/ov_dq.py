# Is it the FakeQuantize->dequantization rewrite, or the conv kernel itself?
# Feed the GPU an already-decompressed weight path (int8 const -> Convert -> Multiply scale) with
# scale shape [C,1,1,1,KW] vs [C,1,1,1,1], no FakeQuantize at all.
import numpy as np, openvino as ov, openvino.opset13 as ops
C,H=16,112; rng=np.random.default_rng(0)
x=rng.uniform(0,2,(1,C,H,H)).astype(np.float32)
def build(scale_shape, act_fq=False):
    p=ops.parameter([1,C,H,H],np.float32)
    inp=p
    if act_fq:
        lo=np.zeros((1,C,1,1),np.float32); hi=np.full((1,C,1,1),2,np.float32)
        inp=ops.fake_quantize(p,*[ops.constant(v) for v in (lo,hi,lo,hi)],256)
    q=rng.integers(-127,128,(C,1,1,3,3)).astype(np.int8)
    s=rng.uniform(0.01,0.05,scale_shape).astype(np.float32)
    w=ops.multiply(ops.convert(ops.constant(q),np.float32),ops.constant(s))
    return ov.Model([ops.group_convolution(inp,w,[2,2],[1,1],[1,1],[1,1])],[p])
core=ov.Core()
for name,sh,afq in [("dequant scale [C,1,1,1,3]",(C,1,1,1,3),False),("dequant scale [C,1,1,1,1]",(C,1,1,1,1),False),
                    ("dequant [C,1,1,1,3] + act FQ",(C,1,1,1,3),True),("dequant [C,1,1,1,1] + act FQ",(C,1,1,1,1),True)]:
    m=build(sh,afq); r={}
    for d in ("CPU","GPU"):
        r[d]=list(core.compile_model(m,d,{"INFERENCE_PRECISION_HINT":"f32"}).create_infer_request().infer({0:x}).values())[0]
    g,c=r["GPU"],r["CPU"]; fin=np.isfinite(g)
    print(f"{name:32s} NaN ch {[k for k in range(C) if not np.isfinite(g[0,k]).all()]}  rel {np.linalg.norm((g-c)[fin])/np.linalg.norm(c[fin]):.2e}")
