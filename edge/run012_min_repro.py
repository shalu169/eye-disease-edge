"""
Run 012: minimal single-layer reproduction of the OpenVINO GPU INT8 NaN.

Rebuilds the failing layer of the NNCF INT8 IR in isolation:
  Parameter [1,C,112,112] -> FakeQuantize(levels 256, per-channel ranges [1,C,1,1])
  weights [C,1,1,3,3] -> FakeQuantize(levels 255, ranges R) -> GroupConvolution(stride 2, pad 1)
with the original constants from run012/int8_variants/base.xml, and compares CPU
vs GPU. Variants change one factor at a time:
  orig        weight FQ ranges exactly as NNCF produced them, shape [C,1,1,1,3]
  perchannel  weight ranges collapsed to per-output-channel [C,1,1,1,1] (max |range| over kW)
  pertensor   one weight range for the whole tensor
  noweightfq  weight FakeQuantize removed (activation FQ kept)
  noactfq     activation FakeQuantize removed (weight FQ kept, original shape)
  stride1     orig, but stride 1
  c32         orig layer tiled to 32 channels
Usage on the N100 (from ~/eye-edge): .venv/bin/python run012_min_repro.py
"""
import json

import numpy as np
import openvino as ov
import openvino.opset13 as ops

DW = "/blocks/blocks.0/blocks.0.0/conv_dw/Conv/WithoutBiases"


def consts():
    m = ov.Core().read_model("run012/int8_variants/base.xml")
    dw = [o for o in m.get_ordered_ops() if o.get_friendly_name() == DW][0]
    afq, wfq = dw.input_value(0).get_node(), dw.input_value(1).get_node()
    a = [np.array(afq.input_value(k).get_node().get_data()) for k in range(1, 5)]
    w_node = wfq.input_value(0).get_node()  # Reshape(Constant, Concat) -> fold by evaluating on CPU
    sub = ov.Model([w_node.output(0)], [], "w")
    w = list(ov.Core().compile_model(sub, "CPU").create_infer_request().infer({}).values())[0]
    r = [np.array(wfq.input_value(k).get_node().get_data()) for k in range(1, 5)]
    return a, w.astype(np.float32), r


def build(a, w, r, act_fq=True, w_fq=True, stride=2):
    c = w.shape[0]
    x = ops.parameter([1, c, 112, 112], np.float32, name="x")
    h = ops.fake_quantize(x, *[ops.constant(v) for v in a], 256) if act_fq else x
    wt = ops.constant(w)
    if w_fq:
        wt = ops.fake_quantize(wt, *[ops.constant(v) for v in r], 255)
    y = ops.group_convolution(h, wt, [stride, stride], [1, 1], [1, 1], [1, 1])
    return ov.Model([y], [x], "dw")


def main():
    core = ov.Core()
    a, w, r = consts()
    rng = np.random.default_rng(0)
    # realistic input: post-hard-swish activations within each channel's quantizer range
    lo, hi = a[0].reshape(1, -1, 1, 1), a[1].reshape(1, -1, 1, 1)
    x16 = (lo + (hi - lo) * rng.random((1, 16, 112, 112))).astype(np.float32)
    rmax = [np.repeat(np.abs(v).max(axis=-1, keepdims=True), 1, -1) * np.sign(v[..., :1]) for v in r]
    rmax = [np.where(np.sign(v) == 0, 0, v) for v in rmax]
    rt = [np.full((1, 1, 1, 1, 1), np.sign(v.ravel()[0]) * np.abs(v).max(), np.float32) for v in r]
    variants = {
        "orig": (a, w, r, True, True, 2),
        "perchannel": (a, w, [v.astype(np.float32) for v in rmax], True, True, 2),
        "pertensor": (a, w, rt, True, True, 2),
        "noweightfq": (a, w, r, True, False, 2),
        "noactfq": (a, w, r, False, True, 2),
        "stride1": (a, w, r, True, True, 1),
        "c32": ([np.concatenate([v, v], 1) for v in a], np.concatenate([w, w], 0),
                [np.concatenate([v, v], 0) for v in r], True, True, 2),
    }
    out = {"weight_fq_range_shape": list(r[0].shape), "act_fq_range_shape": list(a[0].shape),
           "openvino": ov.__version__, "gpu": core.get_property("GPU", "FULL_DEVICE_NAME"), "variants": {}}
    for name, (aa, ww, rr, afq, wfq, st) in variants.items():
        model = build(aa, ww, rr, afq, wfq, st)
        x = np.concatenate([x16, x16], 1) if name == "c32" else x16
        res = {}
        for dev in ("CPU", "GPU"):
            y = list(core.compile_model(model, dev, {"INFERENCE_PRECISION_HINT": "f32"})
                     .create_infer_request().infer({0: x}).values())[0]
            res[dev] = y
        g, c = res["GPU"], res["CPU"]
        nan_ch = [int(k) for k in range(g.shape[1]) if not np.isfinite(g[0, k]).all()]
        fin = np.isfinite(g)
        rel = float(np.linalg.norm((g - c)[fin]) / (np.linalg.norm(c[fin]) + 1e-12))
        out["variants"][name] = {"gpu_nan_channels": nan_ch, "cpu_all_finite": bool(np.isfinite(c).all()),
                                 "rel_err_on_finite": rel}
        print(f"{name:11s} GPU NaN channels {nan_ch}  CPU finite {np.isfinite(c).all()}  rel err (finite) {rel:.2e}")
    json.dump(out, open("run012/min_repro.json", "w"), indent=1)
    print("weight FQ range shape", r[0].shape, "activation FQ range shape", a[0].shape)


if __name__ == "__main__":
    main()
