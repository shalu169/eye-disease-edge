"""
Standalone reproducer: OpenVINO GPU plugin returns wrong results / NaN for a
GroupConvolution (depthwise) whose weight FakeQuantize ranges vary along the
kernel-width axis (range shape [C,1,1,1,KW]). CPU plugin is correct.

Requires only openvino and numpy. Synthetic random data, no model files.
  python ov_gpu_groupconv_fq_repro.py
"""
import numpy as np
import openvino as ov
import openvino.opset13 as ops

C, H, KW = 16, 112, 3
rng = np.random.default_rng(0)


def build(range_shape, weight_fq=True):
    x = ops.parameter([1, C, H, H], np.float32, name="x")
    w = rng.standard_normal((C, 1, 1, 3, KW)).astype(np.float32)
    wt = ops.constant(w)
    if weight_fq:
        hi = np.abs(w).max(axis=tuple(i for i in range(5) if range_shape[i] == 1), keepdims=True)
        hi = np.broadcast_to(hi, range_shape).astype(np.float32).copy()
        lo = -hi
        wt = ops.fake_quantize(wt, ops.constant(lo), ops.constant(hi), ops.constant(lo), ops.constant(hi), 255)
    y = ops.group_convolution(x, wt, [2, 2], [1, 1], [1, 1], [1, 1])
    return ov.Model([y], [x], "dw")


def check(name, model, x):
    core = ov.Core()
    out = {}
    for dev in ("CPU", "GPU"):
        cm = core.compile_model(model, dev, {"INFERENCE_PRECISION_HINT": "f32"})
        out[dev] = list(cm.create_infer_request().infer({0: x}).values())[0]
    g, c = out["GPU"], out["CPU"]
    nan_ch = [k for k in range(C) if not np.isfinite(g[0, k]).all()]
    fin = np.isfinite(g)
    rel = np.linalg.norm((g - c)[fin]) / np.linalg.norm(c[fin])
    print(f"{name:42s} GPU NaN channels: {nan_ch}   rel. error vs CPU (finite values): {rel:.2e}")


if __name__ == "__main__":
    core = ov.Core()
    print("OpenVINO", ov.__version__, "| GPU:", core.get_property("GPU", "FULL_DEVICE_NAME"))
    x = rng.uniform(0, 2, (1, C, H, H)).astype(np.float32)
    check("weight FQ ranges [C,1,1,1,KW]  (fails)", build((C, 1, 1, 1, KW)), x)
    check("weight FQ ranges [C,1,1,1,1]   (ok)", build((C, 1, 1, 1, 1)), x)
    check("weight FQ ranges [1,1,1,1,1]   (ok)", build((1, 1, 1, 1, 1)), x)
    check("no weight FQ                   (ok)", build((C, 1, 1, 1, 1), weight_fq=False), x)
