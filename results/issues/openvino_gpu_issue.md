<!-- DRAFT for https://github.com/openvinotoolkit/openvino/issues -> New issue -> "Bug" template. Not posted.
     Paste each section into the matching form field. The "Relevant log output" field is auto-formatted as code:
     paste its content WITHOUT the ``` fences. Maintainers apply the "category: GPU" label. -->

**Title:** [Bug]: [GPU] GroupConvolution with weight FakeQuantize ranges of shape [C,1,1,1,KW] returns wrong values and NaN in the last output channels

### OpenVINO Version

2026.4.1 (2026.4.1-22982-e213a147257-releases/2026/4), pip wheel

### Operating System

Debian GNU/Linux 13 (trixie), kernel 6.12.107+deb13-amd64

### Device used for inference

GPU: Intel N100 integrated graphics (Alder Lake-N, device id 0x46d1)

### Framework

ONNX  <!-- dropdown; original model came via ONNX -> NNCF INT8; the reproducer builds the graph with the OpenVINO opset API -->

### Model used

Minimal synthetic model (one GroupConvolution), script below. Originally found in an NNCF-quantized MobileNetV3-small (timm `mobilenetv3_small_100`, exported to ONNX).

### Issue description

A depthwise `GroupConvolution` whose weights pass through a `FakeQuantize` with input/output ranges of shape `[C,1,1,1,KW]` gives wrong results on the GPU plugin. The ranges are per output channel and per kernel column, which is valid under FakeQuantize's numpy broadcasting.

- **GPU output:** the **last two output channels are entirely NaN** (channels 14-15 for C=16, 30-31 for C=32). The remaining channels differ from the CPU result.
- **CPU plugin:** correct on the same model.
- **Not affected by:** inference precision (f16 and f32), stride, or GPU driver. It happens identically with Debian's intel-opencl-icd 22.43.24595.41 and with Intel compute-runtime 26.35.39758.10 (IGC 2.41.5).
- **Other range shapes:** the same layer with ranges `[C,1,1,1,1]` (per output channel) or `[1,1,1,1,1]` matches CPU exactly.

This layout is not contrived. NNCF 3.4.0 produces `[C,1,1,1,KW]` ranges for depthwise convolutions imported from ONNX, where the 5-D weights come from a `Reshape` of 4-D constants (separate report to NNCF: <link>). In our MobileNetV3-small this made GPU INT8 inference collapse: mean AUC 0.62, against 0.805 for the same IR on CPU. Keeping depthwise convolutions out of quantization (`ignored_scope=IgnoredScope(types=["GroupConvolution"])`) restored GPU accuracy to the CPU value.

**Expected:** GPU output equal to CPU output (within quantization/precision tolerance) for any range shape that FakeQuantize broadcasting allows. If the plugin does not support this layout, it should reject it or fall back rather than return NaN.

**Possibly relevant:** the pattern of every channel slightly wrong plus NaN in the last channels looks like the weight-range tensor (C·KW values) being indexed as if it held C values. This is inferred from the outputs only, not checked in the plugin source.

### Step-by-step reproduction

```python
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


core = ov.Core()
print("OpenVINO", ov.__version__, "| GPU:", core.get_property("GPU", "FULL_DEVICE_NAME"))
x = rng.uniform(0, 2, (1, C, H, H)).astype(np.float32)
check("weight FQ ranges [C,1,1,1,KW]  (fails)", build((C, 1, 1, 1, KW)), x)
check("weight FQ ranges [C,1,1,1,1]   (ok)", build((C, 1, 1, 1, 1)), x)
check("weight FQ ranges [1,1,1,1,1]   (ok)", build((1, 1, 1, 1, 1)), x)
check("no weight FQ                   (ok)", build((C, 1, 1, 1, 1), weight_fq=False), x)
```

### Relevant log output

```
OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4 | GPU: Intel(R) UHD Graphics (iGPU)
weight FQ ranges [C,1,1,1,KW]  (fails)     GPU NaN channels: [14, 15]   rel. error vs CPU (finite values): 3.49e-03
weight FQ ranges [C,1,1,1,1]   (ok)        GPU NaN channels: []   rel. error vs CPU (finite values): 0.00e+00
weight FQ ranges [1,1,1,1,1]   (ok)        GPU NaN channels: []   rel. error vs CPU (finite values): 0.00e+00
no weight FQ                   (ok)        GPU NaN channels: []   rel. error vs CPU (finite values): 0.00e+00
```

Same output with the Debian 22.43 driver, where the device is reported as "Intel(R) Graphics [0x46d1] (iGPU)". In the real NNCF-quantized model, with real activations and a per-channel activation FakeQuantize in front, the non-NaN channels were off by about 5 %. Removing the activation FakeQuantize or changing stride to 1 did not change the NaN channels.

### Issue submission checklist

- [x] I'm reporting an issue. It's not a question.
- [x] I checked the problem with the documentation, FAQ, open issues, Stack Overflow, etc., and have not found a solution.
- [x] There is reproducer code and related data files such as images, videos, models, etc.
