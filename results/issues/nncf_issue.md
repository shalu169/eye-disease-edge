<!-- DRAFT for https://github.com/openvinotoolkit/nncf/issues -> New issue -> "Bug report" template. Not posted. -->

**Title:** Depthwise GroupConvolution with reshaped weights gets weight quantizer ranges of shape [C,1,1,1,KW] instead of per-output-channel [C,1,1,1,1]

### Environment

- nncf 3.4.0, openvino 2026.4.1, numpy 2.4.6, Python 3.13.5
- Debian 13, x86-64 (Intel N100)

### Describe the bug

When `nncf.quantize` processes an OpenVINO model with a depthwise `GroupConvolution` whose 5-D weights `[C,1,1,KH,KW]` come from a `Reshape` of a 4-D constant `[C,1,KH,KW]`, the weight FakeQuantize ranges have shape **`[C,1,1,1,KW]`**: one range per output channel and per kernel column. This is the graph the ONNX frontend produces for a depthwise `Conv`.

When the same weights are a 5-D constant to begin with, NNCF compresses them with a per-output-channel scale of shape `[C,1,1,1,1]`, as expected.

Per-column ranges look unintended: the usual choice for convolution weights is per-output-channel. They also trigger a GPU plugin bug that produces NaN (OpenVINO issue: <link>). In our MobileNetV3-small (timm, exported to ONNX), INT8 inference on an Intel iGPU collapsed from mean AUC 0.805 (CPU) to 0.62. Excluding `GroupConvolution` via `ignored_scope` was the only NNCF-side workaround we found. Changing `preset` (MIXED/PERFORMANCE) or `target_device` (ANY/GPU) did not change the range shape or the result.

### Expected behavior

Per-output-channel ranges `[C,1,1,1,1]` for depthwise GroupConvolution weights, regardless of whether the 5-D weights are a constant or a `Reshape` of a constant. Alternatively, documentation of why per-kernel-column ranges are chosen here.

### To Reproduce

```python
import nncf
import numpy as np
import openvino as ov
import openvino.opset13 as ops

C = 16
rng = np.random.default_rng(0)
W4 = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)


def model(kind):
    x = ops.parameter([1, C, 56, 56], np.float32, name="x")
    if kind == "A":    # 5-D constant
        w = ops.constant(W4.reshape(C, 1, 1, 3, 3))
    elif kind == "B":  # 4-D constant reshaped to 5-D (constant target shape)
        w = ops.reshape(ops.constant(W4), ops.constant(np.array([C, 1, 1, 3, 3], np.int64)), False)
    else:              # as B, target shape from Concat (what the ONNX frontend emits)
        shp = ops.concat([ops.constant(np.array([C, 1, 1], np.int64)), ops.constant(np.array([3, 3], np.int64))], 0)
        w = ops.reshape(ops.constant(W4), shp, False)
    y = ops.group_convolution(x, w, [1, 1], [1, 1], [1, 1], [1, 1])
    y = ops.relu(y)
    y = ops.convolution(y, ops.constant(rng.standard_normal((8, C, 1, 1)).astype(np.float32)), [1, 1], [0, 0], [0, 0], [1, 1])
    return ov.Model([y], [x], kind)


data = [rng.uniform(0, 2, (1, C, 56, 56)).astype(np.float32) for _ in range(20)]
for kind in "ABC":
    q = nncf.quantize(model(kind), nncf.Dataset(data), subset_size=20)
    for op in q.get_ordered_ops():
        if op.get_type_name() == "GroupConvolution":
            n = op.input_value(1).get_node()
            shapes = [list(n.input_value(i).get_partial_shape().to_shape()) for i in range(n.get_input_size())]
            print(kind, n.get_type_name(), shapes)
```

### Observed output

```
A Multiply [[16, 1, 1, 3, 3], [16, 1, 1, 1, 1]]
B FakeQuantize [[16, 1, 1, 3, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3]]
C FakeQuantize [[16, 1, 1, 3, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3], [16, 1, 1, 1, 3]]
```

### Additional context

In A, NNCF stores the weights as INT8 with a per-output-channel `Multiply` scale; in B and C it inserts a `FakeQuantize` with per-column ranges.

The weights were not passed through `compress_weights`. A per-output-channel layout works correctly on both the CPU and GPU plugins in our tests.
