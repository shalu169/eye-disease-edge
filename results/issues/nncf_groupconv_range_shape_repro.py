"""
Standalone check: what weight-quantizer range shape does NNCF produce for a
depthwise GroupConvolution, depending on how the 5-D weights are formed?
  A: weights stored directly as a [C,1,1,3,3] Constant
  B: weights stored as [C,1,3,3] and reshaped to [C,1,1,3,3] (constant target shape)
  C: as B, but the target shape comes from a Concat (as the ONNX frontend emits
     for depthwise Conv)
Requires openvino, nncf, numpy.  python nncf_groupconv_range_shape_repro.py
"""
import nncf
import numpy as np
import openvino as ov
import openvino.opset13 as ops

C = 16
rng = np.random.default_rng(0)
W4 = rng.standard_normal((C, 1, 3, 3)).astype(np.float32)


def model(kind):
    x = ops.parameter([1, C, 56, 56], np.float32, name="x")
    if kind == "A":
        w = ops.constant(W4.reshape(C, 1, 1, 3, 3))
    elif kind == "B":
        w = ops.reshape(ops.constant(W4), ops.constant(np.array([C, 1, 1, 3, 3], np.int64)), False)
    else:
        shp = ops.concat([ops.constant(np.array([C, 1, 1], np.int64)), ops.constant(np.array([3, 3], np.int64))], 0)
        w = ops.reshape(ops.constant(W4), shp, False)
    y = ops.group_convolution(x, w, [1, 1], [1, 1], [1, 1], [1, 1])
    y = ops.relu(y)
    y = ops.convolution(y, ops.constant(rng.standard_normal((8, C, 1, 1)).astype(np.float32)), [1, 1], [0, 0], [0, 0], [1, 1])
    return ov.Model([y], [x], kind)


if __name__ == "__main__":
    print("nncf", nncf.__version__, "| openvino", ov.__version__)
    data = [rng.uniform(0, 2, (1, C, 56, 56)).astype(np.float32) for _ in range(20)]
    for kind in "ABC":
        q = nncf.quantize(model(kind), nncf.Dataset(data), subset_size=20)
        for op in q.get_ordered_ops():
            if op.get_type_name() == "GroupConvolution":
                fq = op.input_value(1).get_node()
                if fq.get_type_name() == "FakeQuantize":
                    print(f"{kind}: weight FakeQuantize input_low shape {list(fq.input_value(1).get_partial_shape().to_shape())}")
                else:
                    print(f"{kind}: weights not FakeQuantized (input is {fq.get_type_name()})")
