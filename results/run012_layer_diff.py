"""
Run 012: localise where OpenVINO GPU INT8 execution diverges from CPU.

Every Convolution / GroupConvolution / Multiply / Add / HSwish / HSigmoid /
ReduceMean / MatMul / Relu / Clamp output of the run 004 INT8 IR is exposed as an
extra model output. One validation image is run on CPU and on GPU (f32
inference precision on both, so f16 rounding is not the cause), and we
report each node's relative error ||gpu - cpu|| / ||cpu|| in topological
order, plus the first node whose error exceeds 5 %.

Usage on the N100 (from ~/eye-edge, ~/neo/env.sh sourced):
  .venv/bin/python run012_layer_diff.py [--xml run012/int8_variants/base.xml] [--fp32]
"""
import argparse
import json

import numpy as np
import openvino as ov
import pandas as pd

from bench_n100 import ART, IMG_DIR, ONNX, preprocess

TYPES = {"Convolution", "GroupConvolution", "Multiply", "Add", "HSwish", "HSigmoid", "ReduceMean",
         "MatMul", "Relu", "Clamp", "Swish", "Sigmoid"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="run012/int8_variants/base.xml")
    ap.add_argument("--fp32", action="store_true", help="control: use the FP32 ONNX instead of the INT8 IR")
    ap.add_argument("--n-images", type=int, default=3)
    a = ap.parse_args()
    core = ov.Core()
    model = core.read_model(ONNX if a.fp32 else a.xml)
    model.reshape([1, 3, 224, 224])
    nodes = [op for op in model.get_ordered_ops() if op.get_type_name() in TYPES]
    model.add_outputs([op.output(0) for op in nodes])
    cfg = {"INFERENCE_PRECISION_HINT": "f32"}
    cpu = core.compile_model(model, "CPU", cfg)
    gpu = core.compile_model(model, "GPU", cfg)
    files = pd.read_csv(f"{ART}/val_manifest.csv")["filename"].tolist()[:a.n_images]
    rows = []
    for f in files:
        x = preprocess(f"{IMG_DIR}/{f}")
        rc = cpu.create_infer_request().infer({0: x})
        rg = gpu.create_infer_request().infer({0: x})
        oc = {o.get_any_name(): v for o, v in rc.items()}
        og = {o.get_any_name(): v for o, v in rg.items()}
        for i, op in enumerate(nodes):
            name = op.output(0).get_any_name()
            c, g = oc[name], og[name]
            rel = float(np.linalg.norm(g - c) / (np.linalg.norm(c) + 1e-12))
            rows.append({"image": f, "order": i, "node": op.get_friendly_name(), "type": op.get_type_name(), "rel_err": rel})
    df = pd.DataFrame(rows).groupby(["order", "node", "type"], as_index=False)["rel_err"].max()
    first = df[df.rel_err > 0.05].head(1)
    tag = "fp32" if a.fp32 else a.xml.split("/")[-1].replace(".xml", "")
    df.to_csv(f"run012/layer_diff_{tag}.csv", index=False)
    summary = {"model": tag, "n_nodes": len(df), "n_images": len(files),
               "first_node_over_5pct": first.to_dict("records"),
               "max_rel_err": float(df.rel_err.max()),
               "top10": df.sort_values("rel_err", ascending=False).head(10).to_dict("records")}
    json.dump(summary, open(f"run012/layer_diff_{tag}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:3000])
    print(df.head(25).to_string())


if __name__ == "__main__":
    main()
