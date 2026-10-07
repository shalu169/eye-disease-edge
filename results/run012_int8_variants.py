"""
Run 012: why does OpenVINO INT8 collapse on the N100 iGPU (run 004: AUC 0.62)
while the same IR is fine on the CPU (0.805)?

The driver was ruled out (Intel compute-runtime 26.35 gives the identical
result as Debian's 22.43). Here we re-quantize the 224 px run002 model with
NNCF variants and run each on CPU and GPU:
  base       preset MIXED, target_device ANY   (= run 004 IR, rebuilt)
  gpu        preset MIXED, target_device GPU
  perf       preset PERFORMANCE (symmetric activations), target_device ANY
  perf_gpu   preset PERFORMANCE, target_device GPU
Same 300 calibration images as run 004, same 1270-image validation set.

Usage on the N100 (from ~/eye-edge, with ~/neo/env.sh sourced for the new driver):
  .venv/bin/python run012_int8_variants.py                 # NNCF preset / target-device variants
  .venv/bin/python run012_int8_variants.py --workarounds   # per-tensor activations, skip depthwise conv(s)
"""
import json
import time

import nncf
import numpy as np
import openvino as ov
import pandas as pd
from sklearn.metrics import roc_auc_score

from bench_n100 import ART, IMG_DIR, LABEL_COLS, ONNX, preprocess, sigmoid

OUT = "run012/int8_variants"
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters, QuantizationParameters  # noqa: E402

# workarounds added after the layer diff localised NaNs to the GPU INT8 kernel of the first
# depthwise convolution (output channels 14-15 of 16):
FIRST_DW = "/blocks/blocks.0/blocks.0.0/conv_dw/Conv/WithoutBiases"
WORKAROUNDS = {
    "pertensor_act": dict(preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.ANY,
                          advanced_parameters=AdvancedQuantizationParameters(
                              activations_quantization_params=QuantizationParameters(per_channel=False))),
    "skip_first_dw": dict(preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.ANY,
                          ignored_scope=nncf.IgnoredScope(names=[FIRST_DW])),
    "skip_all_dw": dict(preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.ANY,
                        ignored_scope=nncf.IgnoredScope(types=["GroupConvolution"])),
}

VARIANTS = {
    "base": dict(preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.ANY),
    "gpu": dict(preset=nncf.QuantizationPreset.MIXED, target_device=nncf.TargetDevice.GPU),
    "perf": dict(preset=nncf.QuantizationPreset.PERFORMANCE, target_device=nncf.TargetDevice.ANY),
    "perf_gpu": dict(preset=nncf.QuantizationPreset.PERFORMANCE, target_device=nncf.TargetDevice.GPU),
}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    core = ov.Core()
    calib = pd.read_csv(f"{ART}/calib_manifest.csv")["filename"].tolist()
    val = pd.read_csv(f"{ART}/val_manifest.csv")
    ref = np.load(f"{ART}/torch_cpu_val_probs.npy")
    xs = [preprocess(f"{IMG_DIR}/{f}") for f in val["filename"]]
    labels = val[LABEL_COLS].values
    results = {"gpu": core.get_property("GPU", "FULL_DEVICE_NAME"), "openvino": ov.__version__,
               "nncf": nncf.__version__, "variants": {}}
    import sys
    todo = WORKAROUNDS if "--workarounds" in sys.argv else VARIANTS
    out_json = f"{OUT}/results_workarounds.json" if todo is WORKAROUNDS else f"{OUT}/results.json"
    for name, kw in todo.items():
        model = core.read_model(ONNX)
        ds = nncf.Dataset(calib, lambda f: preprocess(f"{IMG_DIR}/{f}"))
        q = nncf.quantize(model, ds, subset_size=len(calib), **kw)
        q.reshape([1, 3, 224, 224])
        ov.save_model(q, f"{OUT}/{name}.xml")
        results["variants"][name] = {"nncf_kwargs": {k: str(v) for k, v in kw.items()}}
        for dev in ("CPU", "GPU"):
            comp = core.compile_model(q, dev, {"PERFORMANCE_HINT": "LATENCY"})
            req, out = comp.create_infer_request(), comp.output(0)
            probs = np.stack([sigmoid(req.infer({0: x})[out])[0] for x in xs])
            for _ in range(50):
                req.infer({0: xs[0]})
            lat = []
            for _ in range(500):
                t = time.perf_counter()
                req.infer({0: xs[0]})
                lat.append((time.perf_counter() - t) * 1e3)
            aucs = {c: round(float(roc_auc_score(labels[:, i], probs[:, i])), 4) for i, c in enumerate(LABEL_COLS)}
            results["variants"][name][dev] = {
                "val_auc": aucs, "mean_auc": round(float(np.mean(list(aucs.values()))), 4),
                "max_abs_prob_diff_vs_torch": float(np.abs(probs - ref).max()),
                "mean_abs_prob_diff_vs_torch": float(np.abs(probs - ref).mean()),
                "latency_ms_mean": round(float(np.mean(lat)), 3), "latency_ms_p95": round(float(np.percentile(lat, 95)), 3),
                "inference_precision": str(comp.get_property("INFERENCE_PRECISION_HINT"))}
            print(name, dev, results["variants"][name][dev]["mean_auc"], results["variants"][name][dev]["latency_ms_mean"], flush=True)
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
