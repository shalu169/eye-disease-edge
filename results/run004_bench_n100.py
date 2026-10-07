"""
Edge benchmark for the run002 model on the N100 mini PC.

One config per process so peak RSS is attributable to that runtime alone.
Each config reports:
  - accuracy: per-label val AUC on the full 1270-image val split, plus max
    absolute probability difference vs. the PyTorch CPU reference
  - latency: batch-1 model-only latency (warmup, then N timed runs on a fixed
    input), and mean preprocessing time (JPEG decode + resize + normalize)
  - peak RSS of the process

Usage on the N100 (from ~/eye-edge):
  .venv/bin/python bench_n100.py quantize          # builds OpenVINO INT8 IR
  .venv/bin/python bench_n100.py run <config>      # one of CONFIGS below
"""

import json
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ART = "artifacts"
IMG_DIR = "images"
ONNX = f"{ART}/run002_mnv3s_fp32.onnx"
INT8_XML = f"{ART}/run002_mnv3s_int8.xml"
LABEL_COLS = ["D", "G", "C", "A", "H"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WARMUP = 50
TIMED = 500

CONFIGS = ["ort_fp32_cpu", "ov_fp32_cpu", "ov_int8_cpu", "ov_fp32_gpu", "ov_int8_gpu"]


def preprocess(path):
    # matches baseline.py val_tfm: Resize((224,224)) on PIL (bilinear), ToTensor, Normalize
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def quantize():
    import nncf
    import openvino as ov

    calib = pd.read_csv(f"{ART}/calib_manifest.csv")["filename"].tolist()
    model = ov.Core().read_model(ONNX)
    ds = nncf.Dataset(calib, lambda f: preprocess(f"{IMG_DIR}/{f}"))
    qmodel = nncf.quantize(model, ds, subset_size=len(calib), preset=nncf.QuantizationPreset.MIXED)
    ov.save_model(qmodel, INT8_XML)
    print(f"saved {INT8_XML} from {len(calib)} calibration images (nncf {nncf.__version__})")


def make_infer(config):
    if config == "ort_fp32_cpu":
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
        return (lambda x: sess.run(None, {"input": x})[0]), f"onnxruntime {ort.__version__}"
    import openvino as ov
    core = ov.Core()
    precision, device = config.split("_")[1], config.split("_")[2].upper()
    model = core.read_model(INT8_XML if precision == "int8" else ONNX)
    model.reshape([1, 3, IMG_SIZE, IMG_SIZE])
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()
    out = compiled.output(0)
    return (lambda x: req.infer({0: x})[out].copy()), f"openvino {ov.__version__}"


def run(config):
    val = pd.read_csv(f"{ART}/val_manifest.csv")
    ref = np.load(f"{ART}/torch_cpu_val_probs.npy")
    t0 = time.perf_counter()
    infer, runtime = make_infer(config)
    load_s = time.perf_counter() - t0

    probs, pre_ms = [], []
    for f in val["filename"]:
        t = time.perf_counter()
        x = preprocess(f"{IMG_DIR}/{f}")
        pre_ms.append((time.perf_counter() - t) * 1e3)
        probs.append(sigmoid(infer(x))[0])
    probs = np.stack(probs)
    labels = val[LABEL_COLS].values
    aucs = {c: round(float(roc_auc_score(labels[:, i], probs[:, i])), 4) for i, c in enumerate(LABEL_COLS)}

    x = preprocess(f"{IMG_DIR}/{val['filename'].iloc[0]}")
    for _ in range(WARMUP):
        infer(x)
    lat = []
    for _ in range(TIMED):
        t = time.perf_counter()
        infer(x)
        lat.append((time.perf_counter() - t) * 1e3)
    lat = np.array(lat)

    result = {
        "config": config,
        "runtime": runtime,
        "val_auc": aucs,
        "mean_auc": round(float(np.mean(list(aucs.values()))), 4),
        "max_abs_prob_diff_vs_torch": float(np.abs(probs - ref).max()),
        "mean_abs_prob_diff_vs_torch": float(np.abs(probs - ref).mean()),
        "latency_ms": {
            "mean": round(float(lat.mean()), 3), "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3), "p99": round(float(np.percentile(lat, 99)), 3),
            "warmup": WARMUP, "timed": TIMED, "batch": 1,
        },
        "preprocess_ms_mean": round(float(np.mean(pre_ms)), 3),
        "model_load_s": round(load_s, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    with open(f"results_{config}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "quantize":
        quantize()
    else:
        assert sys.argv[2] in CONFIGS, sys.argv[2]
        run(sys.argv[2])
