"""
Edge benchmark for an exported MobileNetV3-small ONNX model on the N100 mini PC
(run 004: run002 at 224 px; run 009+: any model / input size / manifest).

One config per process so peak RSS is attributable to that runtime alone.
Each config reports:
  - accuracy: per-label val AUC on the full 1270-image val split, plus max
    absolute probability difference vs. the PyTorch CPU reference
  - latency: batch-1 model-only latency (warmup, then N timed runs on a fixed
    input), and mean preprocessing time (JPEG decode + resize + normalize)
  - peak RSS of the process

Usage on the N100 (from ~/eye-edge):
  .venv/bin/python bench_n100.py quantize          # builds OpenVINO INT8 IR
  .venv/bin/python bench_n100.py run <config>      # one of CONFIGS below (run004 defaults)
Generalised (run 009), all optional, defaults = run 004:
  .venv/bin/python bench_n100.py run ort_fp32_cpu --onnx artifacts/run007c_eyeD512_fp32.onnx --img 512 \
      --manifest artifacts/val_manifest_run009.csv --ref artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
      --label-cols D_eye,G,C,A,H --out-dir run009/lat_512 [--n 300] [--img-dir images]
  --label-cols names the manifest column scored against each of the 5 outputs (order D,G,C,A,H).
  Probabilities are saved as probs_<config>.npy next to the JSON.
"""

import argparse
import json
import os
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


VAL_MANIFEST = f"{ART}/val_manifest.csv"
REF = f"{ART}/torch_cpu_val_probs.npy"


def add_model_args(ap):
    ap.add_argument("--onnx", default=ONNX)
    ap.add_argument("--img", type=int, default=IMG_SIZE)
    ap.add_argument("--manifest", default=VAL_MANIFEST)
    ap.add_argument("--ref", default=REF, help="PyTorch CPU reference probs (rows aligned with manifest)")
    ap.add_argument("--label-cols", default=",".join(LABEL_COLS))
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--n", type=int, default=0, help="only the first N manifest rows (0 = all)")


def configure(a):
    """Point the module-level settings (used by preprocess/make_infer, also from energy_n100) at a model."""
    global ONNX, IMG_SIZE, IMG_DIR, VAL_MANIFEST, REF, LABEL_COLS
    ONNX, IMG_SIZE, IMG_DIR, VAL_MANIFEST, REF = a.onnx, a.img, a.img_dir, a.manifest, a.ref
    LABEL_COLS = a.label_cols.split(",")


def preprocess(path):
    # matches baseline.py / run007 val_tfm: Resize((S,S)) on PIL (bilinear), ToTensor, Normalize
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


def run(config, n=0, out_dir="."):
    val = pd.read_csv(VAL_MANIFEST)
    ref = np.load(REF)
    assert len(ref) == len(val), (len(ref), len(val))
    if n:
        val, ref = val.iloc[:n], ref[:n]
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
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/probs_{config}.npy", probs)

    result = {
        "config": config,
        "runtime": runtime,
        "onnx": ONNX,
        "img": IMG_SIZE,
        "manifest": VAL_MANIFEST,
        "n_images": len(val),
        "label_cols": LABEL_COLS,
        "val_auc": aucs,
        "mean_auc": round(float(np.mean(list(aucs.values()))), 4),
        **({"D_patient_auc": round(float(roc_auc_score(val["D"].values, probs[:, 0])), 4)}
           if "D" in val.columns and LABEL_COLS[0] != "D" else {}),
        "max_abs_prob_diff_vs_torch": float(np.abs(probs - ref).max()),
        "mean_abs_prob_diff_vs_torch": float(np.abs(probs - ref).mean()),
        "latency_ms": {
            "mean": round(float(lat.mean()), 3), "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3), "p99": round(float(np.percentile(lat, 99)), 3),
            "warmup": WARMUP, "timed": TIMED, "batch": 1,
        },
        "preprocess_ms_mean": round(float(np.mean(pre_ms)), 3),
        "preprocess_ms_p50": round(float(np.percentile(pre_ms, 50)), 3),
        "model_load_s": round(load_s, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    with open(f"{out_dir}/results_{config}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "quantize":
        quantize()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["run"])
        ap.add_argument("config", choices=CONFIGS)
        add_model_args(ap)
        ap.add_argument("--out-dir", default=".")
        a = ap.parse_args()
        configure(a)
        run(a.config, a.n, a.out_dir)
