"""
Run 006: edge benchmark for the run002 model on the Raspberry Pi 3B (TFLite).

Mirrors bench_n100.py (run 004) so numbers are directly comparable; one config
per process so peak RSS is attributable to that runtime alone. Each config
reports:
  - accuracy: per-label val AUC on the full 1270-image val split, plus max /
    mean absolute probability difference vs. the PyTorch CPU reference
  - latency: batch-1 model-only latency (WARMUP, then TIMED runs on a fixed
    input), and mean preprocessing time (JPEG decode + resize + normalize)
  - model load time, peak RSS, threads, vcgencmd temp/throttle/clock
    before and after

Differences vs. bench_n100.py, all forced by the Pi:
  - fewer timed iterations (20 warmup + 200 timed instead of 50 + 500)
  - no pandas/sklearn on armv7: CSV via the stdlib, AUC via the rank
    (Mann-Whitney) formula with average ranks for ties, which equals
    sklearn.roc_auc_score; probabilities are also saved so AUCs can be
    recomputed with sklearn off-device
  - TFLite input is NHWC (onnx2tf conversion), same preprocessing otherwise

Usage on the Pi (from ~/eye-edge):
  .venv/bin/python bench_pi.py <config>      # e.g. tfl_fp32_t4 (run 006 defaults)
Generalised (run 009), all optional, defaults = run 006:
  .venv/bin/python bench_pi.py tfl_fp32_t4 --model artifacts/run007c_eyeD512_fp32.tflite --img 512 \
      --manifest artifacts/val_manifest_run009.csv --ref artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
      --label-cols D_eye,G,C,A,H --out-dir run009/lat_512 [--n 300] [--img-dir images]
  --model overrides the file for the config's precision; --n limits the accuracy pass to the
  first N manifest rows (the latency protocol is unchanged).
"""

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ART = "artifacts"
IMG_DIR = "images"
LABEL_COLS = ["D", "G", "C", "A", "H"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WARMUP = 20
TIMED = 200

MODELS = {"fp32": "run002_mnv3s_fp32.tflite", "drq": "run002_mnv3s_drq.tflite",
          "int8": "run002_mnv3s_int8.tflite"}
CONFIGS = [f"tfl_{m}_t{t}" for m in MODELS for t in (4, 1)]


VAL_MANIFEST = f"{ART}/val_manifest.csv"
REF = f"{ART}/torch_cpu_val_probs.npy"


def preprocess(path):
    # matches baseline.py / run007 val_tfm: Resize((S,S)) on PIL (bilinear), ToTensor, Normalize; NHWC
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x[None].astype(np.float32)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, s):
    # rank-based ROC AUC with average ranks for ties (== sklearn.roc_auc_score)
    y = np.asarray(y).astype(bool)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def vcgencmd():
    out = {}
    for k, args in {"temp": ["measure_temp"], "throttled": ["get_throttled"],
                    "arm_clock": ["measure_clock", "arm"]}.items():
        try:
            out[k] = subprocess.run(["vcgencmd"] + args, capture_output=True, text=True,
                                    timeout=10).stdout.strip()
        except Exception as e:  # never let telemetry kill a run
            out[k] = f"error: {e}"
    return out


def mem_available_mb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024, 1)


def run(config, model_file=None, n=0, out_dir="."):
    _, precision, t = config.split("_")
    threads = int(t[1:])
    model_file = model_file or f"{ART}/{MODELS[precision]}"
    vc_before, mem_before = vcgencmd(), mem_available_mb()

    with open(VAL_MANIFEST) as fh:
        rows = list(csv.DictReader(fh))
    ref = np.load(REF)
    assert len(ref) == len(rows), (len(ref), len(rows))
    if n:
        rows, ref = rows[:n], ref[:n]
    files = [r["filename"] for r in rows]
    labels = np.array([[int(r[c]) for c in LABEL_COLS] for r in rows])

    t0 = time.perf_counter()
    import tflite_runtime
    from tflite_runtime.interpreter import Interpreter
    interp = Interpreter(model_path=model_file, num_threads=threads)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def infer(x):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"]).copy()

    load_s = time.perf_counter() - t0

    probs, pre_ms = [], []
    t_acc = time.perf_counter()
    for f in files:
        t = time.perf_counter()
        x = preprocess(f"{IMG_DIR}/{f}")
        pre_ms.append((time.perf_counter() - t) * 1e3)
        probs.append(sigmoid(infer(x))[0])
    acc_pass_s = time.perf_counter() - t_acc
    probs = np.stack(probs)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/probs_{config}.npy", probs)
    aucs = {c: round(auc(labels[:, i], probs[:, i]), 4) for i, c in enumerate(LABEL_COLS)}
    extra = {}
    if "D" in rows[0] and LABEL_COLS[0] != "D":
        extra["D_patient_auc"] = round(auc(np.array([int(r["D"]) for r in rows]), probs[:, 0]), 4)

    x = preprocess(f"{IMG_DIR}/{files[0]}")
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
        "runtime": f"tflite_runtime {tflite_runtime.__version__}",
        "model_file": model_file,
        "img": IMG_SIZE,
        "manifest": VAL_MANIFEST,
        "n_images": len(rows),
        "label_cols": LABEL_COLS,
        "num_threads": threads,
        "val_auc": aucs,
        "mean_auc": round(float(np.mean(list(aucs.values()))), 4),
        **extra,
        "max_abs_prob_diff_vs_torch": float(np.abs(probs - ref).max()),
        "mean_abs_prob_diff_vs_torch": float(np.abs(probs - ref).mean()),
        "latency_ms": {
            "mean": round(float(lat.mean()), 3), "p50": round(float(np.percentile(lat, 50)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3), "p99": round(float(np.percentile(lat, 99)), 3),
            "warmup": WARMUP, "timed": TIMED, "batch": 1,
        },
        "preprocess_ms_mean": round(float(np.mean(pre_ms)), 3),
        "preprocess_ms_p50": round(float(np.percentile(pre_ms, 50)), 3),
        "accuracy_pass_s": round(acc_pass_s, 1),
        "model_load_s": round(load_s, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "mem_available_mb_before": mem_before,
        "mem_available_mb_after": mem_available_mb(),
        "vcgencmd_before": vc_before,
        "vcgencmd_after": vcgencmd(),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
    }
    with open(f"{out_dir}/results_{config}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=CONFIGS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--img", type=int, default=IMG_SIZE)
    ap.add_argument("--manifest", default=VAL_MANIFEST)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--label-cols", default=",".join(LABEL_COLS))
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    IMG_SIZE, VAL_MANIFEST, REF, IMG_DIR = a.img, a.manifest, a.ref, a.img_dir
    LABEL_COLS = a.label_cols.split(",")
    run(a.config, a.model, a.n, a.out_dir)
