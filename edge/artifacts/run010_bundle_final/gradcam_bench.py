"""
Run 010 step 6: on-device cost of Grad-CAM (N100: ONNX Runtime / OpenVINO; Pi 3B: tflite-runtime).
One process per config (so peak RSS = ru_maxrss is attributable to it).

Configs (--mode):
  runtime ort_nospin = ONNX Runtime with intra-op thread spinning disabled.
  plain     : full model (image -> logits) + sigmoid. The deployment without explanations.
  head      : backbone (image -> 576x16x16 feature map) + NumPy head (probs only)
  gradcam1  : backbone + NumPy head + analytic Grad-CAM for 1 label (DR), upsampled to 512 and normalised
  gradcam5  : same for all 5 labels
Latency = batch 1 on one fixed preprocessed image (fixture .npz, so no JPEG
decode/resize is timed), --warmup + --iters timed runs. For backbone modes the
NumPy part (head + Grad-CAM) is also timed separately. Before timing, the
device output is checked against the Mac PyTorch-autograd reference stored in the
fixture (probabilities and 512x512 maps of 3 test images).
Python 3.9 / numpy 1.26 compatible.
Usage: python gradcam_bench.py --runtime ort|ov|tflite --mode plain|head|gradcam1|gradcam5 --tag TAG
           [--threads N] [--warmup 20] [--iters 200] --out results_<cfg>.json
"""
import argparse
import json
import os
import platform
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradcam_numpy import GradCAM  # noqa: E402


def make_runner(runtime, path, threads):
    """Returns (fn: x_nchw float32 [1,3,S,S] -> output np array, version string)."""
    if runtime in ("ort", "ort_nospin"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if runtime == "ort_nospin":
            # ORT worker threads busy-wait after a run by default; that steals CPU
            # from the NumPy Grad-CAM code that runs right after the backbone.
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads:
            so.intra_op_num_threads = threads
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        return (lambda x: s.run(None, {name: x})[0]), "onnxruntime " + ort.__version__
    if runtime == "ov":
        import openvino as ov
        core = ov.Core()
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = threads
        cm = core.compile_model(path, "CPU", cfg)
        req = cm.create_infer_request()
        return (lambda x: req.infer({0: x})[cm.output(0)].copy()), "openvino " + ov.__version__
    try:
        from tflite_runtime.interpreter import Interpreter
        import tflite_runtime
        ver = "tflite-runtime " + tflite_runtime.__version__
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        ver = "tensorflow " + tf.__version__
    it = Interpreter(model_path=path, num_threads=threads or 4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

    def run(x):
        it.set_tensor(i, np.ascontiguousarray(x.transpose(0, 2, 3, 1)))
        it.invoke()
        return it.get_tensor(o)
    return run, ver


def to_hwc(feat, runtime):
    return feat[0] if runtime == "tflite" else feat[0].transpose(1, 2, 0)


RSS_DIV = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0  # ru_maxrss: bytes on macOS, KiB on Linux


def stats(t):
    t = np.asarray(t) * 1000
    return {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "std": float(t.std()), "n": int(len(t))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["ort", "ort_nospin", "ov", "tflite"], required=True)
    ap.add_argument("--mode", choices=["plain", "head", "gradcam1", "gradcam5"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ext = "tflite" if a.runtime == "tflite" else "onnx"
    fx = np.load(os.path.join(a.dir, "run010_%s_fixture.npz" % a.tag))
    xs = fx["x"]  # [K,3,512,512]
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_load = time.perf_counter()
    if a.mode == "plain":
        run, ver = make_runner(a.runtime, os.path.join(a.dir, "run010_%s_full_fp32.%s" % (a.tag, ext)), a.threads)
    else:
        run, ver = make_runner(a.runtime, os.path.join(a.dir, "run010_%s_backbone_fp32.%s" % (a.tag, ext)), a.threads)
        gc = GradCAM(os.path.join(a.dir, "run010_%s_head.npz" % a.tag))
    load_s = time.perf_counter() - t_load
    labels = (0,) if a.mode == "gradcam1" else (0, 1, 2, 3, 4)

    def step(x):
        if a.mode == "plain":
            lg = run(x)[0].astype(np.float64)
            return 1 / (1 + np.exp(-lg)), None, 0.0
        f = to_hwc(run(x), a.runtime)
        t1 = time.perf_counter()
        if a.mode == "head":
            p, m = gc.predict(f), None
        else:
            p, m = gc(f, labels=labels)
        return p, m, time.perf_counter() - t1

    # parity vs the Mac torch-autograd reference
    par = {"prob_max_abs_diff": 0.0, "map_max_abs_diff": None}
    for k in range(len(xs)):
        p, m, _ = step(xs[k:k + 1])
        par["prob_max_abs_diff"] = max(par["prob_max_abs_diff"], float(np.abs(p - fx["probs"][k]).max()))
        if m is not None:
            ref = fx["maps"][k][list(labels)].astype(np.float32)
            d = float(np.abs(m - ref).max())
            par["map_max_abs_diff"] = d if par["map_max_abs_diff"] is None else max(par["map_max_abs_diff"], d)
    x = np.ascontiguousarray(xs[:1])
    for _ in range(a.warmup):
        step(x)
    tot, npt = [], []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        _, _, tn = step(x)
        tot.append(time.perf_counter() - t0)
        npt.append(tn)
    res = {"runtime": a.runtime, "runtime_version": ver, "mode": a.mode, "tag": a.tag, "threads": a.threads,
           "labels": list(labels) if a.mode.startswith("gradcam") else None,
           "warmup": a.warmup, "iters": a.iters, "latency_ms": stats(tot),
           "numpy_part_ms": stats(npt) if a.mode != "plain" else None,
           "load_s": load_s, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_DIV,
           "rss_before_load_mb": rss0 / RSS_DIV,
           "parity_vs_mac_torch": par, "numpy": np.__version__, "python": platform.python_version(),
           "machine": platform.machine(), "node": platform.node(),
           "note_maps_fixture": "reference maps stored as float16 (quantisation ~5e-4)"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
