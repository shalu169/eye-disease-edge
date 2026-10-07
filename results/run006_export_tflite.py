"""
Run 006 (Pi 3B): convert the run002 ONNX model (the exact file benchmarked on
the N100 in run 004, md5 6dc9448de15b3bca9cbf07898a42be22) to full-model
TFLite (logits output, NHWC input), then verify every variant on the Mac
against the PyTorch CPU reference probabilities over the full val split.

Variants:
  fp32  - plain float32 TFLite
  drq   - dynamic-range quantization (int8 weights, float activations)
  int8  - full-integer PTQ (int8 weights + activations; float32 I/O kept so
          preprocessing is identical), calibrated on the 300 *train* images in
          calib_manifest.csv (never val)

Uses the onnx2tf/tensorflow venv from the fundus-app-local-mode worktree
(read-only), CPU only:
  .claude/worktrees/fundus-app-local-mode/.venv-export/bin/python edge/export_tflite.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

ART = Path("edge/artifacts")
ONNX = ART / "run002_mnv3s_fp32.onnx"
IMG_DIR = Path("data/preprocessed_images")
LABEL_COLS = ["D", "G", "C", "A", "H"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT = {k: ART / f"run002_mnv3s_{k}.tflite" for k in ["fp32", "drq", "int8"]}


def preprocess_nhwc(path):
    # same as bench_n100.preprocess (baseline.py val_tfm), but NHWC for TFLite
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x[None].astype(np.float32)


def onnx_to_saved_model(tmp: Path) -> Path:
    # onnx2tf tries to download a calibration sample that 404s; pre-seed it in
    # cwd (same workaround as the worktree's export_model.py). Not used for our
    # INT8 calibration, which uses real train images below.
    np.save(tmp / "calibration_image_sample_data_20x128x128x3_float32.npy",
            np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32), allow_pickle=False)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    sm = tmp / "saved_model"
    subprocess.run([sys.executable, "-m", "onnx2tf", "-i", str(ONNX.resolve()), "-o", str(sm),
                    "-b", "1", "-osd"], check=True, cwd=str(tmp), env=env)
    produced = list(sm.glob("*_float32.tflite"))
    assert produced, list(sm.iterdir())
    shutil.copy(produced[0], OUT["fp32"])
    return sm


def make_converter(sm: Path):
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    # TF 2.18 quantizes the dense classifier per-channel, which emits
    # FULLY_CONNECTED op version 12; tflite-runtime 2.13.0 (newest armv7/cp39
    # wheel, used on the Pi 3B) only has older FC versions and refuses to load
    # the model. Per-tensor weights for the single 1024->5 FC layer only.
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv


def quantize(sm: Path):
    import tensorflow as tf

    OUT["drq"].write_bytes(make_converter(sm).convert())

    calib = pd.read_csv(ART / "calib_manifest.csv")["filename"].tolist()
    assert len(calib) == 300
    val_files = set(pd.read_csv(ART / "val_manifest.csv")["filename"])
    assert not (set(calib) & val_files), "calibration overlaps val"

    def rep():
        for f in calib:
            yield [preprocess_nhwc(IMG_DIR / f)]

    conv = make_converter(sm)
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    OUT["int8"].write_bytes(conv.convert())


def verify(variant):
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(OUT[variant]), num_threads=4)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    val = pd.read_csv(ART / "val_manifest.csv")
    ref = np.load(ART / "torch_cpu_val_probs.npy")
    probs = []
    for f in val["filename"]:
        interp.set_tensor(inp["index"], preprocess_nhwc(IMG_DIR / f))
        interp.invoke()
        probs.append(1 / (1 + np.exp(-interp.get_tensor(out["index"])[0])))
    probs = np.stack(probs)
    labels = val[LABEL_COLS].values
    aucs = {c: round(float(roc_auc_score(labels[:, i], probs[:, i])), 4) for i, c in enumerate(LABEL_COLS)}
    np.save(ART / f"tflite_mac_{variant}_val_probs.npy", probs)
    return {
        "file": str(OUT[variant]), "bytes": OUT[variant].stat().st_size,
        "input": {"name": inp["name"], "shape": inp["shape"].tolist(), "dtype": str(inp["dtype"].__name__)},
        "output": {"name": out["name"], "shape": out["shape"].tolist(), "dtype": str(out["dtype"].__name__)},
        "mac_val_auc": aucs, "mac_mean_auc": round(float(np.mean(list(aucs.values()))), 4),
        "max_abs_prob_diff_vs_torch": float(np.abs(probs - ref).max()),
        "mean_abs_prob_diff_vs_torch": float(np.abs(probs - ref).mean()),
    }


def main():
    import onnx2tf
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp:
        sm = onnx_to_saved_model(Path(tmp))
        quantize(sm)
    summary = {"source_onnx": str(ONNX), "tensorflow": tf.__version__, "onnx2tf": onnx2tf.__version__,
               "python": sys.version.split()[0], "calibration": "edge/artifacts/calib_manifest.csv (300 train imgs)",
               "variants": {v: verify(v) for v in OUT}}
    (ART / "tflite_export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
