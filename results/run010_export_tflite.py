"""
Run 010 step 2b: convert the run010 backbone / full ONNX files to FP32 TFLite
with onnx2tf (same toolchain as run 006: the fundus-app-local-mode worktree's
.venv-export, used read-only, CPU only). NHWC input [1,512,512,3]; backbone
output NHWC [1,16,16,576] (onnx2tf transposes), full output logits [1,5].
Usage: .claude/worktrees/fundus-app-local-mode/.venv-export/bin/python results/run010_export_tflite.py --tag TAG
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

ART = Path("edge/artifacts")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def convert(onnx_path, out_path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "calibration_image_sample_data_20x128x128x3_float32.npy",
                np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32), allow_pickle=False)
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        sm = tmp / "saved_model"
        subprocess.run([sys.executable, "-m", "onnx2tf", "-i", str(Path(onnx_path).resolve()), "-o", str(sm),
                        "-b", "1", "-osd"], check=True, cwd=str(tmp), env=env,
                       stdout=subprocess.DEVNULL)
        produced = list(sm.glob("*_float32.tflite"))
        assert produced, list(sm.iterdir())
        shutil.copy(produced[0], out_path)


def main():
    import onnx2tf
    import tensorflow as tf

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = {"tag": a.tag, "tensorflow": tf.__version__, "onnx2tf": onnx2tf.__version__, "files": {}}
    for name in ["backbone", "full"]:
        src = ART / f"run010_{a.tag}_{name}_fp32.onnx"
        dst = ART / f"run010_{a.tag}_{name}_fp32.tflite"
        convert(src, dst)
        it = tf.lite.Interpreter(model_path=str(dst))
        it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        res["files"][name] = {"source_onnx": str(src), "path": str(dst), "md5": md5(dst), "bytes": dst.stat().st_size,
                              "input_shape": i["shape"].tolist(), "output_shape": o["shape"].tolist()}
    json.dump(res, open(ART / f"run010_{a.tag}_tflite_export.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
