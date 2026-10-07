"""
Run 006 diagnosis: why does full-integer TFLite INT8 lose so much AUC?
Rebuilds the onnx2tf SavedModel, runs tf.lite QuantizationDebugger on train
calibration images only, and writes per-layer error stats to
edge/artifacts/tflite_int8_layer_debug.csv (sorted by rmse/scale).

Then builds a selective-quantization variant (int8mixed): identical full-int8
PTQ (same 300 train calibration images) except all MEAN ops (the SE-block
global average pools, the worst layers in the debug dump) are left in float.
Verified on the Mac against the torch reference like the other variants;
results merged into edge/artifacts/tflite_export_summary.json.

  .claude/worktrees/fundus-app-local-mode/.venv-export/bin/python edge/diagnose_tflite_int8.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import tensorflow as tf

import json

from export_tflite import ART, IMG_DIR, make_converter, onnx_to_saved_model, preprocess_nhwc, OUT, verify

N_DEBUG = 64


def main():
    calib = pd.read_csv(ART / "calib_manifest.csv")["filename"].tolist()

    def rep():
        for f in calib:
            yield [preprocess_nhwc(IMG_DIR / f)]

    def rep_small():
        for f in calib[:N_DEBUG]:
            yield [preprocess_nhwc(IMG_DIR / f)]

    saved_fp32 = OUT["fp32"].read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
        sm = onnx_to_saved_model(Path(tmp))
        OUT["fp32"].write_bytes(saved_fp32)  # keep the already-verified file byte-identical
        conv = make_converter(sm)
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
        dbg = tf.lite.experimental.QuantizationDebugger(converter=conv, debug_dataset=rep_small)
        dbg.run()
        opts = tf.lite.experimental.QuantizationDebugOptions(denylisted_ops=["MEAN"])
        dbg_mixed = tf.lite.experimental.QuantizationDebugger(
            converter=conv, debug_dataset=rep_small, debug_options=opts)
        OUT["int8mixed"] = ART / "run002_mnv3s_int8mixed.tflite"
        OUT["int8mixed"].write_bytes(dbg_mixed.get_nondebug_quantized_model())
        out_csv = ART / "tflite_int8_layer_debug.csv"
        with open(out_csv, "w") as fh:
            dbg.layer_statistics_dump(fh)
    df = pd.read_csv(out_csv)
    df["range"] = df["max"] - df["min"] if "max" in df else np.nan
    df["rmse/scale"] = np.sqrt(df["mean_squared_error"]) / df["scale"]
    df = df.sort_values("rmse/scale", ascending=False)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 250)
    summ_path = ART / "tflite_export_summary.json"
    summ = json.loads(summ_path.read_text())
    summ["variants"]["int8mixed"] = verify("int8mixed")
    summ["variants"]["int8mixed"]["note"] = "full-int8 PTQ, MEAN ops (SE global avg pool) denylisted -> float"
    summ_path.write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ["variants"]["int8mixed"], indent=2))
    print(df[["op_name", "tensor_name", "num_elements", "rmse/scale", "mean_squared_error", "scale"]].head(25).to_string())


if __name__ == "__main__":
    main()
