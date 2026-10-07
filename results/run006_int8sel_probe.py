import os, sys, tempfile, json
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
sys.path.insert(0, "edge")
import numpy as np, pandas as pd, tensorflow as tf
from export_tflite import ART, IMG_DIR, onnx_to_saved_model, preprocess_nhwc, OUT
calib = pd.read_csv(ART/"calib_manifest.csv")["filename"].tolist()
dbg = pd.read_csv(ART/"tflite_int8_layer_debug.csv")
deny = dbg.loc[dbg["rmse/scale"] > 0.5, "tensor_name"].tolist()
print("denylisted nodes:", len(deny))
# train-image parity check: int8 vs fp32 on 100 calib (train) images
def run(path, files):
    it = tf.lite.Interpreter(model_path=str(path), num_threads=4); it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    r = []
    for f in files:
        it.set_tensor(i, preprocess_nhwc(IMG_DIR/f)); it.invoke(); r.append(1/(1+np.exp(-it.get_tensor(o)[0])))
    return np.stack(r)
pf = run(OUT["fp32"], calib[:100]); pq = run(OUT["int8"], calib[:100])
print("train(calib[:100]) int8 vs fp32: max|dp| %.3f mean|dp| %.3f" % (np.abs(pf-pq).max(), np.abs(pf-pq).mean()))
fp32_bytes = OUT["fp32"].read_bytes()
with tempfile.TemporaryDirectory() as tmp:
    sm = onnx_to_saved_model(Path(tmp)); OUT["fp32"].write_bytes(fp32_bytes)
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([preprocess_nhwc(IMG_DIR/f)] for f in calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32; conv.inference_output_type = tf.float32
    d = tf.lite.experimental.QuantizationDebugger(converter=conv,
        debug_dataset=lambda: ([preprocess_nhwc(IMG_DIR/f)] for f in calib[:64]),
        debug_options=tf.lite.experimental.QuantizationDebugOptions(denylisted_nodes=deny))
    p = Path("/tmp/run006_int8sel.tflite"); p.write_bytes(d.get_nondebug_quantized_model())
ps = run(p, calib[:100])
print("train(calib[:100]) int8sel vs fp32: max|dp| %.3f mean|dp| %.3f" % (np.abs(pf-ps).max(), np.abs(pf-ps).mean()))
