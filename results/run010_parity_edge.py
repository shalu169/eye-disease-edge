"""
Run 010 step 2c: parity of the edge Grad-CAM path vs the PyTorch autograd reference.

For N seeded-random TEST images, all 5 labels:
  reference  : results/run010_common.gradcam_torch (autograd w.r.t. feature map),
               display map = max-normalised torch-bilinear upsample to 512
  edge       : backbone feature map from the runtime (ONNX Runtime or TFLite,
               FP32) -> edge/gradcam_numpy.GradCAM (NumPy head + analytic
               gradient + NumPy bilinear upsample)
  head-only  : edge/gradcam_numpy on the *torch* feature map (isolates the
               NumPy head/Grad-CAM maths from backbone-runtime differences)
  plain      : the runtime's full model (logits) vs torch probabilities.
Usage:
  ONNX  : .venv-run010/bin/python results/run010_parity_edge.py --runtime onnx --tag TAG --ckpt auto
  TFLite: .claude/worktrees/fundus-app-local-mode/.venv-export/bin/python results/run010_parity_edge.py --runtime tflite --tag TAG --ckpt auto
"""
import argparse
import json
import platform
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "results")
sys.path.insert(0, "edge")
from gradcam_numpy import GradCAM  # noqa: E402
from run010_common import (DATA_DIR, OUT, ckpt_from_args, gradcam_torch, load_manifest, load_model,  # noqa: E402
                           load_x, norm_max, upsample)

ART = "edge/artifacts"


class Runtime:
    def __init__(self, kind, tag):
        self.kind = kind
        if kind == "onnx":
            import onnxruntime as ort
            self.version = ort.__version__
            self.bb = ort.InferenceSession(f"{ART}/run010_{tag}_backbone_fp32.onnx", providers=["CPUExecutionProvider"])
            self.full = ort.InferenceSession(f"{ART}/run010_{tag}_full_fp32.onnx", providers=["CPUExecutionProvider"])
        else:
            import tensorflow as tf
            self.version = "tensorflow " + tf.__version__
            self.bb = tf.lite.Interpreter(model_path=f"{ART}/run010_{tag}_backbone_fp32.tflite", num_threads=4)
            self.full = tf.lite.Interpreter(model_path=f"{ART}/run010_{tag}_full_fp32.tflite", num_threads=4)
            for it in (self.bb, self.full):
                it.allocate_tensors()

    def _tfl(self, it, x_nhwc):
        it.set_tensor(it.get_input_details()[0]["index"], x_nhwc)
        it.invoke()
        return it.get_tensor(it.get_output_details()[0]["index"])

    def features_hwc(self, x_chw):
        if self.kind == "onnx":
            return self.bb.run(None, {"image": x_chw[None]})[0][0].transpose(1, 2, 0)
        return self._tfl(self.bb, x_chw.transpose(1, 2, 0)[None])[0]

    def probs_full(self, x_chw):
        if self.kind == "onnx":
            lg = self.full.run(None, {"image": x_chw[None]})[0][0]
        else:
            lg = self._tfl(self.full, x_chw.transpose(1, 2, 0)[None])[0]
        return 1 / (1 + np.exp(-lg.astype(np.float64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["onnx", "tflite"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ckpt, info = ckpt_from_args(a.ckpt)
    model = load_model(ckpt)
    rt = Runtime(a.runtime, a.tag)
    gc = GradCAM(f"{ART}/run010_{a.tag}_head.npz")
    files = load_manifest("test").sample(n=a.n, random_state=10)["filename"].tolist()
    acc = {k: [] for k in ["feat_maxabs", "feat_relmax", "p_edge", "p_headonly", "p_plain",
                           "map_edge", "map_headonly", "lr_edge_rel", "rho_edge", "argmax_same"]}
    t0 = time.time()
    for f in files:
        x = load_x(f"{DATA_DIR}/{f}")
        p_ref, lr_ref, A = gradcam_torch(model, torch.from_numpy(x)[None])
        p_ref, lr_ref = p_ref[0].astype(np.float64), lr_ref[0]
        disp_ref = norm_max(upsample(lr_ref))
        A_hwc = A[0].transpose(1, 2, 0)
        F = rt.features_hwc(x)
        acc["feat_maxabs"].append(np.abs(F - A_hwc).max())
        acc["feat_relmax"].append(np.abs(F - A_hwc).max() / np.abs(A_hwc).max())
        p_e, m_e = gc(F)
        p_h, m_h = gc(A_hwc)
        _, lr_e = gc(F, upsample=False)
        acc["p_edge"].append(np.abs(p_e - p_ref).max())
        acc["p_headonly"].append(np.abs(p_h - p_ref).max())
        acc["p_plain"].append(np.abs(rt.probs_full(x) - p_ref).max())
        acc["map_edge"].append(np.abs(m_e - disp_ref).max(axis=(1, 2)))
        acc["map_headonly"].append(np.abs(m_h - disp_ref).max(axis=(1, 2)))
        acc["lr_edge_rel"].append(np.abs(lr_e - lr_ref).max(axis=(1, 2)) / np.maximum(lr_ref.max(axis=(1, 2)), 1e-12))
        for c in range(5):
            if disp_ref[c].max() > 0:
                acc["rho_edge"].append(spearmanr(m_e[c][::4, ::4].ravel(), disp_ref[c][::4, ::4].ravel())[0])
                acc["argmax_same"].append(int(np.argmax(m_e[c]) == np.argmax(disp_ref[c])))
    me, mh = np.concatenate(acc["map_edge"]), np.concatenate(acc["map_headonly"])
    res = {
        "runtime": a.runtime, "runtime_version": acc and rt.version, "tag": a.tag, "checkpoint": ckpt,
        "checkpoint_info": info, "n_images": len(files), "n_maps": int(me.size),
        "images": "seeded random draw from results/run008_splits/test.csv (random_state=10)",
        "host": platform.platform(), "numpy": np.__version__,
        "feature_map_max_abs_diff": float(np.max(acc["feat_maxabs"])),
        "feature_map_max_abs_diff_rel_to_feature_max": float(np.max(acc["feat_relmax"])),
        "prob_max_abs_diff_edge_gradcam_path": float(np.max(acc["p_edge"])),
        "prob_max_abs_diff_numpy_head_on_torch_features": float(np.max(acc["p_headonly"])),
        "prob_max_abs_diff_full_model_plain_inference": float(np.max(acc["p_plain"])),
        "heatmap_max_abs_diff_edge": float(me.max()),
        "heatmap_mean_of_per_map_max_abs_diff_edge": float(me.mean()),
        "heatmap_p99_of_per_map_max_abs_diff_edge": float(np.percentile(me, 99)),
        "heatmap_max_abs_diff_numpy_head_on_torch_features": float(mh.max()),
        "lowres_cam_max_abs_diff_rel_to_map_max_edge": float(np.max(np.concatenate(acc["lr_edge_rel"]))),
        "heatmap_spearman_edge_vs_ref_min": float(np.min(acc["rho_edge"])),
        "heatmap_spearman_edge_vs_ref_median": float(np.median(acc["rho_edge"])),
        "heatmap_argmax_identical_fraction": float(np.mean(acc["argmax_same"])),
        "n_all_zero_reference_maps": int(me.size - len(acc["rho_edge"])),
        "heatmap_scale": "maps normalised to per-map max (range [0,1]); diffs are in those units",
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(res, open(f"{OUT}/parity_{a.runtime}_{a.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
