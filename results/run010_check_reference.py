"""
Run 010 step 1: cross-check the reference Grad-CAM (results/run010_common.gradcam_torch)
against the pytorch-grad-cam library (jacobgil, v1.5.x) on the same images.

pytorch-grad-cam post-processes differently (min-max scaling, then cv2.resize
to the input size), so the comparison is done at two levels:
  (a) low-res 16x16 maps: our ReLU(sum_k w_k A_k), min-max scaled, vs the
      library's internal per-layer CAM (identical definition) -> should agree
      to float precision;
  (b) final 512x512 maps as the library returns them vs ours post-processed the
      library's way (min-max, cv2.resize, then min-max again in
      aggregate_multi_layers) -> should agree to float precision;
  (c) for information: library output vs our display map (max-normalised,
      torch bilinear upsample) -> differs only by the resize kernel.
Usage: .venv/bin/python results/run010_check_reference.py [--ckpt auto|final|path] [--n 64]
"""
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr

sys.path.insert(0, "results")
from run010_common import (DATA_DIR, IMG, NAMES, OUT, ckpt_from_args, gradcam_torch, load_manifest,  # noqa: E402
                           load_model, load_x, norm_max, upsample)


def minmax(m):
    m = m - m.min()
    return m / (1e-7 + m.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    torch.set_num_threads(6)
    ckpt, info = ckpt_from_args(a.ckpt)
    model = load_model(ckpt)
    df = load_manifest("test").sample(n=a.n, random_state=0)
    lib = GradCAM(model=model, target_layers=[model.blocks[-1]])
    d_lr, d_lib, d_disp, rho_disp, d_prob = [], [], [], [], []
    t0 = time.time()
    for f in df["filename"]:
        x = torch.from_numpy(load_x(f"{DATA_DIR}/{f}"))[None]
        probs, cams_lr, _ = gradcam_torch(model, x)
        with torch.no_grad():
            p_plain = torch.sigmoid(model(x)).numpy()
        d_prob.append(np.abs(probs - p_plain).max())
        disp = norm_max(upsample(cams_lr[0]))  # [5,512,512]
        for c in range(5):
            out = lib(input_tensor=x, targets=[ClassifierOutputTarget(c)])[0]  # [512,512]
            acts = lib.activations_and_grads.activations[0].cpu().numpy()
            grads = lib.activations_and_grads.gradients[0].cpu().numpy()
            lr_lib = np.maximum(lib.get_cam_image(x, model.blocks[-1], None, acts, grads, False), 0)[0]
            d_lr.append(np.abs(minmax(lr_lib) - minmax(cams_lr[0, c])).max())
            # library: min-max at 16x16 -> cv2.resize -> (aggregate_multi_layers) min-max again at 512
            ours_libstyle = minmax(cv2.resize(np.float32(minmax(cams_lr[0, c])), (IMG, IMG)))
            d_lib.append(np.abs(out - ours_libstyle).max())
            d_disp.append(np.abs(out - disp[c]).max())
            rho_disp.append(spearmanr(out.ravel(), disp[c].ravel())[0])
    lib.activations_and_grads.release()
    res = {
        "checkpoint": ckpt, "checkpoint_info": info, "n_images": a.n, "n_maps": len(d_lr),
        "pytorch_grad_cam_version": __import__("pytorch_grad_cam").__version__ if hasattr(
            __import__("pytorch_grad_cam"), "__version__") else "1.5.7 (pip)",
        "target_layer": "model.blocks[-1] (ConvBnAct 96->576, output of forward_features)",
        "max_abs_prob_diff_gradcam_forward_vs_plain_forward": float(np.max(d_prob)),
        "lowres_minmax_max_abs_diff_ours_vs_lib": float(np.max(d_lr)),
        "final512_max_abs_diff_ours_libstyle_vs_lib": float(np.max(d_lib)),
        "final512_max_abs_diff_ours_display_vs_lib": float(np.max(d_disp)),
        "final512_mean_of_maxabs_ours_display_vs_lib": float(np.mean(d_disp)),
        "final512_spearman_ours_display_vs_lib_min": float(np.nanmin(rho_disp)),
        "final512_spearman_ours_display_vs_lib_median": float(np.nanmedian(rho_disp)),
        "seconds": round(time.time() - t0, 1),
        "labels": NAMES,
    }
    json.dump(res, open(f"{OUT}/reference_vs_pytorch_grad_cam_{'final' if 'run008' in ckpt else 'dev'}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
