"""
Run 010 shared code: model loading, preprocessing and the PyTorch reference
Grad-CAM for the 5-label MobileNetV3-small.

Grad-CAM definition used everywhere in run 010 (Selvaraju et al. 2017):
  A      = model.forward_features(x)        last conv feature map before GAP,
                                             [B, 576, h, w] (h = w = 16 at 512 px)
  y_c    = logit of label c (pre-sigmoid)
  w_c,k  = mean_{i,j} d y_c / d A_k,ij        (global-average of the gradient)
  L_c    = ReLU( sum_k w_c,k A_k )            [B, h, w]   ("raw" low-res map)
  map    = bilinear upsample of L_c to the input size (torch F.interpolate,
           align_corners=False), divided by its per-image max (0 if max == 0)
All evaluation metrics (pointing game, energy, AUROC, deletion/insertion order)
are invariant to the final per-image scaling; it only matters for display.

Model choice: results/run008_FINAL_CHECKPOINT.json once it says
"updated_after_all_seeds": true; otherwise the development checkpoint
run007c_joint5_eyeD_512_s1 (same architecture/size; NOTE it was trained on the
images that later became the run008 TEST split, so test-set numbers from it
are development numbers only).
Run all run010 scripts from the repo root.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "results")
from run008_common import MEAN, STD, NAMES  # noqa: E402

DEV_CKPT = "results/checkpoints/run007c_joint5_eyeD_512_s1_best_epoch22.pt"
FINAL_JSON = "results/run008_FINAL_CHECKPOINT.json"
IMG = 512
DATA_DIR = "data/preprocessed_images"
OUT = "results/run010"
TEST_CSV = "results/run008_splits/test.csv"
VAL_CSV = "results/run008_splits/val.csv"
# label column used to define "positive" for each output: per-eye D for DR,
# ODIR patient-level labels for the others (as in run 008)
POS_COL = {"D": "D_eye", "G": "G", "C": "C", "A": "A", "H": "H"}
MEAN_A = np.array(MEAN, dtype=np.float32)
STD_A = np.array(STD, dtype=np.float32)


def resolve_ckpt(require_final=False):
    """Return (path, info dict). Final run008 checkpoint only once all seeds are done."""
    if os.path.exists(FINAL_JSON):
        info = json.load(open(FINAL_JSON))
        if info.get("updated_after_all_seeds"):
            return info["checkpoint"], {"source": FINAL_JSON, **info}
    if require_final:
        raise SystemExit("run008_FINAL_CHECKPOINT.json missing or not updated_after_all_seeds")
    return DEV_CKPT, {"source": "development checkpoint (run 008 final not available yet)",
                      "checkpoint": DEV_CKPT}


def ckpt_from_args(argv_ckpt):
    if argv_ckpt in (None, "auto"):
        return resolve_ckpt()
    if argv_ckpt == "final":
        return resolve_ckpt(require_final=True)
    return argv_ckpt, {"source": "command line", "checkpoint": argv_ckpt}


def load_model(path):
    m = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def load_rgb(path, size=IMG):
    """uint8 HWC, PIL bilinear Resize((S,S)) as run007/008 val_tfm (no-op for 512 ODIR)."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def normalize_rgb(rgb_u8):
    """uint8 HWC -> float32 CHW, ToTensor + ImageNet Normalize."""
    x = (rgb_u8.astype(np.float32) / 255.0 - MEAN_A) / STD_A
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def load_x(path, size=IMG):
    return normalize_rgb(load_rgb(path, size))


def gradcam_torch(model, x, labels=range(5)):
    """Reference Grad-CAM with autograd w.r.t. the feature map.
    x: float tensor [B,3,H,W]. Returns probs [B,5] (np), raw low-res maps
    [B,L,h,w] (np, after ReLU), feature map A (np, [B,576,h,w])."""
    labels = list(labels)
    A = model.forward_features(x)
    A_leaf = A.detach().requires_grad_(True)
    logits = model.forward_head(A_leaf)
    cams = []
    for i, c in enumerate(labels):
        (g,) = torch.autograd.grad(logits[:, c].sum(), A_leaf, retain_graph=i < len(labels) - 1)
        w = g.mean(dim=(2, 3), keepdim=True)
        cams.append(F.relu((w * A_leaf).sum(1)))
    cams = torch.stack(cams, 1)
    return (torch.sigmoid(logits).detach().numpy(), cams.detach().numpy(), A.detach().numpy())


def upsample(cam_lr, size=IMG):
    """[..., h, w] -> [..., size, size] bilinear (align_corners=False), torch."""
    t = torch.as_tensor(np.asarray(cam_lr, dtype=np.float32))
    shp = t.shape
    t = t.reshape(-1, 1, shp[-2], shp[-1])
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return up.reshape(*shp[:-2], size, size).numpy()


def norm_max(m):
    """Per-map division by max over the last two axes (0 stays 0)."""
    mx = m.max(axis=(-2, -1), keepdims=True)
    return np.where(mx > 0, m / np.where(mx > 0, mx, 1), 0.0).astype(np.float32)


def fov_mask(rgb_u8, thr=15):
    """Field-of-view (fundus disc) mask: grey level above a small threshold."""
    return rgb_u8.astype(np.float32).mean(-1) > thr


def load_manifest(which="test"):
    return pd.read_csv(TEST_CSV if which == "test" else VAL_CSV)


def centre_bias(size=IMG, sigma_frac=0.25):
    """Isotropic Gaussian centred on the image (= disc) centre, sigma = 0.25*size."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2
    return np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * (sigma_frac * size) ** 2)).astype(np.float32)


def random_smooth_map(rng, lr=16, size=IMG):
    """Random-saliency baseline with the same spatial structure as Grad-CAM:
    i.i.d. U(0,1) on the 16x16 feature grid, bilinearly upsampled."""
    return upsample(rng.random((lr, lr)).astype(np.float32), size)
