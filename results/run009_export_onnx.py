"""
Export a MobileNetV3-small checkpoint to ONNX and write PyTorch CPU reference
probabilities on an evaluation manifest, so every edge runtime can be checked
for numeric parity, not just matching AUC.

Run 004 (default, no arguments): run002 checkpoint at 224 px, writes the
val/calibration manifests (edge/artifacts/val_manifest.csv,
calib_manifest.csv), run002_mnv3s_fp32.onnx, torch_cpu_val_probs.npy and
export_summary.json.

Run 009+ (generalised): any checkpoint / input size / manifest, e.g.
  .venv/bin/python edge/export_onnx.py --ckpt results/checkpoints/run007c_joint5_eyeD_512_s1_best_epoch22.pt \
      --img 512 --tag run007c_eyeD512 --manifest edge/artifacts/val_manifest_run009.csv --ref-name val
  -> edge/artifacts/<tag>_fp32.onnx, edge/artifacts/<tag>_torch_cpu_<ref-name>_probs.npy,
     edge/artifacts/<tag>_export_summary.json
  --no-export only (re)computes reference probabilities for another manifest.
  The manifest must have a `filename` column; AUCs are computed for --label-cols
  that exist in it. Preprocessing = run007/run002 val_tfm: PIL bilinear
  Resize((S,S)), ToTensor, ImageNet Normalize. Always CPU.

Usage (from repo root):  .venv/bin/python edge/export_onnx.py [args]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms

CKPT = "results/checkpoints/run002_baseline_tuned_best_epoch15.pt"
CSV_PATH = "data/full_df.csv"
DATA_DIR = "data/preprocessed_images"
LABEL_COLS = ["D", "G", "C", "A", "H"]
IMG_SIZE = 224
OUT_DIR = "edge/artifacts"
CALIB_N = 300  # train-split images for INT8 post-training quantization


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_model(ckpt):
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def export(model, img, onnx_path):
    dummy = torch.randn(1, 3, img, img)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def torch_probs(model, files, img):
    val_tfm = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    probs = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            batch = torch.stack([val_tfm(Image.open(f"{DATA_DIR}/{f}").convert("RGB")) for f in files[i:i + 64]])
            probs.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(probs)


def aucs_for(df, probs, label_cols):
    # output order is always D,G,C,A,H; label_cols[i] is the column scored against output i
    out = {}
    for i, c in enumerate(label_cols):
        if c in df.columns:
            out[c] = round(float(roc_auc_score(df[c].values, probs[:, i])), 4)
    return out


def main_run004():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    # identical split to baseline.py / run002
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["ID"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    assert len(val_df) == 1270, len(val_df)

    val_df[["filename"] + LABEL_COLS].to_csv(f"{OUT_DIR}/val_manifest.csv", index=False)
    calib_df = train_df.sample(n=CALIB_N, random_state=0)
    calib_df[["filename"]].to_csv(f"{OUT_DIR}/calib_manifest.csv", index=False)

    model = load_model(CKPT)
    onnx_path = f"{OUT_DIR}/run002_mnv3s_fp32.onnx"
    export(model, IMG_SIZE, onnx_path)
    probs = torch_probs(model, val_df["filename"].tolist(), IMG_SIZE)
    np.save(f"{OUT_DIR}/torch_cpu_val_probs.npy", probs)

    aucs = aucs_for(val_df, probs, LABEL_COLS)
    summary = {
        "checkpoint": CKPT,
        "onnx": onnx_path,
        "onnx_bytes": os.path.getsize(onnx_path),
        "torch_cpu_val_auc": aucs,
        "torch_cpu_mean_auc": round(float(np.mean(list(aucs.values()))), 4),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "python": sys.version.split()[0],
    }
    with open(f"{OUT_DIR}/export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main_general(a):
    torch.set_num_threads(a.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(a.ckpt)
    onnx_path = f"{OUT_DIR}/{a.tag}_fp32.onnx"
    sum_path = f"{OUT_DIR}/{a.tag}_export_summary.json"
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
    if not a.no_export:
        export(model, a.img, onnx_path)
        summary.update({"checkpoint": a.ckpt, "checkpoint_md5": md5(a.ckpt), "img": a.img, "onnx": onnx_path,
                        "onnx_bytes": os.path.getsize(onnx_path), "onnx_md5": md5(onnx_path), "opset": 17})
    df = pd.read_csv(a.manifest)
    if a.n:
        df = df.iloc[:a.n]
    probs = torch_probs(model, df["filename"].tolist(), a.img)
    ref_path = f"{OUT_DIR}/{a.tag}_torch_cpu_{a.ref_name}_probs.npy"
    np.save(ref_path, probs)
    label_cols = a.label_cols.split(",")
    aucs = aucs_for(df, probs, label_cols)
    extra = {}
    if "D" in df.columns and label_cols[0] != "D":
        extra["D_patient"] = round(float(roc_auc_score(df["D"].values, probs[:, 0])), 4)
    summary.setdefault("references", {})[a.ref_name] = {
        "manifest": a.manifest, "n": len(df), "probs": ref_path, "label_cols": label_cols,
        "torch_cpu_auc": aucs, "torch_cpu_mean_auc": round(float(np.mean(list(aucs.values()))), 4), **extra}
    summary.update({"torch": torch.__version__, "timm": timm.__version__, "python": sys.version.split()[0],
                    "device": "cpu", "preprocess": f"PIL bilinear Resize(({a.img},{a.img})), ToTensor, ImageNet Normalize"})
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--img", type=int, default=IMG_SIZE)
    ap.add_argument("--tag")
    ap.add_argument("--manifest")
    ap.add_argument("--ref-name", default="val")
    ap.add_argument("--label-cols", default="D_eye,G,C,A,H")
    ap.add_argument("--n", type=int, default=0, help="only first N manifest rows (0 = all)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.ckpt is None:
        main_run004()
    else:
        assert a.tag and a.manifest, "--tag and --manifest are required with --ckpt"
        main_general(a)
