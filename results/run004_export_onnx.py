"""
Export the run002 checkpoint to ONNX and write the val/calibration manifests
used by the edge benchmarks (N100 now, Pi later).

Also writes PyTorch CPU reference probabilities on the val split, so every edge
runtime can be checked for numeric parity, not just matching AUC.

Usage (from repo root):  .venv/bin/python edge/export_onnx.py
"""

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


def main():
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

    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(LABEL_COLS))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    onnx_path = f"{OUT_DIR}/run002_mnv3s_fp32.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    probs = []
    with torch.no_grad():
        for i in range(0, len(val_df), 64):
            batch = torch.stack([
                val_tfm(Image.open(f"{DATA_DIR}/{f}").convert("RGB"))
                for f in val_df["filename"].iloc[i:i + 64]
            ])
            probs.append(torch.sigmoid(model(batch)).numpy())
    probs = np.concatenate(probs)
    np.save(f"{OUT_DIR}/torch_cpu_val_probs.npy", probs)

    labels = val_df[LABEL_COLS].values
    aucs = {c: round(float(roc_auc_score(labels[:, i], probs[:, i])), 4) for i, c in enumerate(LABEL_COLS)}
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


if __name__ == "__main__":
    main()
