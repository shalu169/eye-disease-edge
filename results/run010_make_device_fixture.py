"""
Run 010 step 6a: device fixture for edge/gradcam_bench.py: 3 preprocessed test
images (seeded draw) with the Mac PyTorch-autograd reference probabilities and
512x512 Grad-CAM maps (5 labels, float16), plus a bundle directory with the
artifacts to copy to the devices.
Usage: .venv/bin/python results/run010_make_device_fixture.py --tag TAG --ckpt auto|final|path
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, "results")
from run010_common import DATA_DIR, ckpt_from_args, gradcam_torch, load_manifest, load_model, load_x, norm_max, upsample  # noqa

ART = "edge/artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", default="auto")
    a = ap.parse_args()
    ckpt, _ = ckpt_from_args(a.ckpt)
    m = load_model(ckpt)
    files = load_manifest("test").sample(n=3, random_state=30)["filename"].tolist()
    xs, ps, maps = [], [], []
    for f in files:
        x = load_x(f"{DATA_DIR}/{f}")
        p, lr, _ = gradcam_torch(m, torch.from_numpy(x)[None])
        xs.append(x)
        ps.append(p[0])
        maps.append(norm_max(upsample(lr[0])).astype(np.float16))
    np.savez(f"{ART}/run010_{a.tag}_fixture.npz", x=np.stack(xs), probs=np.stack(ps).astype(np.float64),
             maps=np.stack(maps), files=np.array(files), checkpoint=np.array(ckpt))
    b = f"{ART}/run010_bundle_{a.tag}"
    os.makedirs(b, exist_ok=True)
    for suf in ["backbone_fp32.onnx", "full_fp32.onnx", "backbone_fp32.tflite", "full_fp32.tflite", "head.npz",
                "fixture.npz"]:
        shutil.copy(f"{ART}/run010_{a.tag}_{suf}", b)
    for f in ["edge/gradcam_numpy.py", "edge/gradcam_bench.py", "edge/run010_device.sh"]:
        if os.path.exists(f):
            shutil.copy(f, b)
    print("bundle", b, sorted(os.listdir(b)))


if __name__ == "__main__":
    main()
