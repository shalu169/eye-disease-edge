"""
Run 010 step 2a (torch side, .venv): export the Grad-CAM edge artifacts.
  edge/artifacts/run010_<tag>_backbone_fp32.onnx  image [1,3,512,512] -> features [1,576,16,16]
  edge/artifacts/run010_<tag>_full_fp32.onnx      image -> logits [1,5]  (plain-inference baseline)
  edge/artifacts/run010_<tag>_head.npz            W1 [1024,576], b1, W2 [5,1024], b2 (float32)
TFLite conversion of both ONNX files: results/run010_export_tflite.py (TF venv).
Usage: .venv/bin/python results/run010_export_edge.py --ckpt auto|final|path --tag TAG
"""
import argparse
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "results")
from run010_common import IMG, ckpt_from_args, load_model  # noqa: E402

ART = "edge/artifacts"


class Backbone(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    ckpt, info = ckpt_from_args(a.ckpt)
    m = load_model(ckpt)
    dummy = torch.randn(1, 3, IMG, IMG)
    files = {}
    for name, mod, out in [("backbone", Backbone(m).eval(), "features"), ("full", m, "logits")]:
        p = f"{ART}/run010_{a.tag}_{name}_fp32.onnx"
        torch.onnx.export(mod, dummy, p, input_names=["image"], output_names=[out],
                          opset_version=17, dynamic_axes=None, dynamo=False)
        files[name] = {"path": p, "md5": md5(p)}
    sd = m.state_dict()
    head = {"W1": sd["conv_head.weight"][:, :, 0, 0].numpy(), "b1": sd["conv_head.bias"].numpy(),
            "W2": sd["classifier.weight"].numpy(), "b2": sd["classifier.bias"].numpy()}
    assert type(m.norm_head).__name__ == "Identity" and type(m.act2).__name__ == "Hardswish"
    hp = f"{ART}/run010_{a.tag}_head.npz"
    np.savez(hp, **{k: v.astype(np.float32) for k, v in head.items()})
    files["head"] = {"path": hp, "md5": md5(hp), "shapes": {k: list(v.shape) for k, v in head.items()}}
    res = {"checkpoint": ckpt, "checkpoint_info": info, "tag": a.tag, "img": IMG, "torch": torch.__version__,
           "opset": 17, "files": files}
    json.dump(res, open(f"{ART}/run010_{a.tag}_export.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
