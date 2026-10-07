"""
Run 007b step 1: re-evaluate the existing run002 (joint 5-label) and run003
(DR-only) checkpoints on the same 1270-image val split, scoring DR under both
the patient-level D label and the per-eye keyword-derived D label.
Usage (repo root): .venv/bin/python results/run007b_eval_existing.py
"""
import json
import sys

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, "results")
from run007_common import MEAN, STD, OdirDataset, _auc, dr_metrics, load_df, split  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPTS = {
    "run002": ("results/checkpoints/run002_baseline_tuned_best_epoch15.pt", ["D", "G", "C", "A", "H"]),
    "run003": ("results/checkpoints/run003_dr_only_best_epoch9.pt", ["D"]),
}


def main():
    df = load_df()
    _, val_df = split(df)
    assert len(val_df) == 1270
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = DataLoader(OdirDataset(val_df, ["D"], tfm), batch_size=64, shuffle=False, num_workers=4)
    res = {}
    for name, (path, cols) in CKPTS.items():
        m = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=len(cols))
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.to(DEVICE).eval()
        probs = []
        with torch.no_grad():
            for x, _ in dl:
                probs.append(torch.sigmoid(m(x.to(DEVICE))).cpu().numpy())
        probs = np.concatenate(probs)
        np.save(f"results/run007b_{name}_val_probs.npy", probs)
        r = {"checkpoint": path, "label_cols": cols,
             "label_auc_patient_level": {c: _auc(val_df[c], probs[:, i]) for i, c in enumerate(cols)},
             "dr": dr_metrics(val_df, probs[:, 0])}
        res[name] = r
        print(name, json.dumps(r, indent=1))
    with open("results/run007b_eval_existing.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
