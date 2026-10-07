"""
Run 010 step 3b: data-randomisation sanity check (Adebayo et al. 2018, "data
randomization test"). Train the SAME architecture/resolution/init as the
paper model (MobileNetV3-small, ImageNet-pretrained, 512 px, 5 sigmoid
outputs) on the run008 training images with the label matrix
[D_eye, G, C, A, H] PERMUTED across images (rows shuffled jointly with a fixed
seed, so prevalence and label co-occurrence are kept but the image-label link is
destroyed). No augmentation (so the network can memorise), AdamW lr 3e-4,
2-epoch warmup + cosine, pos_weight as in run008. The final-epoch checkpoint is
used. If Grad-CAM maps of this model resemble the real model's, Grad-CAM
reflects image structure, not what the model learned from labels.
Reports train AUC on the permuted labels (degree of memorisation) and val AUC
on the true labels (should be ~0.5).
Usage: .venv/bin/python results/run010_train_randlabels.py [--epochs 30] [--device mps|cpu]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, "results")
from run008_common import MEAN, STD, OdirDataset, _auc  # noqa: E402

COLS = ["D_eye", "G", "C", "A", "H"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr = pd.read_csv("results/run008_splits/train.csv")
    va = pd.read_csv("results/run008_splits/val.csv")
    perm = rng.permutation(len(tr))
    trp = tr.copy()
    trp[COLS] = tr[COLS].values[perm]
    agree = float((trp[COLS].values == tr[COLS].values).all(1).mean())
    print(f"train={len(tr)} permuted; rows whose 5 labels coincide with the true ones by chance: {agree:.3f}", flush=True)
    tfm = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = DataLoader(OdirDataset(trp, COLS, tfm), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    persistent_workers=True)
    dl_eval_tr = DataLoader(OdirDataset(trp, COLS, tfm), batch_size=64, shuffle=False, num_workers=a.workers)
    dl_va = DataLoader(OdirDataset(va, COLS, tfm), batch_size=64, shuffle=False, num_workers=a.workers)
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=5).to(a.device)
    pos = trp[COLS].sum().values
    pw = torch.tensor(np.clip((len(trp) - pos) / np.maximum(pos, 1), 1, 20), dtype=torch.float32).to(a.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    spe = len(dl)
    warm, total = 2 * spe, a.epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    hist = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in dl:
            x, y = x.to(a.device), y.to(a.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(x)
        hist.append({"epoch": ep, "train_loss": tl / len(trp), "s": round(time.time() - t0)})
        print(json.dumps(hist[-1]), flush=True)

    def probs(d):
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in d:
                out.append(torch.sigmoid(model(x.to(a.device))).cpu().numpy())
        return np.concatenate(out)

    ptr, pva = probs(dl_eval_tr), probs(dl_va)
    res = {"seed": a.seed, "epochs": a.epochs, "device": a.device,
           "train_auc_on_permuted_labels": {c: _auc(trp[c], ptr[:, i]) for i, c in enumerate(COLS)},
           "val_auc_on_true_labels": {c: _auc(va[c], pva[:, i]) for i, c in enumerate(COLS)},
           "history": hist, "checkpoint": "results/checkpoints/run010_randlabels_512_final.pt",
           "frac_rows_label_vector_unchanged_by_permutation": agree}
    torch.save(model.state_dict(), res["checkpoint"])
    json.dump(res, open("results/run010/randlabels_train.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
