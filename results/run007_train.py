"""
Run 007 training: run002 recipe (MobileNetV3-small, AdamW 3e-4, wd 1e-4, 2-ep
warmup + cosine, run002 augmentation, pos_weight inverse prevalence capped
[1,20], best checkpoint by mean val AUC) with two knobs:
  --img      input resolution (source images are 512x512)
  --dlabel   which DR label to TRAIN on: 'patient' (ODIR D, runs 001-003) or
             'eye' (per-eye D from that eye's keywords, see run007_common.py)
  --cols     joint5 (D,G,C,A,H) or D (DR-only)
Every epoch logs DR AUC under BOTH label definitions. Checkpoint selection uses
mean AUC over the label columns with D scored under the TRAINING definition.
Optional --patience stops early (cosine schedule itself is unchanged).
Usage (repo root): .venv/bin/python results/run007_train.py --tag run007a_joint5_384 --img 384 --dlabel patient
"""
import argparse
import json
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, "results")
from run007_common import MEAN, STD, OdirDataset, _auc, dr_metrics, load_df, split  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--dlabel", choices=["patient", "eye"], default="patient")
    ap.add_argument("--cols", choices=["joint5", "D"], default="joint5")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    print("args", vars(a), "device", DEVICE, flush=True)
    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

    df = load_df()
    df["D_train"] = df["D"] if a.dlabel == "patient" else df["D_eye"]
    train_df, val_df = split(df)
    assert len(val_df) == 1270
    names = ["D", "G", "C", "A", "H"] if a.cols == "joint5" else ["D"]
    cols = ["D_train"] + names[1:]
    print(f"train={len(train_df)} val={len(val_df)}  train D positives={int(train_df['D_train'].sum())} "
          f"(patient D={int(train_df['D'].sum())}, eye D={int(train_df['D_eye'].sum())})", flush=True)

    S = a.img
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tfm = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_dl = DataLoader(OdirDataset(train_df, cols, train_tfm), batch_size=a.bs, shuffle=True,
                          num_workers=a.workers, persistent_workers=True)
    val_dl = DataLoader(OdirDataset(val_df, cols, val_tfm), batch_size=64, shuffle=False,
                        num_workers=a.workers, persistent_workers=True)

    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=len(cols)).to(DEVICE)
    pos = train_df[cols].sum().values
    neg = len(train_df) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32).to(DEVICE)
    print("pos_weight", dict(zip(names, [round(x, 3) for x in pos_weight.tolist()])), flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(train_dl)
    warm, total = 2 * spe, a.epochs * spe

    def lr_lambda(step):
        if step < warm:
            return step / max(1, warm)
        return 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best, best_ep, since, history = -1.0, -1, 0, []
    ckpt_tmp = f"results/checkpoints/{a.tag}_best.pt"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        te = time.time()
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * x.size(0)
        tl /= len(train_df)
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in val_dl:
                probs.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
        probs = np.concatenate(probs)
        aucs = {n: _auc(val_df[c], probs[:, i]) for i, (n, c) in enumerate(zip(names, cols))}
        d_pat, d_eye = _auc(val_df["D"], probs[:, 0]), _auc(val_df["D_eye"], probs[:, 0])
        sel = float(np.mean(list(aucs.values())))
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "train_loss": tl, "sel_mean_auc": sel,
               "D_auc_patientD": d_pat, "D_auc_eyeD": d_eye,
               **{f"{n}_auc": aucs[n] for n in names[1:]}, "epoch_s": time.time() - te}
        history.append(rec)
        print(f"epoch {ep}/{a.epochs} lr={rec['lr']:.2e} loss={tl:.4f} sel_mean_auc={sel:.4f} "
              f"D[patient]={d_pat:.4f} D[eye]={d_eye:.4f} "
              + " ".join(f"{n}={aucs[n]:.4f}" for n in names[1:]) + f" ({rec['epoch_s']:.0f}s)", flush=True)
        if sel > best:
            best, best_ep, since = sel, ep, 0
            torch.save(model.state_dict(), ckpt_tmp)
            np.save(f"results/{a.tag}_val_probs.npy", probs)
            print(f"  -> new best sel_mean_auc={sel:.4f}", flush=True)
        else:
            since += 1
            if a.patience and since >= a.patience:
                print(f"early stop: no improvement for {a.patience} epochs", flush=True)
                break

    import os
    final_ckpt = f"results/checkpoints/{a.tag}_best_epoch{best_ep}.pt"
    os.replace(ckpt_tmp, final_ckpt)
    probs = np.load(f"results/{a.tag}_val_probs.npy")
    summary = {
        "tag": a.tag, "args": vars(a), "best_epoch": best_ep, "epochs_run": len(history),
        "best_sel_mean_auc": best, "checkpoint": final_ckpt, "wall_s": time.time() - t0,
        "best_label_auc": {n: _auc(val_df[c], probs[:, i]) for i, (n, c) in enumerate(zip(names, cols))},
        "best_label_auc_patient_level": {n: _auc(val_df[n], probs[:, i]) for i, n in enumerate(names)},
        "dr": dr_metrics(val_df, probs[:, 0]),
        "history": history,
    }
    with open(f"results/{a.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("BEST", json.dumps({k: summary[k] for k in ["best_epoch", "best_sel_mean_auc", "best_label_auc",
                                                       "checkpoint", "wall_s"]}), flush=True)
    print("DR", json.dumps(summary["dr"]), flush=True)


if __name__ == "__main__":
    main()
