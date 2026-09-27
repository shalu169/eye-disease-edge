"""
Multi-label baseline: DR / glaucoma / cataract / AMD / hypertensive retinopathy
from a single fundus image (ODIR-5K, per-eye images with patient-level labels).

Non-quantized, Mac/MPS only. Purpose: confirm feasibility before investing in
the edge (OpenVINO/N100, TFLite/Pi) pipeline. See brainstorm.md section 5-8.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

DATA_DIR = "data/preprocessed_images"
CSV_PATH = "data/full_df.csv"
LABEL_COLS = ["D", "G", "C", "A", "H"]  # DR, glaucoma, cataract, AMD, hypertensive
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class OdirDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{DATA_DIR}/{row['filename']}").convert("RGB")
        img = self.tfm(img)
        label = torch.tensor(row[LABEL_COLS].values.astype("float32"))
        return img, label


def build_model():
    m = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=len(LABEL_COLS))
    return m


def main():
    df = pd.read_csv(CSV_PATH)

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # split by patient ID so both eyes of a patient stay on the same side
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["ID"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    print(f"train={len(train_df)} val={len(val_df)} (split by patient ID, no leakage)")

    train_ds = OdirDataset(train_df, train_tfm)
    val_ds = OdirDataset(val_df, val_tfm)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model().to(DEVICE)

    # class-imbalance handling: pos_weight per label (inverse prevalence, capped)
    pos = train_df[LABEL_COLS].sum().values
    neg = len(train_df) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32).to(DEVICE)
    print("pos_weight per label", dict(zip(LABEL_COLS, pos_weight.tolist())))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_dl)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_mean_auc = -1.0
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(DEVICE)
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        aucs = {}
        for i, name in enumerate(LABEL_COLS):
            if len(np.unique(all_labels[:, i])) < 2:
                aucs[name] = float("nan")
            else:
                aucs[name] = roc_auc_score(all_labels[:, i], all_probs[:, i])
        mean_auc = float(np.nanmean(list(aucs.values())))
        cur_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch+1}/{EPOCHS} lr={cur_lr:.2e} train_loss={train_loss:.4f} "
              f"mean_auc={mean_auc:.4f} val_auc={ {k: round(v,4) for k,v in aucs.items()} }")

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            torch.save(model.state_dict(), "baseline_mobilenetv3_small_best.pt")
            print(f"  -> new best mean_auc={mean_auc:.4f}, saved baseline_mobilenetv3_small_best.pt")

    print(f"done. best mean_auc={best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
