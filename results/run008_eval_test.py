"""
Run 008 one-shot evaluation on the held-out TEST set.

For every finished run008 model (results/run008{A,B,C}_*_summary.json):
  - compute per-image probabilities on VALIDATION and TEST (PyTorch, CPU)
  - per-label test AUC; DR under both per-eye and patient-level labels,
    severity strata and referable DR (run007_common.dr_metrics)
  - per-label threshold chosen on VALIDATION at 90% sensitivity, applied
    unchanged to TEST -> test sensitivity / specificity
Inference runs on MPS if available.
Then, per model group (A, B, C), mean +- s.d. over seeds and a
patient-clustered bootstrap (2000 resamples of test patients) of the
seed-averaged AUC, plus paired differences A-B and A-C.

The test set is only read here. Thresholds and checkpoints never use it.

Usage (repo root):
  .venv/bin/python results/run008_eval_test.py                # all finished run008 models
  .venv/bin/python results/run008_eval_test.py --dry-run      # code check: run007 ckpt, VALIDATION only
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

sys.path.insert(0, "results")
from run008_common import MEAN, STD, NAMES, dr_metrics, load_df, split3, thr_at_sens  # noqa: E402

DATA_DIR = "data/preprocessed_images"
PROB_DIR = "results/run008_probs"
GROUPS = {"A": "512 px, per-eye DR labels", "B": "224 px, patient-level DR labels",
          "C": "512 px, patient-level DR labels"}
B_BOOT = 2000
# MPS once training has released the GPU; CPU vs MPS probabilities agree to ~1e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def probs_for(ckpt, img, df):
    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval().to(DEVICE)
    tfm = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])
    out = []
    with torch.no_grad():
        files = df["filename"].tolist()
        for i in range(0, len(files), 32):
            x = torch.stack([tfm(Image.open(f"{DATA_DIR}/{f}").convert("RGB")) for f in files[i:i + 32]])
            out.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def label_cols(dlabel):
    # column scored against output i; D follows the model's TRAINING label definition
    return ["D_eye" if dlabel == "eye" else "D", "G", "C", "A", "H"]


def auc(y, p):
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def evaluate(tag, s, val, test, dry):
    img = s.get("img") or s["args"]["img"]
    dlabel = s.get("dlabel") or s["args"]["dlabel"]
    os.makedirs(PROB_DIR, exist_ok=True)
    pv_path, pt_path = f"{PROB_DIR}/{tag}_val.npy", f"{PROB_DIR}/{tag}_test.npy"
    pv = np.load(pv_path) if os.path.exists(pv_path) else probs_for(s["checkpoint"], img, val)
    np.save(pv_path, pv)
    cols = label_cols(dlabel)
    r = {"tag": tag, "checkpoint": s["checkpoint"], "img": img, "dlabel": dlabel,
         "val_auc": {n: auc(val[c], pv[:, i]) for i, (n, c) in enumerate(zip(NAMES, cols))}}
    thr = {n: thr_at_sens(val[c].values, pv[:, i], 0.90) for i, (n, c) in enumerate(zip(NAMES, cols))}
    r["thr_val_sens90"] = thr
    if dry:
        return r, pv, None
    pt = np.load(pt_path) if os.path.exists(pt_path) else probs_for(s["checkpoint"], img, test)
    np.save(pt_path, pt)
    r["test_auc"] = {n: auc(test[c], pt[:, i]) for i, (n, c) in enumerate(zip(NAMES, cols))}
    r["test_auc"]["D_patient"] = auc(test["D"], pt[:, 0])
    r["test_auc"]["D_eye"] = auc(test["D_eye"], pt[:, 0])
    r["test_mean_auc"] = float(np.mean([r["test_auc"][n] for n in NAMES]))
    r["test_dr"] = dr_metrics(test, pt[:, 0])
    ops = {}
    for i, (n, c) in enumerate(zip(NAMES, cols)):
        y = test[c].values.astype(bool)
        yhat = pt[:, i] >= thr[n]
        ops[n] = {"label_col": c, "thr": thr[n], "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                  "sensitivity": float((yhat & y).sum() / y.sum()),
                  "specificity": float((~yhat & ~y).sum() / (~y).sum())}
    r["test_ops_val_sens90"] = ops
    return r, pv, pt


def bootstrap(test, groups_probs):
    """groups_probs: {group: (cols, [probs per seed])}. Patient-clustered bootstrap of seed-mean AUC."""
    ids = test["ID"].values
    uniq = np.unique(ids)
    idx_by = {u: np.where(ids == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    samples = [np.concatenate([idx_by[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
               for _ in range(B_BOOT)]
    targets = {n: n for n in NAMES}
    out, boots = {}, {}
    for g, (cols, plist) in groups_probs.items():
        out[g], boots[g] = {}, {}
        for i, n in enumerate(NAMES + ["D_patient", "D_eye"]):
            col = {"D_patient": "D", "D_eye": "D_eye"}.get(n, cols[NAMES.index(n)] if n in targets else n)
            k = 0 if n in ("D_patient", "D_eye") else NAMES.index(n)
            y = test[col].values
            point = float(np.mean([roc_auc_score(y, p[:, k]) for p in plist]))
            bs = np.array([np.mean([roc_auc_score(y[s], p[s, k]) for p in plist]) for s in samples])
            boots[g][n] = bs
            out[g][n] = {"label_col": col, "seed_mean_auc": point,
                         "seed_sd": float(np.std([roc_auc_score(y, p[:, k]) for p in plist], ddof=1))
                         if len(plist) > 1 else float("nan"),
                         "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    diffs = {}
    for other in [g for g in ("B", "C") if g in boots and "A" in boots]:
        diffs[f"A_minus_{other}"] = {}
        for n in ("G", "C", "A", "H", "D_patient", "D_eye"):
            d = boots["A"][n] - boots[other][n]
            diffs[f"A_minus_{other}"][n] = {
                "diff": out["A"][n]["seed_mean_auc"] - out[other][n]["seed_mean_auc"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_diff_le_0": float((d <= 0).mean())}
    return out, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    df = load_df()
    _, val, test = split3(df)
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)

    if a.dry_run:
        tag = "run007c_joint5_eyeD_512_s1"
        s = json.load(open(f"results/{tag}_summary.json"))
        s.setdefault("img", 512)
        s.setdefault("dlabel", "eye")
        global PROB_DIR
        PROB_DIR = "results/run008_probs_dryrun"
        r, pv, _ = evaluate(tag, s, val, test, dry=True)
        ref = np.load(f"results/{tag}_val_probs.npy")
        r["max_abs_diff_vs_run007_val_probs"] = float(np.abs(pv - ref).max())
        print(json.dumps(r, indent=1))
        return

    results, groups_probs = {}, {}
    for f in sorted(glob.glob("results/run008[ABC]_*_summary.json")):
        tag = os.path.basename(f)[:-len("_summary.json")]
        s = json.load(open(f))
        r, _, pt = evaluate(tag, s, val, test, dry=False)
        results[tag] = r
        g = tag[len("run008")]
        groups_probs.setdefault(g, (label_cols(r["dlabel"]), []))[1].append(pt)
        print(tag, json.dumps({k: round(v, 4) for k, v in r["test_auc"].items()}), flush=True)
    test[["filename", "ID"]].to_csv(f"{PROB_DIR}/test_order.csv", index=False)
    val[["filename", "ID"]].to_csv(f"{PROB_DIR}/val_order.csv", index=False)
    summary, diffs = bootstrap(test, groups_probs)
    out = {"groups": GROUPS, "n_seeds": {g: len(v[1]) for g, v in groups_probs.items()},
           "test_n_images": len(test), "test_n_patients": int(test["ID"].nunique()),
           "per_model": results, "group_summary": summary, "paired_diffs": diffs}
    with open("results/run008_test_metrics.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"group_summary": summary, "paired_diffs": diffs}, indent=1))


if __name__ == "__main__":
    main()
