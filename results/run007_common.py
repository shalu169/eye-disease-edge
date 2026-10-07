"""
Run 007 shared code: per-eye DR label derivation from ODIR keywords, the
run001-003 patient split, dataset, and DR metrics under both label definitions.
Mapping documented in results/run007_dr_per_eye_label_mapping.md.
Run all run007 scripts from the repo root.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

DATA_DIR = "data/preprocessed_images"
CSV_PATH = "data/full_df.csv"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# keyword -> DR grade (1 mild NPDR, 2 moderate NPDR, 3 severe NPDR, 4 PDR, -1 DR ungraded)
DR_KEYWORDS = {
    "mild nonproliferative retinopathy": 1,
    "moderate non proliferative retinopathy": 2,
    "suspected moderate non proliferative retinopathy": 2,
    "severe nonproliferative retinopathy": 3,
    "proliferative diabetic retinopathy": 4,
    "severe proliferative diabetic retinopathy": 4,
    "diabetic retinopathy": -1,
    "suspected diabetic retinopathy": -1,
    "suspicious diabetic retinopathy": -1,
}
SUSPECTED_DR = {
    "suspected moderate non proliferative retinopathy",
    "suspected diabetic retinopathy",
    "suspicious diabetic retinopathy",
}
# DR-adjacent findings that are NOT DR keywords (label stays 0) but may be treated
# / co-occurring DR; excluded from the "strict" per-eye evaluation only.
DR_ADJACENT = {
    "laser spot",
    "post laser photocoagulation",
    "fundus laser photocoagulation spots",
    "post retinal laser surgery",
    "intraretinal hemorrhage",
    "intraretinal microvascular abnormality",
    "suspected microvascular anomalies",
    "maculopathy",
}


def tokens(s):
    return [k.strip() for k in s.replace("，", ",").split(",") if k.strip()]


def load_df():
    df = pd.read_csv(CSV_PATH)
    eye = df["filename"].str.extract(r"_(left|right)\.jpg$")[0]
    assert eye.notna().all()
    df["eye"] = eye
    df["kw"] = np.where(df["eye"] == "left", df["Left-Diagnostic Keywords"], df["Right-Diagnostic Keywords"])
    grades, susp, adj = [], [], []
    for s in df["kw"]:
        t = tokens(s)
        g = [DR_KEYWORDS[k] for k in t if k in DR_KEYWORDS]
        if not g:
            grades.append(0)
        elif max(g) > 0:
            grades.append(max(g))
        else:
            grades.append(-1)
        susp.append(any(k in SUSPECTED_DR for k in t))
        adj.append(any(k in DR_ADJACENT for k in t))
    df["dr_grade_eye"] = grades           # 0 none, 1..4 graded, -1 DR ungraded
    df["D_eye"] = (df["dr_grade_eye"] != 0).astype(int)
    df["dr_suspected_eye"] = susp
    df["dr_adjacent_eye"] = adj
    # sanity: patient-level D == any eye of that patient (in keywords) has a DR keyword
    return df


def split(df):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["ID"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


class OdirDataset(Dataset):
    def __init__(self, df, label_cols, tfm):
        self.files = df["filename"].tolist()
        self.labels = torch.tensor(df[label_cols].values.astype("float32"))
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(f"{DATA_DIR}/{self.files[idx]}").convert("RGB")
        return self.tfm(img), self.labels[idx]


def _auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def dr_metrics(val_df, p_dr):
    """DR AUC of a per-image DR probability under every label definition."""
    v = val_df.reset_index(drop=True)
    p = np.asarray(p_dr)
    g = v["dr_grade_eye"].values
    out = {
        "n": int(len(v)),
        "n_pos_patientD": int(v["D"].sum()),
        "n_pos_eyeD": int(v["D_eye"].sum()),
        # (1) ODIR patient-level D copied to each eye (runs 001-003 definition)
        "auc_patientD": _auc(v["D"], p),
        # (2) per-eye D from that eye's own keywords
        "auc_eyeD": _auc(v["D_eye"], p),
    }
    # (2b) strict per-eye: drop suspected-DR positives and DR-adjacent negatives
    m = ~((v["D_eye"] == 1) & v["dr_suspected_eye"]) & ~((v["D_eye"] == 0) & v["dr_adjacent_eye"])
    out["n_eyeD_strict"] = int(m.sum())
    out["auc_eyeD_strict"] = _auc(v.loc[m, "D_eye"], p[m.values])
    # (3) patient-level task: max prob over the patient's eyes vs patient D
    pp = pd.DataFrame({"ID": v["ID"], "p": p, "D": v["D"]}).groupby("ID").agg(p=("p", "max"), D=("D", "max"))
    out["n_patients"] = int(len(pp))
    out["auc_patient_maxeye"] = _auc(pp["D"], pp["p"])
    # severity strata vs per-eye non-DR eyes
    neg = g == 0
    for name, sel in [("mild", g == 1), ("moderate", g == 2), ("severe_npdr", g == 3), ("pdr", g == 4),
                      ("moderate_plus", g >= 2)]:
        mm = neg | sel
        out[f"n_{name}"] = int(sel.sum())
        out[f"auc_{name}_vs_noDR"] = _auc(sel[mm].astype(int), p[mm])
    # referable-ish: moderate+ vs (no DR + mild), ungraded DR eyes excluded
    mm = g != -1
    out["n_ungraded_excluded"] = int((g == -1).sum())
    out["auc_referable_modplus_vs_rest"] = _auc((g[mm] >= 2).astype(int), p[mm])
    # the noisy positives: patient D=1 but eye has no DR keyword -> mean prob
    noisy = (v["D"].values == 1) & (v["D_eye"].values == 0)
    out["n_noisy_pos"] = int(noisy.sum())
    out["mean_p_noisy_pos"] = float(p[noisy].mean())
    out["mean_p_eyeDR_pos"] = float(p[v["D_eye"].values == 1].mean())
    out["mean_p_patientD0"] = float(p[v["D"].values == 0].mean())
    return out
