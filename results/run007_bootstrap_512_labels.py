"""Paired patient-clustered bootstrap: 512px eye-D-trained vs patient-D-trained (seed-matched pairs), scored on per-eye D.
Usage (repo root): .venv/bin/python results/run007_bootstrap_512_labels.py"""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "results")
from run007_common import load_df, split
_, v = split(load_df())
pairs = [("run007b_joint5_eyeD_512", "run007a_joint5_patientD_512"), ("run007c_joint5_eyeD_512_s1", "run007c_joint5_patientD_512_s1")]
ids = v["ID"].values; uniq = np.unique(ids); idx = {u: np.where(ids == u)[0] for u in uniq}
rng = np.random.default_rng(0)
S = [np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(2000)]
out = {}
for lab in ["D_eye", "D"]:
    y = v[lab].values
    for a, b in pairs:
        pa, pb = np.load(f"results/{a}_val_probs.npy")[:, 0], np.load(f"results/{b}_val_probs.npy")[:, 0]
        d = np.array([roc_auc_score(y[s], pa[s]) - roc_auc_score(y[s], pb[s]) for s in S])
        r = {"diff": roc_auc_score(y, pa) - roc_auc_score(y, pb), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_diff_le_0": float((d <= 0).mean())}
        out[f"{lab}:{a}-{b}"] = r; print(lab, a, "-", b, json.dumps(r))
json.dump(out, open("results/run007_bootstrap_512_labels.json", "w"), indent=1)
