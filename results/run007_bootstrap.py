"""Patient-clustered paired bootstrap (2000 resamples of val patients) of DR AUC:
95% CI per model and CI of the difference vs run002, under both D definitions.
Usage (repo root): .venv/bin/python results/run007_bootstrap.py
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "results")
from run007_common import load_df, split  # noqa: E402

MODELS = {
    "run002": "results/run007b_run002_val_probs.npy",
    "run003": "results/run007b_run003_val_probs.npy",
    "run007b_joint5_eyeD_224": "results/run007b_joint5_eyeD_224_val_probs.npy",
    "run007c_joint5_patientD_224_s1": "results/run007c_joint5_patientD_224_s1_val_probs.npy",
    "run007a_joint5_patientD_384": "results/run007a_joint5_patientD_384_val_probs.npy",
    "run007c_joint5_patientD_384_s1": "results/run007c_joint5_patientD_384_s1_val_probs.npy",
    "run007b_joint5_eyeD_384": "results/run007b_joint5_eyeD_384_val_probs.npy",
    "run007a_joint5_patientD_512": "results/run007a_joint5_patientD_512_val_probs.npy",
    "run007c_joint5_patientD_512_s1": "results/run007c_joint5_patientD_512_s1_val_probs.npy",
    "run007b_joint5_eyeD_512": "results/run007b_joint5_eyeD_512_val_probs.npy",
    "run007c_joint5_eyeD_512_s1": "results/run007c_joint5_eyeD_512_s1_val_probs.npy",
}


def main():
    _, v = split(load_df())
    p = {}
    for k, f in MODELS.items():
        try:
            p[k] = np.load(f)[:, 0]
        except FileNotFoundError:
            print("missing", f)
    ids = v["ID"].values
    uniq = np.unique(ids)
    idx_by = {u: np.where(ids == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    B = 2000
    samples = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        samples.append(np.concatenate([idx_by[u] for u in pick]))
    out = {}
    for lab in ["D", "D_eye"]:
        y = v[lab].values
        boot = {k: np.array([roc_auc_score(y[s], p[k][s]) for s in samples]) for k in p}
        out[lab] = {}
        for k in p:
            r = {"auc": float(roc_auc_score(y, p[k])),
                 "ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))]}
            if k != "run002":
                d = boot[k] - boot["run002"]
                r["diff_vs_run002"] = float(roc_auc_score(y, p[k]) - roc_auc_score(y, p["run002"]))
                r["diff_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                r["p_diff_le_0"] = float((d <= 0).mean())
            out[lab][k] = r
            print(lab, k, json.dumps(r))
    with open("results/run007_bootstrap.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
