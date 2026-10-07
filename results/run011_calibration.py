"""
Run 011: confidence calibration of the run 008 models, on device and under INT8.

Per model and per label (binary, sigmoid outputs):
  - temperature scaling (Guo et al. 2017): one scalar T per label, fitted by
    minimising validation NLL on logit(p); applied unchanged to test
  - Platt scaling (Platt 1999): sigma(a*z + b) per label, also fitted on
    validation NLL. Needed because the loss's positive-class weight w shifts
    logits by about log(w), a bias that a single temperature cannot remove
  - test metrics before / after scaling: ECE (15 equal-width bins), adaptive
    ECE (15 equal-mass bins), Brier score, NLL; AUC is unchanged by scaling
  - patient-clustered bootstrap (1000) 95% CIs for test ECE of the final model
Device check: the final model's test probabilities from the N100 (ONNX
Runtime, OpenVINO CPU/iGPU) and the Pi (TFLite) with the validation-fitted T.
INT8 check (development, 224 px run002 model, validation set, from run 006
Pi probabilities): ECE of FP32 vs dynamic-range and full-integer INT8, raw.

Thresholds and temperatures are fitted on VALIDATION only; test is read once.

Usage (repo root): .venv/bin/python results/run011_calibration.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

sys.path.insert(0, "results")
from run008_common import NAMES, load_df, split3  # noqa: E402

PROB_DIR = "results/run008_probs"
OUT = "results/run011"
FIG = "paper/figs"
FINAL = json.load(open("results/run008_FINAL_CHECKPOINT.json"))
EPS = 1e-6
NB = 15
FULL = {"D": "DR", "G": "Glaucoma", "C": "Cataract", "A": "AMD", "H": "Hypertensive ret."}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, nb=NB, adaptive=False):
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, nb + 1))
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.linspace(0, 1, nb + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, nb - 1)
    e = 0.0
    for b in range(nb):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def fit_T(y, p):
    z = logit(p)
    r = minimize_scalar(lambda lt: nll(y, sig(z / np.exp(lt))), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def fit_platt(y, p):
    from scipy.optimize import minimize
    z = logit(p)
    r = minimize(lambda ab: nll(y, sig(ab[0] * z + ab[1])), x0=[1.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    return [float(r.x[0]), float(r.x[1])]


def platt(p, ab):
    return sig(ab[0] * logit(p) + ab[1])


def metrics(y, p):
    return {"ece": ece(y, p), "ece_adaptive": ece(y, p, adaptive=True),
            "brier": float(np.mean((p - y) ** 2)), "nll": nll(y, p),
            "mean_p": float(p.mean()), "prevalence": float(y.mean())}


def label_cols(tag):
    return ["D_eye" if "eyeD" in tag else "D", "G", "C", "A", "H"]


def reliability(ax, y, p, color, label):
    edges = np.linspace(0, 1, NB + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, NB - 1)
    xs, ys = [], []
    for b in range(NB):
        m = idx == b
        if m.sum() >= 5:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    ax.plot(xs, ys, "o-", ms=3, lw=1.2, color=color, label=label)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_df()
    _, val, test = split3(df)
    val, test = val.reset_index(drop=True), test.reset_index(drop=True)
    out = {"per_model": {}}

    for f in sorted(glob.glob(f"{PROB_DIR}/run008*_val.npy")):
        tag = os.path.basename(f)[:-len("_val.npy")]
        pv, pt = np.load(f), np.load(f"{PROB_DIR}/{tag}_test.npy")
        cols = label_cols(tag)
        res = {}
        for i, (n, c) in enumerate(zip(NAMES, cols)):
            yv, yt = val[c].values.astype(float), test[c].values.astype(float)
            T = fit_T(yv, pv[:, i])
            ab = fit_platt(yv, pv[:, i])
            res[n] = {"label_col": c, "T": T, "platt_ab": ab,
                      "test_raw": metrics(yt, pt[:, i]),
                      "test_ts": metrics(yt, sig(logit(pt[:, i]) / T)),
                      "test_platt": metrics(yt, platt(pt[:, i], ab))}
        out["per_model"][tag] = res

    # group means over seeds
    groups = {}
    for tag, res in out["per_model"].items():
        g = tag[len("run008")]
        for n in NAMES:
            for k in ("test_raw", "test_ts", "test_platt"):
                groups.setdefault(g, {}).setdefault(n, {}).setdefault(k, []).append(res[n][k]["ece"])
            groups[g][n].setdefault("T", []).append(res[n]["T"])
    out["group_ece_mean_sd"] = {g: {n: {k: [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")]
                                        for k, v in d.items()} for n, d in gd.items()} for g, gd in groups.items()}

    # final model: bootstrap CI, device check, reliability figure
    ftag = os.path.basename(FINAL["checkpoint"]).split("_best_epoch")[0]
    pv, pt = np.load(f"{PROB_DIR}/{ftag}_val.npy"), np.load(f"{PROB_DIR}/{ftag}_test.npy")
    cols = label_cols(ftag)
    Ts = {n: out["per_model"][ftag][n]["T"] for n in NAMES}
    ABs = {n: out["per_model"][ftag][n]["platt_ab"] for n in NAMES}
    ids = test["ID"].values
    uniq = np.unique(ids)
    idx_by = {u: np.where(ids == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    samples = [np.concatenate([idx_by[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(1000)]
    boot = {}
    for i, (n, c) in enumerate(zip(NAMES, cols)):
        y = test[c].values.astype(float)
        praw, pts = pt[:, i], sig(logit(pt[:, i]) / Ts[n])
        ppl = platt(pt[:, i], ABs[n])
        br = np.array([ece(y[s], praw[s]) for s in samples])
        bt = np.array([ece(y[s], pts[s]) for s in samples])
        bp = np.array([ece(y[s], ppl[s]) for s in samples])

        def q(a):
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        boot[n] = {"raw_ci95": q(br), "ts_ci95": q(bt), "platt_ci95": q(bp),
                   "diff_ts_minus_raw_ci95": q(bt - br), "diff_platt_minus_raw_ci95": q(bp - br)}
    out["final"] = {"tag": ftag, "T": Ts, "platt_ab": ABs, "ece_bootstrap": boot}

    devices = {"n100_ort_cpu": "results/n100_run009/test_final/probs_ort_fp32_cpu.npy",
               "n100_ov_cpu": "results/n100_run009/test_final/probs_ov_fp32_cpu.npy",
               "n100_ov_igpu": "results/n100_run009/test_final/probs_ov_fp32_gpu.npy",
               "pi_tflite_fp32_t4": "results/pi_run009/test_final/probs_tfl_fp32_t4.npy"}
    dev = {}
    for k, path in devices.items():
        pd_ = np.load(path)
        dev[k] = {}
        for i, (n, c) in enumerate(zip(NAMES, cols)):
            y = test[c].values.astype(float)
            ref = platt(pt[:, i], ABs[n])
            cal = platt(pd_[:, i], ABs[n])
            dev[k][n] = {"ece_platt": ece(y, cal), "ece_platt_reference": ece(y, ref),
                         "max_abs_calibrated_prob_diff": float(np.abs(cal - ref).max())}
    out["device_check"] = dev

    # INT8 drift on calibration (development: 224 px run002, validation set, Pi run 006)
    int8 = {}
    vcols = ["D", "G", "C", "A", "H"]
    pf = np.load("results/pi_run006/probs_tfl_fp32_t4.npy")
    for k in ["fp32_t4", "drq_t4", "int8_t4"]:
        pq = np.load(f"results/pi_run006/probs_tfl_{k}.npy")
        int8[k] = {}
        for i, (n, c) in enumerate(zip(NAMES, vcols)):
            y = val[c].values.astype(float)
            int8[k][n] = {"ece_raw": ece(y, pq[:, i]), "mean_abs_prob_shift_vs_fp32": float(np.abs(pq[:, i] - pf[:, i]).mean())}
    out["int8_dev_val"] = int8

    with open(f"{OUT}/calibration.json", "w") as fh:
        json.dump(out, fh, indent=1)

    # reliability diagram, final model, test set
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), sharey=True)
    for i, (n, c) in enumerate(zip(NAMES, cols)):
        ax = axes[i]
        y = test[c].values.astype(float)
        ax.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
        reliability(ax, y, pt[:, i], "#c0392b", "uncalibrated")
        reliability(ax, y, sig(logit(pt[:, i]) / Ts[n]), "#e69f00", "temperature")
        reliability(ax, y, platt(pt[:, i], ABs[n]), "#1f4e9c", "Platt")
        r = out["per_model"][ftag][n]
        ax.set_title(f"{FULL[n]}\nECE {r['test_raw']['ece']:.3f} / {r['test_ts']['ece']:.3f} / {r['test_platt']['ece']:.3f}",
                     fontsize=8)
        ax.set_xlabel("Predicted probability", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
    axes[0].set_ylabel("Observed frequency", fontsize=8)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/run011_reliability.pdf")
    fig.savefig(f"{FIG}/run011_reliability.png", dpi=150)

    print(json.dumps({"final": out["final"],
                      "final_per_label": {n: {"T": round(out["per_model"][ftag][n]["T"], 3),
                                              "raw": {k: round(v, 4) for k, v in out["per_model"][ftag][n]["test_raw"].items()},
                                              "ts": {k: round(v, 4) for k, v in out["per_model"][ftag][n]["test_ts"].items()},
                                              "platt": {k: round(v, 4) for k, v in out["per_model"][ftag][n]["test_platt"].items()}}
                                          for n in NAMES},
                      "group_ece_mean_sd": out["group_ece_mean_sd"],
                      "device_check": dev, "int8_dev_val": int8}, indent=1))


if __name__ == "__main__":
    main()
