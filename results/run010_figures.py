"""
Run 010 step 7: paper figures (PDF, vector except the embedded image rasters,
plus PNG previews) under paper/figs/run010_*.

Fig 1  run010_gradcam_examples : Grad-CAM on run008 TEST images.
       One correct positive (TP) per disease, plus failure cases: one DR false
       positive, one false positive of another disease, one mild-NPDR miss.
       Operating threshold per label = the largest threshold reaching 90 %
       sensitivity on VALIDATION (run008_common.thr_at_sens), never the test set.
       Selection is NOT curated: each example is a seeded random draw
       (numpy default_rng(2026)) from its category; the list of candidates per
       category and the chosen files are written to
       results/run010/figure_selection_<tag>.json.
Fig 2  run010_idrid_overlays : 4 IDRiD images (seeded random draw, rng 2027)
       with expert lesion masks and the DR Grad-CAM.
Fig 3  run010_deletion_insertion : mean deletion/insertion curves per label
       (blur fill) for Grad-CAM vs random vs centre baselines.
Fig 4  run010_sanity_cascade : Spearman similarity to the original map under
       cascading parameter randomisation, with the different-image floor.
Colour map: viridis (perceptually uniform, CVD-safe). Grad-CAM shows positive
evidence for the named label, scaled to the per-map maximum: 0 (dark purple) =
no positive contribution, 1 (yellow) = strongest contribution in that image.
Usage: .venv/bin/python results/run010_figures.py [--ckpt auto|final|path]
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, "results")
from run008_common import thr_at_sens  # noqa: E402
from run010_common import (DATA_DIR, NAMES, OUT, POS_COL, ckpt_from_args, gradcam_torch, load_manifest,  # noqa: E402
                           load_model, load_rgb, load_x, norm_max, normalize_rgb, upsample)

FIG = "paper/figs"
FULL = {"D": "DR", "G": "Glaucoma", "C": "Cataract", "A": "AMD", "H": "Hypert. retinopathy"}
GRADE = {0: "no DR", 1: "mild NPDR", 2: "moderate NPDR", 3: "severe NPDR", 4: "PDR", -1: "DR, ungraded"}
SERIES = {"gradcam": ("#2a78d6", "-", "Grad-CAM"), "random": ("#eb6834", "--", "Random (smooth)"),
          "centre": ("#1baf7a", ":", "Centre bias")}
LES = {"MA": ("#CC79A7", "Microaneurysms"), "HE": ("#D55E00", "Haemorrhages"),
       "EX": ("#F0E442", "Hard exudates"), "SE": ("#56B4E9", "Soft exudates")}
CMAP = "viridis"
CBAR_LABEL = "Grad-CAM evidence for the named label\n(0 = none, 1 = image maximum)"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8, "pdf.fonttype": 42})


def probs_for(model, files, cache):
    if os.path.exists(cache):
        return np.load(cache)
    out = []
    with torch.no_grad():
        for i in range(0, len(files), 16):
            x = torch.from_numpy(np.stack([load_x(f"{DATA_DIR}/{f}") for f in files[i:i + 16]]))
            out.append(torch.sigmoid(model(x)).numpy())
    p = np.concatenate(out)
    np.save(cache, p)
    return p


def overlay(ax, rgb, m, alpha=0.5):
    ax.imshow(rgb)
    im = ax.imshow(m, cmap=CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_examples(model, tag):
    te, va = load_manifest("test"), load_manifest("val")
    pt = probs_for(model, te["filename"].tolist(), f"{OUT}/probs_{tag}_test.npy")
    pv = probs_for(model, va["filename"].tolist(), f"{OUT}/probs_{tag}_val.npy")
    thr = {n: thr_at_sens(va[POS_COL[n]].values, pv[:, i], 0.90) for i, n in enumerate(NAMES)}
    rng = np.random.default_rng(2026)
    cats, sel = [], {}
    for i, n in enumerate(NAMES):
        y = te[POS_COL[n]].values == 1
        cats.append((f"TP_{n}", i, np.where(y & (pt[:, i] >= thr[n]))[0]))
    yD = te["D_eye"].values == 1
    cats.append(("FP_D", 0, np.where(~yD & (pt[:, 0] >= thr["D"]) & (te["D"].values == 0))[0]))
    # false positive of another disease: pooled over G/C/A/H, label drawn first then image
    fp_other = [(i, j) for i, n in enumerate(NAMES[1:], 1)
                for j in np.where((te[n].values == 0) & (pt[:, i] >= thr[n]))[0]]
    cats.append(("FP_other", None, fp_other))
    cats.append(("FN_mildDR", 0, np.where((te["dr_grade_eye"].values == 1) & (pt[:, 0] < thr["D"]))[0]))
    chosen = []
    for name, i, cand in cats:
        if name == "FP_other":
            k = int(rng.integers(len(cand)))
            i, j = cand[k]
            ncand = len(cand)
        else:
            j = int(cand[rng.integers(len(cand))])
            ncand = len(cand)
        r = te.iloc[j]
        chosen.append((name, i, j))
        sel[name] = {"file": r["filename"], "label": NAMES[i], "p": float(pt[j, i]), "thr": thr[NAMES[i]],
                     "n_candidates": ncand, "dr_grade_eye": int(r["dr_grade_eye"]),
                     "true_labels": {n: int(r[POS_COL[n]]) for n in NAMES}}
    fig, axes = plt.subplots(2, len(chosen), figsize=(1.45 * len(chosen) + 0.6, 3.4), gridspec_kw={"hspace": 0.12, "wspace": 0.06})
    for col, (name, i, j) in enumerate(chosen):
        f = te.iloc[j]["filename"]
        rgb = load_rgb(f"{DATA_DIR}/{f}")
        _, lr, _ = gradcam_torch(model, torch.from_numpy(normalize_rgb(rgb))[None], labels=[i])
        m = norm_max(upsample(lr[0, 0]))
        axes[0, col].imshow(rgb)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        kind = {"TP": "correct positive", "FP": "false positive", "FN": "missed"}[name[:2]]
        truth = [FULL[n] for n in NAMES if sel[name]["true_labels"][n]]
        extra = f"\n{GRADE[sel[name]['dr_grade_eye']]}" if NAMES[i] == "D" and name != "FP_D" else "\n"
        axes[0, col].set_title(f"{FULL[NAMES[i]]}\n{kind}{extra}", fontsize=7)
        im = overlay(axes[1, col], rgb, m)
        axes[1, col].set_xlabel(f"p={sel[name]['p']:.2g} (thr {sel[name]['thr']:.2g})\n"
                                f"truth: {', '.join(truth) if truth else 'none of 5'}", fontsize=6)
    cb = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    cb.set_label(CBAR_LABEL, fontsize=6)
    cb.ax.tick_params(labelsize=6)
    axes[0, 0].set_ylabel("input", fontsize=7)
    axes[1, 0].set_ylabel("Grad-CAM of\nthe named label", fontsize=7)
    save(fig, "run010_gradcam_examples")
    json.dump({"thresholds_val_sens90": thr, "seed": 2026, "rule": "uniform random draw within each category",
               "selected": sel}, open(f"{OUT}/figure_selection_{tag}.json", "w"), indent=1)


def fig_idrid(model, tag):
    import pandas as pd
    from scipy.ndimage import binary_erosion
    man = pd.read_csv("results/run010/idrid_manifest.csv")
    ids = sorted(np.random.default_rng(2027).choice(man["id"].values, 4, replace=False).tolist())
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for col, iid in enumerate(ids):
        rgb = load_rgb(f"data/idrid/prep512/{iid}.png")
        mz = np.load(f"data/idrid/prep512/{iid}_masks.npz")
        p, lr, _ = gradcam_torch(model, torch.from_numpy(normalize_rgb(rgb))[None], labels=[0])
        m = norm_max(upsample(lr[0, 0]))
        axes[0, col].imshow(rgb)
        for t, (c, _) in LES.items():
            mk = mz[t].astype(bool)
            axes[0, col].imshow(np.ma.masked_where(~mk, mk), cmap=ListedColormap([c]), alpha=1.0,
                                interpolation="nearest")
        axes[0, col].set_title(f"{iid}  p(DR)={p[0, 0]:.2f}", fontsize=7)
        im = overlay(axes[1, col], rgb, m, alpha=0.55)
        anym = np.any([mz[t].astype(bool) for t in LES], axis=0)
        edge = anym & ~binary_erosion(anym)
        axes[1, col].imshow(np.ma.masked_where(~edge, edge), cmap=ListedColormap(["white"]), interpolation="nearest")
        yx = np.unravel_index(np.argmax(m), m.shape)
        axes[1, col].plot(yx[1], yx[0], marker="x", color="#ff3030", ms=7, mew=2)
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("expert lesion masks", fontsize=7)
    axes[1, 0].set_ylabel("DR Grad-CAM\n(white = lesion outline,\nx = map maximum)", fontsize=7)
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=l) for c, l in LES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.45, -0.02))
    cb = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    cb.set_label("Grad-CAM evidence for DR\n(0 = none, 1 = image maximum)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    save(fig, "run010_idrid_overlays")
    json.dump({"seed": 2027, "rule": "uniform random draw of 4 of the 81 IDRiD images", "ids": ids},
              open(f"{OUT}/figure_idrid_selection_{tag}.json", "w"), indent=1)


def fig_curves(tag):
    p = f"{OUT}/faithfulness_{tag}_curves.npz"
    if not os.path.exists(p):
        return
    cv = np.load(p)
    fr = np.array(json.load(open(f"{OUT}/faithfulness_{tag}.json"))["fractions"])
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for col, n in enumerate(NAMES):
        for row, kind in enumerate(["del", "ins"]):
            ax = axes[row, col]
            for meth, (c, ls, lab) in SERIES.items():
                y = cv[f"{n}|{meth}|blur|{kind}"]
                ax.plot(fr, y.mean(0), color=c, ls=ls, lw=2, label=lab)
            ax.grid(alpha=0.25, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"{FULL[n]} (n={len(cv[f'{n}|gradcam|blur|del'])})", fontsize=7)
        axes[0, 0].set_ylabel("deletion\nmean p(label)", fontsize=7)
        axes[1, 0].set_ylabel("insertion\nmean p(label)", fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("fraction of pixels", fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.11))
    save(fig, "run010_deletion_insertion")


def fig_sanity(tag):
    p = f"{OUT}/sanity_{tag}.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    steps = ["original"] + [s["randomised_down_to"] for s in d["cascading_randomisation"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, (n, c) in enumerate(zip(NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"])):
        y = [1.0] + [s[n]["spearman_median"] for s in d["cascading_randomisation"]]
        ax.plot(range(len(steps)), y, marker="o", ms=3, lw=1.5, color=c, label=FULL[n])
    fl = d["reference_different_image_same_label"]["all"]["spearman_median"]
    ax.axhline(fl, color="#555555", lw=1, ls="--")
    ax.text(len(steps) - 1, fl + 0.03, "different image, same label", ha="right", fontsize=6, color="#333333")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Spearman ρ vs original map\n(median over images)", fontsize=7)
    ax.set_xlabel("re-initialised from the top down to", fontsize=7)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    save(fig, "run010_sanity_cascade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--only", default="all")
    a = ap.parse_args()
    torch.set_num_threads(6)
    ckpt, _ = ckpt_from_args(a.ckpt)
    tag = "final" if "run008" in ckpt else "dev"
    model = load_model(ckpt)
    if a.only in ("all", "examples"):
        fig_examples(model, tag)
    if a.only in ("all", "idrid"):
        fig_idrid(model, tag)
    if a.only in ("all", "curves"):
        fig_curves(tag)
    if a.only in ("all", "sanity"):
        fig_sanity(tag)


if __name__ == "__main__":
    main()
