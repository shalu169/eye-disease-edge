"""
Run 010 step 5b: does the DR Grad-CAM map point at expert-annotated DR lesions?
Data: the 81 IDRiD segmentation images, preprocessed to 512 px by
results/run010_idrid_prep.py (masks: MA microaneurysms, HE haemorrhages, EX hard
exudates, SE soft exudates; OD optic disc is used only as a non-lesion
reference). All 81 images contain DR lesions.

Saliency maps (512x512, evaluated inside the field of view FOV = grey > 15):
  gradcam_D : Grad-CAM of the DR output (the map under test)
  gradcam_G / _C / _A / _H : Grad-CAM of the other outputs (wrong-class controls)
  random    : U(0,1) 16x16 grid bilinear-upsampled; metrics averaged over R=50 draws per image
  centre    : isotropic Gaussian at the image centre, sigma 0.25*512
Metrics per image, for the union of the 4 lesion types and per type (only
images where that type is present):
  pointing game (PG): saliency argmax inside the lesion mask (strict), and
     within 15 px of it (mask dilated by a 15 px disc, Zhang et al. 2018)
  energy-based pointing game (EPG): sum(s * mask) / sum(s) over the FOV
     (Wang et al. 2020, Score-CAM); chance level for a uniform map = lesion
     area fraction of the FOV (reported as 'area_frac')
  pixel AUROC and average precision (AP) of s vs the mask, FOV pixels
     (every 2nd row/column, 65k pixels/image)
Also: fraction of Grad-CAM energy on the optic disc, and model DR probability.
Usage: .venv/bin/python results/run010_localisation.py [--ckpt auto|final|path]
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "results")
from run010_common import (IMG, NAMES, OUT, centre_bias, ckpt_from_args, gradcam_torch, load_model,  # noqa: E402
                           load_rgb, normalize_rgb, random_smooth_map, upsample)

PREP = "data/idrid/prep512"
TYPES = ["MA", "HE", "EX", "SE"]
R_RANDOM = 50
TOL = 15


def disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r


DISC = disc(TOL)


def metrics(s, mask, mask_tol, fov, sub):
    """s: saliency [S,S] >= 0; returns dict of PG/EPG/AUROC/AP inside FOV."""
    sf = np.where(fov, s, -np.inf)
    am = np.unravel_index(np.argmax(sf), s.shape)
    pos = s[fov].clip(min=0)
    tot = pos.sum()
    y = mask[sub][fov[sub]]
    v = s[sub][fov[sub]]
    out = {"pg": float(mask[am]), "pg_tol": float(mask_tol[am]),
           "epg": float((s.clip(min=0) * mask * fov).sum() / tot) if tot > 0 else float("nan")}
    if y.min() != y.max():
        out["auroc"] = float(roc_auc_score(y, v))
        out["ap"] = float(average_precision_score(y, v))
    else:
        out["auroc"] = out["ap"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    a = ap.parse_args()
    torch.set_num_threads(6)
    t0 = time.time()
    ckpt, info = ckpt_from_args(a.ckpt)
    model = load_model(ckpt)
    man = pd.read_csv("results/run010/idrid_manifest.csv")
    cb = centre_bias()
    sub = (slice(None, None, 2), slice(None, None, 2))
    rows = []
    for k, iid in enumerate(man["id"]):
        rgb = load_rgb(f"{PREP}/{iid}.png")
        fov = rgb.astype(np.float32).mean(-1) > 15
        mz = np.load(f"{PREP}/{iid}_masks.npz")
        masks = {t: mz[t].astype(bool) & fov for t in TYPES}
        masks["any"] = np.any([masks[t] for t in TYPES], axis=0)
        od = mz["OD"].astype(bool) & fov
        p, lr, _ = gradcam_torch(model, torch.from_numpy(normalize_rgb(rgb))[None])
        g = upsample(lr[0])  # [5,S,S] raw (unnormalised; all metrics are scale-invariant)
        rng = np.random.default_rng(k + 5000)
        rmaps = [random_smooth_map(rng) for _ in range(R_RANDOM)]
        r = {"id": iid, "split": man.loc[k, "split"], **{f"p_{n}": float(p[0, i]) for i, n in enumerate(NAMES)},
             "fov_px": int(fov.sum()), "gradcam_D_all_zero": bool(g[0].max() <= 0),
             "gradcam_D_energy_on_OD": float((g[0] * od).sum() / max(g[0][fov].sum(), 1e-12)),
             "centre_energy_on_OD": float((cb * od).sum() / cb[fov].sum()), "OD_area_frac": float(od.sum() / fov.sum())}
        for t in ["any"] + TYPES:
            m = masks[t]
            if m.sum() == 0:
                continue
            mt = binary_dilation(m, structure=DISC) & fov
            r[f"{t}_area_frac"] = float(m.sum() / fov.sum())
            r[f"{t}_area_frac_tol"] = float(mt.sum() / fov.sum())
            for meth, s in [("gradcam_D", g[0]), ("gradcam_G", g[1]), ("gradcam_C", g[2]), ("gradcam_A", g[3]),
                            ("gradcam_H", g[4]), ("centre", cb)]:
                for kk, v in metrics(s, m, mt, fov, sub).items():
                    r[f"{t}_{meth}_{kk}"] = v
            rm = [metrics(s, m, mt, fov, sub) for s in rmaps]
            for kk in rm[0]:
                r[f"{t}_random_{kk}"] = float(np.nanmean([d[kk] for d in rm]))
        rows.append(r)
        if k % 10 == 0:
            print(f"{k}/{len(man)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/localisation_{'final' if 'run008' in ckpt else 'dev'}_per_image.csv", index=False)

    rng = np.random.default_rng(0)

    def ci(v):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        bs = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    methods = ["gradcam_D", "random", "centre", "gradcam_G", "gradcam_C", "gradcam_A", "gradcam_H"]
    summ = {}
    for t in ["any"] + TYPES:
        cols = [c for c in df.columns if c.startswith(f"{t}_")]
        d = df[df[f"{t}_area_frac"].notna()] if f"{t}_area_frac" in df else df.iloc[:0]
        s = {"n_images": int(len(d)), "area_frac_mean": float(d[f"{t}_area_frac"].mean()),
             "area_frac_tol_mean": float(d[f"{t}_area_frac_tol"].mean())}
        for meth in methods:
            for kk in ["pg", "pg_tol", "epg", "auroc", "ap"]:
                v = d[f"{t}_{meth}_{kk}"].values.astype(float)
                s[f"{meth}_{kk}_mean"] = float(np.nanmean(v))
                s[f"{meth}_{kk}_ci95"] = ci(v)
        for other in ["random", "centre", "gradcam_G"]:
            for kk in ["epg", "auroc", "ap"]:
                x, y = d[f"{t}_gradcam_D_{kk}"].values.astype(float), d[f"{t}_{other}_{kk}"].values.astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                s[f"gradcam_D_minus_{other}_{kk}_mean"] = float((x[ok] - y[ok]).mean())
                s[f"gradcam_D_minus_{other}_{kk}_ci95"] = ci(x[ok] - y[ok])
                s[f"wilcoxon_p_gradcam_D_vs_{other}_{kk}"] = float(wilcoxon(x[ok], y[ok]).pvalue)
        s["epg_gradcam_D_over_area_frac_median"] = float(np.median(d[f"{t}_gradcam_D_epg"] / d[f"{t}_area_frac"]))
        summ[t] = s
        del cols
    rho = spearmanr(df["p_D"], df["any_gradcam_D_epg"])
    out = {"checkpoint": ckpt, "checkpoint_info": info, "n_images": int(len(df)),
           "data": "IDRiD A. Segmentation, 54 train + 27 test images, preprocessed by results/run010_idrid_prep.py",
           "tolerance_px": TOL, "random_draws_per_image": R_RANDOM,
           "model_on_idrid": {"p_D_mean": float(df["p_D"].mean()), "p_D_median": float(df["p_D"].median()),
                              "frac_p_D_ge_0.5": float((df["p_D"] >= 0.5).mean()),
                              "n_gradcam_D_all_zero": int(df["gradcam_D_all_zero"].sum())},
           "gradcam_D_energy_on_OD_mean": float(df["gradcam_D_energy_on_OD"].mean()),
           "centre_energy_on_OD_mean": float(df["centre_energy_on_OD"].mean()),
           "OD_area_frac_mean": float(df["OD_area_frac"].mean()),
           "spearman_pD_vs_epg_any": {"rho": float(rho[0]), "p": float(rho[1])},
           "summary": summ, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(f"{OUT}/localisation_{'final' if 'run008' in ckpt else 'dev'}.json", "w"), indent=1)
    for t in ["any"] + TYPES:
        s = summ[t]
        print(t, s["n_images"], "area", round(s["area_frac_mean"], 4),
              " | ".join(f"{m}: PG {s[f'{m}_pg_mean']:.2f} PGtol {s[f'{m}_pg_tol_mean']:.2f} "
                         f"EPG {s[f'{m}_epg_mean']:.3f} AUROC {s[f'{m}_auroc_mean']:.3f}"
                         for m in ["gradcam_D", "random", "centre", "gradcam_G"]))
    print(json.dumps(out["model_on_idrid"]), out["gradcam_D_energy_on_OD_mean"], out["spearman_pD_vs_epg_any"])


if __name__ == "__main__":
    main()
