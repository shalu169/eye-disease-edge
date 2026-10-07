"""
Run 010 step 3: Adebayo et al. (2018) sanity checks for the Grad-CAM maps.

(a) Cascading model-parameter randomisation, top -> bottom. Stage k re-initialises
    everything from the classifier down to the k-th module group with the
    architecture's own random initialisation (weights of a fresh
    timm.create_model("mobilenetv3_small_100", pretrained=False) built with a fixed
    seed; BatchNorm gets gamma=1, beta=0, running mean 0 / var 1). Order:
      classifier, conv_head, blocks[5] (1x1 ConvBnAct 96->576 = the Grad-CAM
      layer), blocks[4], blocks[3], blocks[2], blocks[1], blocks[0], stem (conv_stem+bn1)
(b) Data (label) randomisation, if a model trained on permuted labels is given
    (--randlabel-ckpt, see results/run010_train_randlabels.py).
Similarity of each randomised map to the original map of the same image and
label: Spearman rank correlation and SSIM (Gaussian window sigma 1.5, data range 1)
on the max-normalised 512x512 maps. Maps that do not change fail the check.
Reference levels reported alongside:
  - original map vs the original map of a *different* image (same label):
    the similarity explained by generic fundus layout alone;
  - original map vs the centre-bias map;
  - original D map vs the other labels' maps of the same image (class specificity).
Usage: .venv/bin/python results/run010_sanity.py [--ckpt auto|final|path] [--n 100] [--randlabel-ckpt PATH]
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import timm
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

sys.path.insert(0, "results")
from run010_common import (DATA_DIR, NAMES, OUT, centre_bias, ckpt_from_args, gradcam_torch,  # noqa: E402
                           load_manifest, load_model, load_x, norm_max, upsample)

CASCADE = [("classifier", ["classifier"]), ("conv_head", ["conv_head"]), ("blocks.5", ["blocks.5"]),
           ("blocks.4", ["blocks.4"]), ("blocks.3", ["blocks.3"]), ("blocks.2", ["blocks.2"]),
           ("blocks.1", ["blocks.1"]), ("blocks.0", ["blocks.0"]), ("stem", ["conv_stem", "bn1"])]


def ssim(a, b, sigma=1.5, L=1.0):
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu_a, mu_b = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    saa = gaussian_filter(a * a, sigma) - mu_a ** 2
    sbb = gaussian_filter(b * b, sigma) - mu_b ** 2
    sab = gaussian_filter(a * b, sigma) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(s.mean())


def rho(a, b):
    if a.max() == a.min() or b.max() == b.min():
        return float("nan")
    return float(spearmanr(a[::4, ::4].ravel(), b[::4, ::4].ravel())[0])


def maps_for(model, xs):
    out, probs = [], []
    for x in xs:
        p, lr, _ = gradcam_torch(model, torch.from_numpy(x)[None])
        out.append(norm_max(upsample(lr[0])))
        probs.append(p[0])
    return np.stack(out), np.stack(probs)  # [N,5,S,S], [N,5]


def summarise(orig, new):
    """Per label: median/IQR Spearman and SSIM (NaN = constant map, counted separately)."""
    res = {}
    for c, name in enumerate(NAMES):
        r = np.array([rho(orig[i, c], new[i, c]) for i in range(len(orig))])
        s = np.array([ssim(orig[i, c], new[i, c]) for i in range(len(orig))])
        zero = int(sum(new[i, c].max() == 0 for i in range(len(orig))))
        res[name] = {"spearman_median": float(np.nanmedian(r)) if np.isfinite(r).any() else None,
                     "spearman_q25": float(np.nanpercentile(r, 25)) if np.isfinite(r).any() else None,
                     "spearman_q75": float(np.nanpercentile(r, 75)) if np.isfinite(r).any() else None,
                     "spearman_mean_abs": float(np.nanmean(np.abs(r))) if np.isfinite(r).any() else None,
                     "ssim_median": float(np.median(s)), "ssim_q25": float(np.percentile(s, 25)),
                     "ssim_q75": float(np.percentile(s, 75)),
                     "n_all_zero_maps": zero, "n_constant_for_spearman": int(np.isnan(r).sum())}
    allr = np.array([rho(orig[i, c], new[i, c]) for i in range(len(orig)) for c in range(5)])
    alls = np.array([ssim(orig[i, c], new[i, c]) for i in range(len(orig)) for c in range(5)])
    res["all"] = {"spearman_median": float(np.nanmedian(allr)) if np.isfinite(allr).any() else None,
                  "ssim_median": float(np.median(alls))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randlabel-ckpt", default=None)
    a = ap.parse_args()
    torch.set_num_threads(6)
    t0 = time.time()
    ckpt, info = ckpt_from_args(a.ckpt)
    model = load_model(ckpt)
    files = load_manifest("test").sample(n=a.n, random_state=20)["filename"].tolist()
    xs = [load_x(f"{DATA_DIR}/{f}") for f in files]
    orig, p_orig = maps_for(model, xs)
    out = {"checkpoint": ckpt, "checkpoint_info": info, "n_images": len(files),
           "images": "seeded random draw from run008 test split (random_state=20), all 5 label maps per image",
           "similarity": "Spearman on 128x128 subsample of the 512 maps; SSIM sigma=1.5, data_range=1 on 512 maps",
           "randomisation_init": "fresh timm mobilenetv3_small_100 pretrained=False, torch.manual_seed(seed)",
           "seed": a.seed}
    # reference levels
    perm = np.roll(np.arange(len(files)), 1)
    out["reference_different_image_same_label"] = summarise(orig, orig[perm])
    cb = np.broadcast_to(centre_bias()[None, None], orig.shape)
    out["reference_centre_bias"] = summarise(orig, cb)
    cls = {}
    for c, name in enumerate(NAMES[1:], start=1):
        r = [rho(orig[i, 0], orig[i, c]) for i in range(len(files))]
        s = [ssim(orig[i, 0], orig[i, c]) for i in range(len(files))]
        cls[f"D_vs_{name}"] = {"spearman_median": float(np.nanmedian(r)), "ssim_median": float(np.median(s))}
    pair_r = [rho(orig[i, c1], orig[i, c2]) for i in range(len(files)) for c1 in range(5) for c2 in range(c1 + 1, 5)]
    cls["all_label_pairs_spearman_median"] = float(np.nanmedian(pair_r))
    cls["all_label_pairs_spearman_q25"] = float(np.nanpercentile(pair_r, 25))
    cls["all_label_pairs_spearman_q75"] = float(np.nanpercentile(pair_r, 75))
    out["class_specificity_same_image_other_label"] = cls

    # (a) cascading randomisation
    torch.manual_seed(a.seed)
    fresh = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=5).eval()
    fsd = fresh.state_dict()
    m = copy.deepcopy(model)
    casc = []
    for name, prefixes in CASCADE:
        sd = m.state_dict()
        n_rep = 0
        for k in sd:
            if any(k == p or k.startswith(p + ".") for p in prefixes):
                sd[k] = fsd[k].clone()
                n_rep += 1
        m.load_state_dict(sd)
        new, p_new = maps_for(m, xs)
        casc.append({"randomised_down_to": name, "n_tensors_reinitialised": n_rep,
                     "mean_abs_prob_change": float(np.abs(p_new - p_orig).mean()),
                     **summarise(orig, new)})
        print(name, json.dumps(casc[-1]["all"]), f"{time.time() - t0:.0f}s", flush=True)
    out["cascading_randomisation"] = casc
    # independent randomisation of the Grad-CAM layer only (blocks[5]), rest trained
    m = copy.deepcopy(model)
    sd = m.state_dict()
    for k in sd:
        if k.startswith("blocks.5."):
            sd[k] = fsd[k].clone()
    m.load_state_dict(sd)
    new, p_new = maps_for(m, xs)
    out["independent_randomisation_blocks5_only"] = {"mean_abs_prob_change": float(np.abs(p_new - p_orig).mean()),
                                                      **summarise(orig, new)}
    # (b) data randomisation
    if a.randlabel_ckpt:
        mr = load_model(a.randlabel_ckpt)
        new, p_new = maps_for(mr, xs)
        out["data_randomisation"] = {"checkpoint": a.randlabel_ckpt, **summarise(orig, new)}
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{OUT}/sanity_{'final' if 'run008' in ckpt else 'dev'}.json", "w"), indent=1)
    np.save(f"{OUT}/sanity_files_{'final' if 'run008' in ckpt else 'dev'}.npy", np.array(files))
    print(json.dumps({k: v for k, v in out.items() if k != "cascading_randomisation"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
