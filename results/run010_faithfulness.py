"""
Run 010 step 4: faithfulness of Grad-CAM on the run008 TEST split via deletion
and insertion curves (Petsiuk et al. 2018, RISE), per label, on the label's
positive test images (D: per-eye DR label; G/C/A/H: ODIR labels).

For a saliency map s (512x512) pixels are ranked by s (descending, ties broken
by a seeded jitter of 1e-6 * U(0,1) relative to the map range). At fractions
f = 0, 1/32, ..., 1 of the image:
  deletion : the top-f pixels of the image are replaced by the baseline image,
  insertion: start from the baseline image and put back the top-f original pixels,
and the target label's sigmoid probability is recorded. AUC = trapezoid area
under p(f) on f in [0,1]. Deletion: lower = more faithful; insertion: higher.
Baselines (perturbation fill): 'blur' = Gaussian blur sigma 10 px of the
image (primary, as RISE), 'mean' = ImageNet mean colour (0 after normalisation).
Saliency methods compared:
  gradcam : class-specific Grad-CAM of the target label (run010_common)
  random  : U(0,1) on the 16x16 grid, bilinear-upsampled (same smoothness as Grad-CAM),
            one draw per image (seeded)
  centre  : isotropic Gaussian centred on the image (the fundus disc centre), sigma 0.25*512
Whole image ranked (incl. the black corners outside the disc).
Usage: .venv-run010/bin/python results/run010_faithfulness.py --ckpt auto|final|path --onnx edge/artifacts/run010_<tag>_full_fp32.onnx [--max-per-label N]
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

sys.path.insert(0, "results")
from run010_common import (DATA_DIR, IMG, NAMES, OUT, POS_COL, centre_bias, ckpt_from_args,  # noqa: E402
                           gradcam_torch, load_manifest, load_model, load_rgb, normalize_rgb, random_smooth_map,
                           upsample)

FR = np.linspace(0, 1, 33)
METHODS = ["gradcam", "random", "centre"]
FILLS = ["blur", "mean"]


def order_of(sal, rng):
    s = sal.astype(np.float64).ravel()
    rngv = s.max() - s.min()
    s = s + rng.random(s.size) * 1e-6 * (rngv if rngv > 0 else 1.0)
    rank = np.empty(s.size, dtype=np.int64)
    rank[np.argsort(-s, kind="stable")] = np.arange(s.size)
    return rank.reshape(sal.shape)


def curves(sess, x, base, rank):
    """Deletion and insertion prob curves [len(FR), 5]. Forward passes use the
    exported full-model ONNX (ONNX Runtime CPU, parity with torch ~1e-5 in prob,
    see parity_onnx_*.json) because it is ~4x faster than torch CPU here."""
    n = rank.size
    ks = np.round(FR * n).astype(np.int64)
    out = {"del": [], "ins": []}
    for k in ks:
        m = (rank < k)[None]  # True = perturbed (deletion) / inserted (insertion)
        for kind, img in (("del", np.where(m, base, x)), ("ins", np.where(m, x, base))):
            lg = sess.run(None, {"image": img[None].astype(np.float32)})[0][0]
            out[kind].append(1 / (1 + np.exp(-lg.astype(np.float64))))
    return {k: np.stack(v) for k, v in out.items()}


def auc(y):
    return float(np.trapezoid(y, FR)) if hasattr(np, "trapezoid") else float(np.trapz(y, FR))


def boot_ci(v, rng, B=2000):
    v = np.asarray(v)
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--max-per-label", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--onnx", required=True, help="full-model ONNX exported from the same checkpoint")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    ckpt, info = ckpt_from_args(a.ckpt)
    model = load_model(ckpt)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    with torch.no_grad():  # guard: ONNX must be this checkpoint
        xx = np.random.default_rng(0).standard_normal((1, 3, IMG, IMG)).astype(np.float32)
        dd = np.abs(sess.run(None, {"image": xx})[0] - model(torch.from_numpy(xx)).numpy()).max()
    assert dd < 1e-3, f"ONNX does not match checkpoint (max logit diff {dd})"
    df = load_manifest("test")
    tasks = {}  # filename -> list of label indices positive
    for c, name in enumerate(NAMES):
        pos = df[df[POS_COL[name]] == 1]["filename"].tolist()
        if a.max_per_label:
            pos = pos[: a.max_per_label]
        for f in pos:
            tasks.setdefault(f, []).append(c)
    cb = centre_bias()
    rec = []  # one record per (image, label)
    allcurves = {}
    for ii, f in enumerate(sorted(tasks)):
        rng = np.random.default_rng(ii + 1000)
        rgb = load_rgb(f"{DATA_DIR}/{f}")
        x = normalize_rgb(rgb)
        bases = {"blur": normalize_rgb(np.stack([gaussian_filter(rgb[..., k].astype(np.float32), 10)
                                                  for k in range(3)], -1).clip(0, 255)),
                 "mean": np.zeros_like(x)}
        p0, lr, _ = gradcam_torch(model, torch.from_numpy(x)[None])
        gmaps = upsample(lr[0])
        rmap = random_smooth_map(rng)
        shared = {}
        for fill in FILLS:
            shared[("random", fill)] = curves(sess, x, bases[fill], order_of(rmap, rng))
            shared[("centre", fill)] = curves(sess, x, bases[fill], order_of(cb, rng))
        for c in tasks[f]:
            r = {"file": f, "label": NAMES[c], "p_orig": float(p0[0, c]),
                 "gradcam_all_zero": bool(gmaps[c].max() <= 0)}
            for fill in FILLS:
                cg = curves(sess, x, bases[fill], order_of(gmaps[c], rng))
                for meth, cv in [("gradcam", cg), ("random", shared[("random", fill)]),
                                 ("centre", shared[("centre", fill)])]:
                    for kind in ["del", "ins"]:
                        y = cv[kind][:, c]
                        r[f"{meth}_{fill}_{kind}_auc"] = auc(y)
                        allcurves.setdefault((NAMES[c], meth, fill, kind), []).append(y)
            rec.append(r)
        if ii % 25 == 0:
            print(f"{ii}/{len(tasks)} images, {len(rec)} pairs, {time.time() - t0:.0f}s", flush=True)

    # summary
    rng = np.random.default_rng(0)
    summ = {}
    for name in NAMES + ["all"]:
        rr = [r for r in rec if name == "all" or r["label"] == name]
        if not rr:
            continue
        s = {"n": len(rr), "mean_p_orig": float(np.mean([r["p_orig"] for r in rr])),
             "n_gradcam_all_zero": int(sum(r["gradcam_all_zero"] for r in rr))}
        for fill in FILLS:
            for kind in ["del", "ins"]:
                for meth in METHODS:
                    v = [r[f"{meth}_{fill}_{kind}_auc"] for r in rr]
                    s[f"{meth}_{fill}_{kind}_auc_mean"] = float(np.mean(v))
                    s[f"{meth}_{fill}_{kind}_auc_ci95"] = boot_ci(v, rng)
                g = np.array([r[f"gradcam_{fill}_{kind}_auc"] for r in rr])
                for meth in ["random", "centre"]:
                    o = np.array([r[f"{meth}_{fill}_{kind}_auc"] for r in rr])
                    better = (g < o) if kind == "del" else (g > o)
                    d = g - o
                    s[f"gradcam_minus_{meth}_{fill}_{kind}_mean"] = float(d.mean())
                    s[f"gradcam_minus_{meth}_{fill}_{kind}_ci95"] = boot_ci(d, rng)
                    s[f"gradcam_better_than_{meth}_{fill}_{kind}_frac"] = float(better.mean())
                    s[f"wilcoxon_p_vs_{meth}_{fill}_{kind}"] = float(wilcoxon(g, o).pvalue) if len(g) > 5 else None
        summ[name] = s
    out = {"checkpoint": ckpt, "checkpoint_info": info, "onnx": a.onnx, "onnx_vs_torch_max_logit_diff": float(dd), "split": "run008 test (results/run008_splits/test.csv)",
           "positives": {n: POS_COL[n] for n in NAMES}, "fractions": FR.tolist(),
           "fills": {"blur": "Gaussian sigma 10 px per channel", "mean": "ImageNet mean colour"},
           "random_baseline": "U(0,1) 16x16 bilinear-upsampled, 1 draw/image",
           "centre_baseline": "Gaussian centred, sigma 0.25*512", "summary": summ,
           "seconds": round(time.time() - t0, 1)}
    tag = "final" if "run008" in ckpt else "dev"
    json.dump(out, open(f"{OUT}/faithfulness_{tag}.json", "w"), indent=1)
    json.dump(rec, open(f"{OUT}/faithfulness_{tag}_per_image.json", "w"))
    np.savez_compressed(f"{OUT}/faithfulness_{tag}_curves.npz",
                        **{"|".join(k): np.stack(v) for k, v in allcurves.items()})
    print(json.dumps(summ["all"], indent=1))


if __name__ == "__main__":
    main()
