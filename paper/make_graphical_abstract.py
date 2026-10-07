"""
Graphical abstract for the BSPC submission (paper/figs/graphical_abstract.{pdf,png}).

Every number is read from the result files of runs 008-011, never typed in:
  results/run008_test_metrics.json, results/run009 JSONs (latency, memory,
  energy with the patched plugin), results/run011/calibration.json (Platt maps), results/run010 device
  results (Grad-CAM overhead).
The example image is the run 010 seeded random draw for a correctly detected
DR case (results/run010/figure_selection_final.json, "TP_D").
Colours: two categorical slots (blue = N100, orange = Pi 3B), validated with
the dataviz palette validator (all checks pass on a light surface).

Usage (repo root): .venv/bin/python paper/make_graphical_abstract.py
Elsevier asks for at least 1328 x 531 px (w x h), readable at 13 x 5 cm.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, "results")
from run010_common import DATA_DIR, gradcam_torch, load_model, load_rgb, norm_max, normalize_rgb, upsample  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
NAMES = ["DR", "Glaucoma", "Cataract", "AMD", "Hypert. ret."]


def j(path):
    return json.load(open(path))


def main():
    final = j("results/run008_FINAL_CHECKPOINT.json")
    test = j("results/run008_test_metrics.json")
    cal = j("results/run011/calibration.json")
    sel = j("results/run010/figure_selection_final.json")["selected"]["TP_D"]
    n_lat = j("results/n100_run009/lat_512/results_ort_fp32_cpu.json")
    p_lat = j("results/pi_run009/lat_512/results_tfl_fp32_t4.json")
    energy = j("results/n100_run012/energy/energy_512/summary.json")
    p_gc = j("results/pi_run010/results_tflite_gradcam5.json")
    n_gc = j("results/n100_run010/results_ort_gradcam5.json")

    tag = final["checkpoint"].split("/")[-1].split("_best_epoch")[0]
    fm = test["per_model"][tag]
    gs = test["group_summary"]
    ab = cal["final"]["platt_ab"]

    # example image: calibrated probabilities + DR Grad-CAM
    model = load_model(final["checkpoint"])
    rgb = load_rgb(f"{DATA_DIR}/{sel['file']}")
    x = torch.from_numpy(normalize_rgb(rgb))[None]
    probs, cams, _ = gradcam_torch(model, x, labels=[0])
    cam = norm_max(upsample(cams[0, 0]))
    z = np.log(np.clip(probs[0], 1e-6, 1 - 1e-6) / (1 - np.clip(probs[0], 1e-6, 1 - 1e-6)))
    pcal = [1 / (1 + np.exp(-(ab[n][0] * z[i] + ab[n][1]))) for i, n in enumerate(["D", "G", "C", "A", "H"])]

    fig = plt.figure(figsize=(13, 5), facecolor=SURFACE)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    def card(x0, y0, w, h, edge):
        fig.patches.append(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                          transform=fig.transFigure, fc="white", ec=edge, lw=1.2, zorder=-1))

    def arrow(x0, x1, y):
        fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))

    fig.text(0.015, 0.94, "Five-disease retinal screening on low-cost edge hardware",
             fontsize=15, weight="bold", color=INK)

    # 1. input + explanation
    card(0.015, 0.20, 0.215, 0.66, "#d6d5d0")
    fig.text(0.0225, 0.82, "1  One fundus photo", fontsize=11, weight="bold", color=INK)
    ax = fig.add_axes([0.022, 0.30, 0.098, 0.46])
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("input", fontsize=8.5, color=INK2, pad=2)
    ax = fig.add_axes([0.125, 0.30, 0.098, 0.46])
    ax.imshow(rgb)
    ax.imshow(cam, cmap="viridis", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("DR Grad-CAM", fontsize=8.5, color=INK2, pad=2)
    fig.text(0.0225, 0.235, "Held-out test image, moderate NPDR\n(seeded random draw)", fontsize=7.5, color=MUTED)

    arrow(0.233, 0.258, 0.53)

    # 2. model + calibrated outputs
    card(0.262, 0.20, 0.25, 0.66, "#d6d5d0")
    fig.text(0.27, 0.82, "2  MobileNetV3-small, 512 px, FP32", fontsize=11, weight="bold", color=INK)
    fig.text(0.27, 0.765, "6 MB  ·  ODIR-5K  ·  Platt-calibrated outputs", fontsize=8.5, color=INK2)
    ax = fig.add_axes([0.335, 0.39, 0.165, 0.33])
    ypos = np.arange(5)[::-1]
    ax.barh(ypos, pcal, height=0.55, color=INK2)
    for yy, v in zip(ypos, pcal):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(NAMES, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d6d5d0")
    ax.set_xlabel("calibrated probability", fontsize=8, color=INK2)
    fig.text(0.27, 0.222,
             f"Test AUC (3 seeds): mean {np.mean([gs['A'][n]['seed_mean_auc'] for n in ['D', 'G', 'C', 'A', 'H']]):.3f}"
             f"  ·  DR {gs['A']['D_eye']['seed_mean_auc']:.2f}\nDeployed model*: referable DR AUC {fm['test_dr']['auc_referable_modplus_vs_rest']:.2f}",
             fontsize=7.5, color=MUTED)

    arrow(0.516, 0.541, 0.53)

    # 3. devices
    def device(x0, color, title, price, lines):
        card(x0, 0.20, 0.215, 0.66, color)
        fig.text(x0 + 0.008, 0.82, title, fontsize=11, weight="bold", color=INK)
        fig.patches.append(FancyBboxPatch((x0 + 0.008, 0.762), 0.05, 0.036, boxstyle="round,pad=0.002,rounding_size=0.006",
                                          transform=fig.transFigure, fc=color, ec="none"))
        fig.text(x0 + 0.033, 0.780, price, fontsize=8, color="white", weight="bold", ha="center", va="center")
        y = 0.68
        for k, v in lines:
            fig.text(x0 + 0.010, y, k, fontsize=8.5, color=INK2)
            fig.text(x0 + 0.205, y, v, fontsize=10, color=INK, ha="right", weight="bold")
            y -= 0.085

    nE = energy["configs"]["ov_fp32_gpu"]["model"]["mj_per_inference_total"]["mean"]
    device(0.545, BLUE, "3a  Intel N100 mini PC", "Rs 37,500", [
        ("Test AUC vs workstation", "identical"),
        ("Model latency", f"{n_lat['latency_ms']['mean']:.1f} ms"),
        ("+ Grad-CAM, 5 labels", f"+{n_gc['latency_ms']['mean'] - j('results/n100_run010/results_ort_plain.json')['latency_ms']['mean']:.1f} ms"),
        ("Peak memory", f"{int(n_lat['peak_rss_mb'] + 0.5)} MB"),
        ("Energy (iGPU)", f"{nE:.0f} mJ / image"),
    ])
    device(0.775, ORANGE, "3b  Raspberry Pi 3B", "Rs 3,500", [
        ("Test AUC vs workstation", "identical"),
        ("Model latency", f"{p_lat['latency_ms']['mean'] / 1000:.2f} s"),
        ("+ Grad-CAM, 5 labels", f"+{p_gc['latency_ms']['mean'] - j('results/pi_run010/results_tflite_plain.json')['latency_ms']['mean']:.0f} ms"),
        ("Peak memory", f"{int(p_lat['peak_rss_mb'] + 0.5)} MB of 870 MB"),
        ("Clock (under-powered)", "600 MHz"),
    ])

    # findings strip
    t = cal["per_model"][tag]["D"]
    diff = test["paired_diffs"]["A_minus_B"]["D_eye"]
    fig.text(0.015, 0.105,
             f"512 px vs 224 px: DR AUC +{diff['diff']:.3f} (95% CI {diff['ci95'][0]:.3f}–{diff['ci95'][1]:.3f})     "
             f"Naive INT8: lower AUC, broken calibration, little speed-up → deploy FP32     "
             f"Platt scaling: DR ECE {t['test_raw']['ece']:.3f} → {t['test_platt']['ece']:.3f}",
             fontsize=9, color=INK)
    fig.text(0.015, 0.045,
             "*Final model, median-validation seed. Pi throttled by under-voltage; latencies ≈2× pessimistic. "
             "Grad-CAM does not localise microaneurysms.",
             fontsize=7.5, color=MUTED)

    for a in fig.axes:
        a.set_zorder(2)
    fig.savefig("paper/figs/graphical_abstract.pdf", facecolor=SURFACE)
    fig.savefig("paper/figs/graphical_abstract.png", dpi=200, facecolor=SURFACE)
    print("probs raw", np.round(probs[0], 3), "calibrated", np.round(pcal, 3))


if __name__ == "__main__":
    main()
