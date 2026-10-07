"""
Run 010: generate the markdown tables for results/run010_ENTRY.md directly
from the JSON outputs (no hand transcription).
Usage: .venv/bin/python results/run010_tables.py --tag dev|final [--edge-tag dev007c|final]
"""
import argparse
import glob
import json
import os

NAMES = ["D", "G", "C", "A", "H"]
L = "results/run010"


def f(x, n=3):
    return "–" if x is None else f"{x:.{n}f}"


def ci(v, n=3):
    return f"[{v[0]:.{n}f}, {v[1]:.{n}f}]"


def e(x):
    return f"{x:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--edge-tag", required=True)
    a = ap.parse_args()
    out = []
    # reference vs library
    p = f"{L}/reference_vs_pytorch_grad_cam_{a.tag}.json"
    if os.path.exists(p):
        r = json.load(open(p))
        out += ["### Reference Grad-CAM vs pytorch-grad-cam", "",
                f"{r['n_images']} test images x 5 labels = {r['n_maps']} maps. "
                f"Low-res (16x16, min-max) max |diff| {e(r['lowres_minmax_max_abs_diff_ours_vs_lib'])}; "
                f"512 px library-style post-processing max |diff| {e(r['final512_max_abs_diff_ours_libstyle_vs_lib'])}; "
                f"our display map vs library output max |diff| {e(r['final512_max_abs_diff_ours_display_vs_lib'])}, "
                f"min Spearman {r['final512_spearman_ours_display_vs_lib_min']:.10f}. "
                f"Grad-CAM forward vs plain forward prob max |diff| {e(r['max_abs_prob_diff_gradcam_forward_vs_plain_forward'])}.", ""]
    # parity
    rows = []
    for rt in ["onnx", "tflite"]:
        p = f"{L}/parity_{rt}_{a.edge_tag}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            rows.append(f"| {rt} ({r['runtime_version']}) | {r['n_images']} / {r['n_maps']} | "
                        f"{e(r['feature_map_max_abs_diff'])} ({e(r['feature_map_max_abs_diff_rel_to_feature_max'])} rel) | "
                        f"{e(r['prob_max_abs_diff_full_model_plain_inference'])} | {e(r['prob_max_abs_diff_edge_gradcam_path'])} | "
                        f"{e(r['prob_max_abs_diff_numpy_head_on_torch_features'])} | {e(r['heatmap_max_abs_diff_edge'])} | "
                        f"{e(r['heatmap_mean_of_per_map_max_abs_diff_edge'])} | {e(r['heatmap_max_abs_diff_numpy_head_on_torch_features'])} | "
                        f"{r['heatmap_spearman_edge_vs_ref_min']:.8f} | {r['heatmap_argmax_identical_fraction']:.3f} |")
    if rows:
        out += ["### Edge Grad-CAM parity vs PyTorch autograd (Mac)", "",
                "| runtime | images / maps | feature map max abs diff | prob max diff, plain full model | prob max diff, backbone + NumPy head | prob max diff, NumPy head on torch features | heatmap max abs diff | mean of per-map max diff | heatmap max diff, NumPy head on torch features | min Spearman | argmax identical |",
                "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    # sanity
    p = f"{L}/sanity_{a.tag}.json"
    if os.path.exists(p):
        s = json.load(open(p))
        out += ["### Sanity checks (Adebayo 2018): Spearman rho / SSIM of randomised vs original map",
                "", f"{s['n_images']} test images, all 5 labels. Median over images; per-label Spearman medians.", "",
                "| condition | mean abs Δp | Spearman (all) | SSIM (all) | D | G | C | A | H | all-zero maps |",
                "|---|---|---|---|---|---|---|---|---|---|"]

        def row(name, d, dp=None):
            return (f"| {name} | {f(dp)} | {f(d['all']['spearman_median'])} | {f(d['all']['ssim_median'])} | "
                    + " | ".join(f(d[n]["spearman_median"], 2) for n in NAMES)
                    + f" | {sum(d[n]['n_all_zero_maps'] for n in NAMES)} |")
        for c in s["cascading_randomisation"]:
            out.append(row(f"cascade → {c['randomised_down_to']}", c, c["mean_abs_prob_change"]))
        ir = s["independent_randomisation_blocks5_only"]
        out.append(row("independent: blocks.5 only", ir, ir["mean_abs_prob_change"]))
        if "data_randomisation" in s:
            out.append(row("data randomisation (permuted-label model)", s["data_randomisation"]))
        out.append(row("reference: different image, same label", s["reference_different_image_same_label"]))
        out.append(row("reference: centre-bias map", s["reference_centre_bias"]))
        cs = s["class_specificity_same_image_other_label"]
        out += ["", "Class specificity (same image, other label's map), Spearman median: "
                + ", ".join(f"{k} {v['spearman_median']:.3f}" for k, v in cs.items() if isinstance(v, dict))
                + f"; all label pairs median {cs['all_label_pairs_spearman_median']:.3f} "
                  f"[IQR {cs['all_label_pairs_spearman_q25']:.3f}, {cs['all_label_pairs_spearman_q75']:.3f}].", ""]
        rp = f"{L}/randlabels_train.json"
        if "data_randomisation" in s and os.path.exists(rp):
            rr = json.load(open(rp))
            out += ["Permuted-label model: train AUC on permuted labels "
                    + ", ".join(f"{k} {v:.3f}" for k, v in rr["train_auc_on_permuted_labels"].items())
                    + "; val AUC on true labels " + ", ".join(f"{k} {v:.3f}" for k, v in rr["val_auc_on_true_labels"].items()) + ".", ""]
    # faithfulness
    p = f"{L}/faithfulness_{a.tag}.json"
    if os.path.exists(p):
        F = json.load(open(p))["summary"]
        for fill in ["blur", "mean"]:
            out += [f"### Deletion / insertion AUC, {fill} fill (test positives; deletion lower = better, insertion higher = better)", "",
                    "| label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for n in NAMES + ["all"]:
                s = F[n]
                cells = [n, str(s["n"]), f(s["mean_p_orig"], 2)]
                for k in ["del", "ins"]:
                    cells += [f(s[f"gradcam_{fill}_{k}_auc_mean"]), f(s[f"random_{fill}_{k}_auc_mean"]),
                              f(s[f"centre_{fill}_{k}_auc_mean"])]
                    for m in ["random", "centre"]:
                        cells.append(f"{s[f'gradcam_minus_{m}_{fill}_{k}_mean']:+.3f} {ci(s[f'gradcam_minus_{m}_{fill}_{k}_ci95'])} "
                                     f"(p={e(s[f'wilcoxon_p_vs_{m}_{fill}_{k}'])}, win {s[f'gradcam_better_than_{m}_{fill}_{k}_frac']:.2f})")
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
    # localisation
    p = f"{L}/localisation_{a.tag}.json"
    if os.path.exists(p):
        Lz = json.load(open(p))
        S = Lz["summary"]
        out += ["### IDRiD localisation of the DR Grad-CAM (81 images, FOV only)", "",
                f"Model on IDRiD: mean p(DR) {Lz['model_on_idrid']['p_D_mean']:.3f}, median {Lz['model_on_idrid']['p_D_median']:.3f}, "
                f"{100 * Lz['model_on_idrid']['frac_p_D_ge_0.5']:.1f}% with p >= 0.5. Grad-CAM energy on the optic disc "
                f"{Lz['gradcam_D_energy_on_OD_mean']:.3f} (centre map {Lz['centre_energy_on_OD_mean']:.3f}, OD area {Lz['OD_area_frac_mean']:.3f}). "
                f"Spearman(p(DR), EPG any-lesion) = {Lz['spearman_pD_vs_epg_any']['rho']:.3f} (p={e(Lz['spearman_pD_vs_epg_any']['p'])}).", "",
                "| lesions | n img | area frac (chance EPG) | area frac +15px (≈chance PG-tol) | method | PG strict | PG ±15 px | EPG | pixel AUROC | AP |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        lab = {"any": "any (MA∪HE∪EX∪SE)", "MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates",
               "SE": "soft exudates"}
        for t in ["any", "MA", "HE", "EX", "SE"]:
            s = S[t]
            for i, m in enumerate(["gradcam_D", "random", "centre", "gradcam_G"]):
                first = [lab[t], str(s["n_images"]), f(s["area_frac_mean"], 4), f(s["area_frac_tol_mean"], 3)] if i == 0 else ["", "", "", ""]
                out.append("| " + " | ".join(first + [m.replace("gradcam_D", "Grad-CAM DR").replace("gradcam_G", "Grad-CAM glaucoma (wrong class)"),
                                                      f(s[f"{m}_pg_mean"], 2), f(s[f"{m}_pg_tol_mean"], 2),
                                                      f"{s[f'{m}_epg_mean']:.4f}", f(s[f"{m}_auroc_mean"]),
                                                      f"{s[f'{m}_ap_mean']:.4f}"]) + " |")
        out += ["", "Paired differences Grad-CAM DR minus baseline (mean [95% bootstrap CI], Wilcoxon p):", ""]
        for t in ["any", "MA", "HE", "EX", "SE"]:
            s = S[t]
            parts = []
            for m in ["random", "centre", "gradcam_G"]:
                for k in ["epg", "auroc"]:
                    parts.append(f"{k.upper()} vs {m} {s[f'gradcam_D_minus_{m}_{k}_mean']:+.3f} {ci(s[f'gradcam_D_minus_{m}_{k}_ci95'])} p={e(s[f'wilcoxon_p_gradcam_D_vs_{m}_{k}'])}")
            out.append(f"- {lab[t]}: " + "; ".join(parts))
        out.append("")
    # devices
    for dev, sub, note in [("n100", "", "OPENBLAS_NUM_THREADS=1 (primary)"),
                           ("n100", "openblas_default", "OpenBLAS default threads (first pass)"),
                           ("pi", "", "OPENBLAS_NUM_THREADS=1")]:
        fs = sorted(glob.glob(f"results/{dev}_run010/{sub + '/' if sub else ''}results_*.json"))
        if not fs:
            continue
        out += [f"### On-device cost: {dev}, {note}", "",
                "| runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |",
                "|---|---|---|---|---|---|---|---|"]
        R = {(json.load(open(x))["runtime"], json.load(open(x))["mode"]): json.load(open(x)) for x in fs}
        for (rt, mode), r in sorted(R.items(), key=lambda kv: (kv[0][0], ["plain", "head", "gradcam1", "gradcam5"].index(kv[0][1]))):
            base = R.get((rt, "plain"))
            lat = r["latency_ms"]
            extra = lat["mean"] - base["latency_ms"]["mean"] if base else None
            out.append(f"| {rt} ({r['runtime_version'].split('-')[0] if r['runtime_version'].startswith('openvino') else r['runtime_version']}) | {mode} | {lat['mean']:.1f} / {lat['p50']:.1f} / {lat['p95']:.1f} | "
                       f"{f(r['numpy_part_ms']['mean'], 1) if r['numpy_part_ms'] else '–'} | {f(extra, 1) if mode != 'plain' else '–'} | "
                       f"{r['peak_rss_mb']:.1f} | {e(r['parity_vs_mac_torch']['prob_max_abs_diff'])} | "
                       f"{e(r['parity_vs_mac_torch']['map_max_abs_diff']) if r['parity_vs_mac_torch']['map_max_abs_diff'] is not None else '–'} |")
        out.append("")
    open(f"{L}/tables_{a.tag}.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
