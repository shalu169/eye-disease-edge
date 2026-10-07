"""Write results/run008_FINAL_CHECKPOINT.json for the edge-benchmarking agent.
  --mode first : point at model A seed 0 as soon as it has finished.
  --mode all   : after all 3 A seeds, point at the seed with the MEDIAN validation
                 mean AUC (selection uses validation only, never test).
Usage (repo root): .venv/bin/python results/run008_final_ckpt.py --mode first|all
"""
import argparse
import json

SEEDS = [0, 1, 2]
TAG = "run008A_joint5_eyeD_512_s{}"


def entry(seed, s):
    return {"checkpoint": s["checkpoint"], "seed": seed, "best_epoch": s["best_epoch"],
            "val_sel_mean_auc": s["best_sel_mean_auc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["first", "all"], required=True)
    a = ap.parse_args()
    if a.mode == "first":
        seeds = [0]
    else:
        seeds = SEEDS
    summ = {sd: json.load(open(f"results/{TAG.format(sd)}_summary.json")) for sd in seeds}
    if a.mode == "first":
        chosen = 0
        why = "first finished seed of model A (provisional; will be replaced by the median-val seed)"
    else:
        order = sorted(seeds, key=lambda sd: summ[sd]["best_sel_mean_auc"])
        chosen = order[1]
        why = ("seed with the median VALIDATION selection mean AUC (5 labels, D scored on per-eye labels) "
               "among seeds " + str(SEEDS) + "; test set not used for this choice")
    s = summ[chosen]
    out = {
        "model": "run008 A (final): MobileNetV3-small (timm mobilenetv3_small_100), joint 5-label",
        "checkpoint": s["checkpoint"],
        "seed": chosen,
        "best_epoch": s["best_epoch"],
        "input_size": 512,
        "preprocessing": "PIL RGB -> Resize((512,512)) bilinear -> ToTensor -> Normalize(ImageNet mean/std)",
        "output": "5 logits, sigmoid per label, order [D, G, C, A, H]",
        "label_config": {"labels": ["D", "G", "C", "A", "H"],
                         "D": "per-eye DR label from that eye's diagnostic keywords (results/run007_dr_per_eye_label_mapping.md)",
                         "G_C_A_H": "ODIR patient-level labels"},
        "test_manifest": "results/run008_splits/test.csv",
        "val_manifest": "results/run008_splits/val.csv",
        "train_manifest": "results/run008_splits/train.csv",
        "val_sel_mean_auc": s["best_sel_mean_auc"],
        "selection_rule": why,
        "all_seeds": [entry(sd, summ[sd]) for sd in seeds],
        "note": ("Edge INT8 calibration must use images from train.csv only; the old "
                 "edge/artifacts/calib_manifest.csv has 55 images that are now in test.csv."),
        "updated_after_all_seeds": a.mode == "all",
    }
    json.dump(out, open("results/run008_FINAL_CHECKPOINT.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
