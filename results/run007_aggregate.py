"""Build run007 comparison table from the JSON outputs (no numbers typed by hand).
Usage (repo root): .venv/bin/python results/run007_aggregate.py > results/run007_table.md
"""
import glob
import json

rows = []
ex = json.load(open("results/run007b_eval_existing.json"))
for k in ["run002", "run003"]:
    r = ex[k]
    rows.append((f"{k} (existing ckpt)", "224", "patient", "joint5" if len(r["label_cols"]) == 5 else "D",
                 "best", r["dr"], r["label_auc_patient_level"], None))
for f in sorted(glob.glob("results/run007*_summary.json")):
    s = json.load(open(f))
    a = s["args"]
    last = s["history"][-1]
    rows.append((s["tag"], str(a["img"]), a["dlabel"], a["cols"], f"best ep{s['best_epoch']}", s["dr"],
                 s["best_label_auc_patient_level"], last))

print("| model | res | D train label | heads | ckpt | D AUC patient-D | D AUC eye-D | eye-D strict | patient max-eye | mild vs noDR | mod vs noDR | severe NPDR vs noDR | PDR vs noDR | referable (mod+ vs rest) | G | C | A | H | mean5 (patient-D) |")
print("|" + "---|" * 19)
f4 = lambda x: f"{x:.4f}"
for name, res, dl, cols, ck, d, la, last in rows:
    other = [f4(la[c]) if c in la else "-" for c in ["G", "C", "A", "H"]]
    mean5 = f4(sum(la.values()) / 5) if len(la) == 5 else "-"
    print(f"| {name} | {res} | {dl} | {cols} | {ck} | {f4(d['auc_patientD'])} | {f4(d['auc_eyeD'])} | "
          f"{f4(d['auc_eyeD_strict'])} | {f4(d['auc_patient_maxeye'])} | {f4(d['auc_mild_vs_noDR'])} | "
          f"{f4(d['auc_moderate_vs_noDR'])} | {f4(d['auc_severe_npdr_vs_noDR'])} | {f4(d['auc_pdr_vs_noDR'])} | "
          f"{f4(d['auc_referable_modplus_vs_rest'])} | " + " | ".join(other) + f" | {mean5} |")

print()
print("Final-epoch (epoch 25, no checkpoint selection) values:")
print()
print("| model | D AUC patient-D | D AUC eye-D | G | C | A | H |")
print("|---|---|---|---|---|---|---|")
for name, res, dl, cols, ck, d, la, last in rows:
    if last is None:
        continue
    print(f"| {name} | {f4(last['D_auc_patientD'])} | {f4(last['D_auc_eyeD'])} | "
          + " | ".join(f4(last.get(f'{c}_auc', float('nan'))) for c in "GCAH") + " |")

print()
print("Mean predicted DR prob: noisy positives (patient D=1, eye no DR) / true eye-DR / patient D=0")
for name, *_, d, la, last in rows:
    print(f"- {name}: {d['mean_p_noisy_pos']:.3f} / {d['mean_p_eyeDR_pos']:.3f} / {d['mean_p_patientD0']:.3f}")
