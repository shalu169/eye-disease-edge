## Run 007 — DR ceiling: input resolution (224/384/512) × DR label definition (patient-level vs per-eye)

- **Date**: 2026-10-02
- **Purpose**: test the two hypotheses left open by run 003 for DR's ~0.69-0.72
  val AUC plateau: (b) 224 px is too coarse for microaneurysms/haemorrhages;
  (c) ODIR's D flag is patient-level and copied to both eyes, so some
  "DR-positive" eyes are healthy. Sub-runs: 007b-eval (existing run002/run003
  checkpoints re-scored against per-eye labels, no training), 007a (joint
  5-label at 384/512 px, patient-level D as in run002), 007b (joint 5-label
  trained with per-eye D at 224/384/512), 007c (seed-1 repeats to measure
  run-to-run noise).
- **Hardware / env**: Mac, Apple M5, MPS. Same env as runs 001-003 (Python
  3.14, torch 2.14, timm 1.0.30); `results/env_mac_freeze.txt` unchanged.
- **Native image resolution**: all `data/preprocessed_images/*.jpg` are
  **512×512** (checked every 20th of the 6392 files: 320/320 are 512×512). So
  512 px is the maximum input without going back to the raw ODIR-5K images
  (raw images are in `data/ODIR-5K/ODIR-5K/`; not used here).
- **Per-eye DR label**: derived from that eye's own
  `Left-/Right-Diagnostic Keywords`. Full mapping, vocabulary and counts:
  `results/run007_dr_per_eye_label_mapping.md`. Headline facts:
  - The 9 DR keywords (mild/moderate/severe NPDR, PDR, severe PDR, "diabetic
    retinopathy", 3 "suspected/suspicious" variants) reproduce ODIR's
    patient-level D exactly (1105/1105 D-positive patients, 0 false hits
    among 2253 D-negative ones). `hypertensive retinopathy`,
    `myopia retinopathy` etc. are not DR.
  - **401 of 2123 D=1 images (18.9%) are eyes with no DR keyword**; 250 of
    them are literally `normal fundus`. In val: 85 of 408 (20.8%). Per-eye
    D=1: 1722 images overall, 1399 train, 323 val.
  - Val per-eye grades: none 947, mild 98, moderate 180, severe NPDR 30,
    PDR 5, ungraded 10.
- **Config**: identical to run002 unless stated — MobileNetV3-small (timm,
  ImageNet-pretrained), same patient `GroupShuffleSplit(test_size=0.2,
  random_state=42)` → train 5122 / val 1270, AdamW lr 3e-4 wd 1e-4, 2-epoch
  warmup + cosine, 25 epochs (no early stop), batch 32, run002 augmentation
  with `RandomResizedCrop(S)` / `Resize((S,S))` at S ∈ {224, 384, 512},
  pos_weight inverse prevalence capped [1,20] (per-eye D: 2.661 vs patient D:
  1.987). Best checkpoint = max mean val AUC over the 5 labels, D scored under
  the label definition it was trained on. num_workers 8 (6 for the 224 eye-D
  run). Seeds: original runs unseeded (like run002), 007c repeats `--seed 1`.
- **Metrics** (all on the same 1270 val images): DR AUC under
  (1) **patient-D** (runs 001-003 definition), (2) **eye-D** (per-eye
  keywords), (2b) eye-D strict (drops 2 suspected-DR positives and 10
  DR-adjacent negatives such as laser spots, n=1258), (3) patient-level
  (max prob over the patient's eyes vs D, 672 patients); severity strata vs
  per-eye no-DR eyes; "referable-ish" = moderate NPDR+ vs (none + mild), 10
  ungraded eyes excluded. 95% CIs: patient-clustered paired bootstrap, 2000
  resamples (`results/run007_bootstrap.json`).

- **Result 007b-eval — existing checkpoints re-scored (no retraining)**:

  | ckpt | D AUC patient-D | D AUC eye-D | eye-D strict | patient max-eye | mild | moderate | severe NPDR | PDR (n=5) | referable |
  |---|---|---|---|---|---|---|---|---|---|
  | run002 joint 224 | 0.6947 | 0.7139 | 0.7167 | 0.7193 | 0.6775 | 0.7065 | 0.8647 | 0.9698 | 0.7210 |
  | run003 DR-only 224 | 0.7195 | 0.7339 | 0.7376 | 0.7285 | 0.6970 | 0.7195 | 0.9106 | 0.9945 | 0.7381 |

  run002 reproduces exactly (mean AUC 0.8215, all 5 labels identical to
  run002's log). **Fixing the evaluation labels alone adds only ~+0.02 DR
  AUC** (0.695 → 0.714). The model does score the 85 noisy positives lower
  than true DR eyes (mean p 0.449 vs 0.561; patient-D-negatives 0.372), but
  label noise is not what caps DR at 224. Severe NPDR/PDR are already easy
  (0.86-0.97); mild and moderate NPDR (278 of 323 positive val eyes) are
  where the model fails.

- **Result 007a/007b/007c — trained models (best-by-mean-AUC checkpoint)**:

  | run | res | D train label | seed | best ep | D AUC patient-D | D AUC eye-D | patient max-eye | mild | moderate | severe NPDR | referable | G | C | A | H | mean5 (patient-D) |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | run002 (ref) | 224 | patient | – | 15 | 0.6947 | 0.7139 | 0.7193 | 0.6775 | 0.7065 | 0.8647 | 0.7210 | 0.8722 | 0.9376 | 0.8575 | 0.7452 | 0.8215 |
  | 007c_patientD_224_s1 | 224 | patient | 1 | 10 | 0.6865 | 0.7063 | 0.7030 | 0.6997 | 0.6917 | 0.8710 | 0.7055 | 0.8798 | 0.9369 | 0.8563 | 0.7524 | 0.8224 |
  | 007b_eyeD_224 | 224 | eye | – | 19 | 0.7003 | 0.7169 | 0.7095 | 0.6895 | 0.7015 | 0.8966 | 0.7207 | 0.8817 | 0.9313 | 0.8713 | 0.7492 | 0.8267 |
  | 007a_patientD_384 | 384 | patient | – | 25 | 0.7518 | 0.7642 | 0.7595 | 0.6865 | 0.7720 | 0.9571 | 0.7919 | 0.8944 | 0.9421 | 0.8602 | 0.7912 | 0.8479 |
  | 007c_patientD_384_s1 | 384 | patient | 1 | 13 | 0.7140 | 0.7335 | 0.7239 | 0.6801 | 0.7294 | 0.9335 | 0.7529 | 0.8769 | 0.9558 | 0.8553 | 0.8111 | 0.8426 |
  | 007b_eyeD_384 | 384 | eye | – | 17 | 0.7339 | 0.7626 | 0.7512 | 0.6764 | 0.7723 | 0.9539 | 0.7925 | 0.8964 | 0.9389 | 0.8491 | 0.7973 | 0.8431 |
  | 007a_patientD_512 | 512 | patient | – | 14 | 0.7529 | 0.7720 | 0.7832 | 0.7268 | 0.7699 | 0.9555 | 0.7874 | 0.8923 | 0.9309 | 0.8778 | 0.8430 | 0.8594 |
  | 007c_patientD_512_s1 | 512 | patient | 1 | 19 | 0.7545 | 0.7770 | 0.7751 | 0.6993 | 0.7947 | 0.9524 | 0.8106 | 0.8827 | 0.9351 | 0.8551 | 0.8096 | 0.8474 |
  | **007b_eyeD_512** | 512 | eye | – | 23 | 0.7694 | **0.8029** | 0.7921 | 0.7021 | 0.8312 | 0.9484 | 0.8406 | 0.8931 | 0.9505 | 0.8734 | 0.7931 | 0.8559 |
  | **007c_eyeD_512_s1** | 512 | eye | 1 | 22 | 0.7832 | **0.8131** | 0.8101 | 0.7285 | 0.8366 | 0.9721 | 0.8486 | 0.9090 | 0.9364 | 0.8493 | 0.8245 | 0.8605 |

  G/C/A/H are scored on ODIR patient-level labels throughout (unchanged).
  "mean5 (patient-D)" averages all 5 labels with D scored on patient-D, i.e.
  comparable to run002's 0.8215. Final-epoch (epoch 25, no checkpoint
  selection) values and all other columns: `results/run007_table.md`
  (generated from the JSONs by `results/run007_aggregate.py`). Final-epoch
  D AUC eye-D: 224 0.713/0.713, 384 0.764/0.753 (patient-D trained) and 0.762
  (eye-D), 512 0.794/0.784 (patient-D trained), 0.805/0.814 (eye-D trained) —
  same ordering as the selected checkpoints.

- **Significance (paired, patient-clustered bootstrap, D AUC diff vs run002)**:
  - eye-D labels: 224 eye-D-trained +0.003 [−0.027, +0.031] (n.s.);
    384 runs +0.050 [+0.020, +0.081] / +0.020 [−0.009, +0.049] / +0.049
    [+0.018, +0.079]; 512 patient-D-trained +0.058 [+0.027, +0.091] and
    +0.063 [+0.031, +0.097]; **512 eye-D-trained +0.089 [+0.057, +0.120] and
    +0.099 [+0.068, +0.129]**.
  - 512 eye-D-trained vs 512 patient-D-trained, seed-matched, scored on
    eye-D: +0.031 [+0.007, +0.055] and +0.036 [+0.015, +0.057]; scored on
    patient-D: +0.017 [−0.006, +0.040] and +0.029 [+0.007, +0.050]
    (`results/run007_bootstrap_512_labels.json`).
  - Val-set 95% CI width for a single DR AUC is ≈ ±0.035.

- **Wall-clock (MPS, 25 epochs)**: 224 ≈ 6.5 min (15-16 s/epoch), 384 ≈
  16-18 min (37-42 s/epoch), 512 ≈ 25-29 min (59-70 s/epoch). ~2.8 h total
  GPU time across 9 training runs.

- **Artifacts**:
  - Scripts: `results/run007_common.py` (labels/split/metrics),
    `results/run007_train.py`, `results/run007b_eval_existing.py`,
    `results/run007_aggregate.py`, `results/run007_bootstrap.py`,
    `results/run007_bootstrap_512_labels.py`.
  - Label mapping: `results/run007_dr_per_eye_label_mapping.md`.
  - Logs: `results/logs/run007b_eval_existing.log`,
    `results/logs/run007{a,b,c}_joint5_*.log`,
    `results/logs/run007_bootstrap*.log`.
  - Per-run JSON (args, per-epoch history under both D definitions, best-epoch
    metrics): `results/run007*_summary.json`; best-epoch val probabilities
    `results/run007*_val_probs.npy`; existing-ckpt eval
    `results/run007b_eval_existing.json` + `results/run007b_run00{2,3}_val_probs.npy`.
  - Comparison table: `results/run007_table.md`.
  - Checkpoints: `results/checkpoints/run007a_joint5_patientD_384_best_epoch25.pt`,
    `run007a_joint5_patientD_512_best_epoch14.pt`,
    `run007b_joint5_eyeD_224_best_epoch19.pt`,
    `run007b_joint5_eyeD_384_best_epoch17.pt`,
    `run007b_joint5_eyeD_512_best_epoch23.pt`,
    `run007c_joint5_patientD_224_s1_best_epoch10.pt`,
    `run007c_joint5_patientD_384_s1_best_epoch13.pt`,
    `run007c_joint5_patientD_512_s1_best_epoch19.pt`,
    `run007c_joint5_eyeD_512_s1_best_epoch22.pt`.

- **Verdict**:
  - **(c) label noise is real but small on its own.** ~19-21% of
    patient-level DR positives are eyes with no DR keyword (most of them
    "normal fundus"). Re-scoring run002 against per-eye labels adds only +0.02,
    and training on per-eye labels at 224 adds nothing (0.717 vs 0.714 eye-D,
    n.s.).
  - **(b) resolution is the main lever.** 512 px lifts DR AUC by ~+0.06
    (patient-D 0.753/0.755 vs 0.695; eye-D 0.772/0.777 vs 0.714), consistent
    across 2 seeds and significant. 384 px lands in between but is noisy
    across seeds (patient-D 0.752 vs 0.714). Gains come from moderate NPDR
    (0.71 → 0.77-0.79) and severe NPDR (0.86 → 0.95); mild NPDR stays
    0.68-0.73 at every resolution.
  - **Per-eye labels help once the resolution is high enough to use them.**
    At 512 px, per-eye D training adds another +0.03 eye-D AUC over
    patient-D training in both seed pairs (CIs exclude 0): **DR AUC 0.803 /
    0.813 (eye-D), 0.769 / 0.783 (patient-D), referable-ish (moderate+)
    0.841 / 0.849**. The best configuration is +0.09-0.10 over run002.
  - **Still not 0.9+.** Even the best model is ~0.81 per-eye. The remaining
    gap to the literature is plausibly the label protocol (no referable/DME
    grading, and mild NPDR is 30% of positives and barely separable here),
    ODIR image heterogeneity, and a 2.5M-param backbone at ≤512 px. Raw
    ODIR images at >512 px were not tried.
  - **Other labels**: higher resolution does not hurt any disease. H
    (hypertensive retinopathy) gains most (0.745 → 0.79-0.84 at 384/512),
    G is up slightly (0.872 → 0.88-0.91), and C (0.93-0.96) and A
    (0.85-0.88) are flat within seed noise. Mean 5-label AUC goes 0.8215 →
    0.847-0.861 at 512. Per-eye D training does not change G/C/A/H beyond
    noise.
  - **Caveats**: the checkpoint is selected on the same val set it is
    reported on (as in runs 001-003), so best-epoch numbers are slightly
    optimistic; final-epoch numbers give the same ordering. Seed-to-seed
    spread for DR is ~0.01-0.04, so single-run differences under ~0.03 should
    not be claimed. There is no held-out test split yet.
  - **Recommendation for the edge benchmarks**: carry the **512 px joint
    5-label model trained with per-eye D** (`run007c_joint5_eyeD_512_s1_best_epoch22.pt`,
    or `run007b_joint5_eyeD_512_best_epoch23.pt`; they are equivalent within
    noise) as the paper's main model. Report DR under both label definitions,
    with per-eye as primary and the label derivation documented. Keep the
    224 px run002 as the low-cost reference point. 512 px costs ~5.2× the
    FLOPs of 224 px. The N100 latency (2 ms at 224) has headroom for that,
    but **512 px latency and RAM on the N100 and especially the Pi 3B
    (870 MB ceiling) are not measured**. They need a run004-style benchmark
    before the paper claims 512 px is deployable on the Pi; 384 px is the
    fallback if the Pi can't handle 512.
