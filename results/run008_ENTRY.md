## Run 008 — Final protocol: held-out test set, 3 seeds, A (512 px per-eye DR) vs B (224 px patient DR) vs C (512 px patient DR)

- **Date**: 2026-10-03
- **Purpose**: replace the development numbers of runs 002-007, which used
  the same 1270-image validation set both to choose checkpoints and to
  report them, with an untouched held-out test set and seed variation.
- **Protocol**:
  - Original patient-level split (GroupShuffleSplit test_size=0.2,
    random_state=42) kept. The original 1270-image validation set is used
    ONLY for checkpoint selection and for choosing operating thresholds.
  - New TEST set: GroupShuffleSplit(test_size=0.2, random_state=2026) by
    patient applied to the original training split. These images were
    only ever trained on in runs 001-007, never evaluated, so no earlier
    design decision used them. The rest is the new training set.
  - Sizes: train 4097 images / 2148 patients, val 1270 / 672, test 1025 /
    538. Zero patient and file overlap asserted
    (`results/logs/run008_make_splits.log`). Test: patient-level D=346,
    per-eye D=285, G=62, C=62, A=42, H=34.
  - Recipe unchanged from run 007 (MobileNetV3-small, AdamW 3e-4, wd 1e-4,
    2-epoch warm-up + cosine, 25 epochs, pos_weight capped [1,20], run002
    augmentation). Best epoch by mean validation AUC (D scored on the
    training label definition).
  - Models, 3 seeds each (0, 1, 2): **A** 512 px, per-eye DR labels;
    **B** 224 px, patient-level DR labels (= run002 recipe); **C** 512 px,
    patient-level DR labels.
  - Final deployment checkpoint: the A seed with the MEDIAN validation
    mean AUC (s0 0.8597, s1 0.8517, s2 0.8574 -> **s2**,
    `results/checkpoints/run008A_joint5_eyeD_512_s2_best_epoch21.pt`).
    Chosen before any test evaluation (`results/run008_FINAL_CHECKPOINT.json`).
  - Test evaluated once by `results/run008_eval_test.py` (per-label AUC,
    DR under both label definitions, severity strata, referable DR,
    validation-chosen 90 %-sensitivity thresholds applied to test,
    patient-clustered bootstrap 2000 of the seed-mean AUC, paired A-B and
    A-C differences). Test set never touched during training or model
    selection.
- **Execution note**: the first launch (01:40) died before epoch 1 when its
  agent was stopped (`results/logs/run008_driver_attempt1_killed.log`);
  relaunched unchanged at 11:51 with nohup + caffeinate, all 9 runs rc=0,
  ALL DONE 14:37. Test inference ran on MPS after training released the GPU
  (dry-run on CPU matched run 007's stored validation probabilities to
  8e-6).
- **Results — test set, seed mean ± s.d. (3 seeds), 95 % CI of the seed-mean AUC**:

  | label | A: 512 per-eye | B: 224 patient | C: 512 patient |
  |---|---|---|---|
  | DR, per-eye labels | 0.780 ± 0.032 [0.747, 0.812] | 0.708 ± 0.028 [0.670, 0.742] | 0.775 ± 0.031 [0.741, 0.806] |
  | DR, patient labels | 0.752 ± 0.023 [0.716, 0.786] | 0.698 ± 0.024 [0.661, 0.735] | 0.755 ± 0.019 [0.720, 0.790] |
  | Glaucoma | 0.889 ± 0.012 [0.822, 0.939] | 0.860 ± 0.004 [0.798, 0.908] | 0.885 ± 0.008 [0.815, 0.937] |
  | Cataract | 0.918 ± 0.023 [0.873, 0.955] | 0.905 ± 0.015 [0.850, 0.948] | 0.920 ± 0.009 [0.872, 0.957] |
  | AMD | 0.852 ± 0.024 [0.765, 0.924] | 0.841 ± 0.022 [0.735, 0.928] | 0.878 ± 0.011 [0.798, 0.946] |
  | Hypertensive ret. | 0.791 ± 0.021 [0.703, 0.871] | 0.694 ± 0.038 [0.609, 0.776] | 0.798 ± 0.009 [0.705, 0.880] |
  | Mean of 5 (D on training label) | 0.846 ± 0.017 | 0.800 ± 0.018 | 0.847 ± 0.004 |

  Paired bootstrap differences of the seed-mean AUC (95 % CI):

  | label | A − B (resolution + labels) | A − C (labels only, both 512) |
  |---|---|---|
  | DR, per-eye | +0.073 [0.047, 0.099] | +0.005 [−0.014, 0.024] |
  | DR, patient | +0.053 [0.029, 0.080] | −0.004 [−0.023, 0.015] |
  | Glaucoma | +0.029 [0.004, 0.053] | +0.003 [−0.009, 0.019] |
  | Cataract | +0.012 [−0.011, 0.037] | −0.002 [−0.014, 0.011] |
  | AMD | +0.011 [−0.032, 0.061] | −0.027 [−0.048, −0.007] |
  | Hypertensive ret. | +0.097 [0.028, 0.164] | −0.007 [−0.050, 0.032] |

  Per seed (val selection mean AUC -> test mean AUC; test referable DR /
  moderate vs no-DR / mild vs no-DR AUC):
  A s0 0.8597 -> 0.8569 (0.826 / 0.817 / 0.695); A s1 0.8517 -> 0.8260
  (0.789 / 0.773 / 0.644); A s2 0.8574 -> 0.8545 (0.869 / 0.865 / 0.667);
  B s0 0.8219 -> 0.8098; B s1 0.8099 -> 0.7792; B s2 0.8157 -> 0.8102;
  C s0 0.8445 -> 0.8506; C s1 0.8438 -> 0.8428; C s2 0.8423 -> 0.8484.

  Final model (A s2) on test: DR per-eye 0.809, patient 0.771, patient-level
  (max over eyes) 0.799, referable (moderate+ vs rest) 0.869; DR severity vs
  no-DR eyes: mild 0.667 (n=86), moderate 0.865 (156), severe NPDR 0.940
  (25), PDR 0.966 (6). G 0.892, C 0.897, A 0.862, H 0.813; mean 0.855.
  At thresholds giving 90 % sensitivity on validation: DR sens 0.888 / spec
  0.469; G 0.887 / 0.667; C 0.806 / 0.821; A 0.810 / 0.746; H 1.000 / 0.286.
- **Artifacts**: splits `results/run008_splits/{train,val,test}.csv`;
  checkpoints `results/checkpoints/run008{A,B,C}_*_s{0,1,2}_best_epochN.pt`;
  logs `results/logs/run008*.log`; scripts `results/run008_{common,make_splits,train,final_ckpt,eval_test}.py`,
  `results/run008_run_all.sh`; per-model summaries `results/run008*_summary.json`;
  test metrics `results/run008_test_metrics.json`; per-image probabilities
  `results/run008_probs/<tag>_{val,test}.npy` with row order
  `results/run008_probs/{val,test}_order.csv`.
- **Verdict**:
  - **Resolution is the effect that replicates.** On untouched data, going
    from 224 to 512 px lifts DR by +0.05 to +0.07 AUC, hypertensive
    retinopathy by +0.10 and glaucoma by +0.03, with CIs excluding zero.
    Mean AUC over five labels 0.800 -> 0.846.
  - **The per-eye-label gain seen in run 007 does NOT replicate.** At 512 px,
    A and C are indistinguishable for DR on test (+0.005, CI −0.014 to
    0.024), and A is slightly worse for AMD (−0.027). The run-007 gain of
    about +0.03 was within what checkpoint selection on the same validation
    set can produce. Per-eye labels remain the more appropriate target for a
    per-image screen and give the higher per-eye DR score for the deployed
    model, but the paper must not claim they improve accuracy.
  - Seed variation is large (DR s.d. 0.02-0.03; A s1 is clearly weaker).
    The deployed seed (A s2, chosen on validation) happens to be the
    strongest on test for DR (0.809 vs seed mean 0.780); the paper must
    report the seed mean as the headline and the deployed seed separately.
  - Validation and test numbers agree within about 0.01-0.03 mean AUC, so
    the earlier development numbers were only mildly optimistic.
  - Mild NPDR remains the failure mode (AUC 0.64-0.70 across A seeds).
  - Operating points at 90 % sensitivity carry low specificity for DR (0.47)
    and hypertensive retinopathy (0.29): as a stand-alone screen this model
    would refer many negatives.
- **Limitations**: resolution and label definition were chosen on the
  validation set in run 007, which also selects checkpoints here; the test
  set guards the reported numbers, not those design choices. Single
  dataset; no external validation. 55 images of the run 004/006 INT8
  calibration manifest fall in this test set (INT8 is not evaluated on
  test).
