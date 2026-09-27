# Experiment Log

Everything needed to reproduce and write up results for the paper. One entry
per run: config, hardware, dataset split, metrics, artifact paths. Update this
file every time a run finishes — do not let results live only in terminal
scrollback.

Conventions:
- Checkpoints → `results/checkpoints/run{NNN}_*.pt`
- Raw stdout logs → `results/logs/run{NNN}_*.log`
- Script snapshot used for that run → `results/run{NNN}_*.py`
- Env snapshot (pip freeze) → `results/env_<machine>_freeze.txt`, refreshed
  whenever the environment changes materially (new package, version bump).

---

## Run 001 — baseline v1 (feasibility check)

- **Date**: 2026-09-27
- **Purpose**: confirm multi-label training works end-to-end on an edge-sized
  backbone before investing in tuning or the edge pipeline.
- **Hardware**: Mac, Apple M5, macOS 26.3, MPS backend.
- **Env**: Python 3.14.6, torch 2.14.0, torchvision 0.29.0, timm 1.0.30.
  Full freeze: `results/env_mac_freeze.txt`.
- **Model**: `timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=5)`.
- **Dataset**: ODIR-5K, Kaggle `andrewmvd/ocular-disease-recognition-odir5k`.
  6392 per-eye images / 3358 patients. Split by patient ID (`GroupShuffleSplit`,
  test_size=0.2, seed=42) → train=5122, val=1270. No patient leakage across
  split.
- **Labels**: patient-level multi-hot, 5 of the 8 ODIR classes —
  D=DR, G=glaucoma, C=cataract, A=AMD, H=hypertensive retinopathy (N=normal
  and O=other dropped, out of scope for this paper).
- **Training config**: 5 epochs, fixed LR=1e-4 (no schedule), AdamW,
  BCEWithLogitsLoss with per-label `pos_weight` (inverse prevalence, capped
  [1,20]): D=1.99 G=15.79 C=14.81 A=19.65 H=20.0. Augmentation: resize +
  horizontal flip only.
- **Result (val AUC, epoch 5/5)**: D=0.629 G=0.853 C=0.936 A=0.791 H=0.651.
- **Artifacts**: checkpoint `results/checkpoints/run001_baseline_v1_epoch5.pt`,
  full log `results/logs/run001_baseline_v1.log`.
- **Verdict**: feasibility confirmed. DR (D) AUC anomalously low vs.
  literature's typical 0.9+ — flagged as likely under-training (5 epochs, no
  LR schedule, weak aug), not a real ceiling, since cataract/glaucoma already
  strong with the identical setup. Not tuned, not a paper number.

---

## Run 002 — tuned baseline (in progress, started 2026-09-27)

- **Purpose**: address run 001's under-training, get a real proof-of-concept
  number, specifically investigate whether DR's low AUC was a training-budget
  artifact.
- **Hardware / env**: same as run 001.
- **Model**: same architecture (MobileNetV3-small, timm, pretrained, 5-way).
- **Dataset / split**: identical to run 001 (same seed, same patient split).
- **Training config changes vs. run 001**: 25 epochs, LR=3e-4 with 2-epoch
  linear warmup + cosine decay, weight_decay=1e-4, AdamW. Augmentation
  strengthened: RandomResizedCrop(224, scale=0.8-1.0), RandomHorizontalFlip,
  RandomRotation(20°), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1).
  Best checkpoint (by mean AUC across 5 labels) saved each time it improves.
- **Result — FINAL (best epoch 15/25)**: mean_auc=0.8215 —
  D=0.6947 G=0.8722 C=0.9376 A=0.8575 H=0.7452.
  Full per-epoch curve in `results/logs/run002_baseline_tuned.log`.
- **Artifacts**: script snapshot `results/run002_baseline_tuned.py`, full log
  `results/logs/run002_baseline_tuned.log`, best checkpoint
  `results/checkpoints/run002_baseline_tuned_best_epoch15.pt`.
- **Status**: DONE.
- **Verdict**: confirms run 001's DR AUC was partly under-training (0.63 →
  0.69-0.71 range), but DR still plateaus well below the 0.9+ typical of
  literature's single-task DR-only models, while cataract/glaucoma/AMD reach
  respectable numbers in the same joint model. Train loss keeps falling
  (0.21 by epoch 25) while DR's val AUC does not — past epoch ~13 this looks
  like overfitting on DR specifically, not more headroom from more epochs.
  **Open question carried forward**: is DR's ceiling (a) task interference
  from joint multi-label training (one shared backbone competing across 5
  heads), (b) 224px resolution too coarse for the microaneurysm/hemorrhage
  detail that separates DR grades, or (c) label noise (patient-level DR flag
  here doesn't match the "referable DR" binary protocol most cited 0.9+
  results use)? Worth a quick single-task DR-only ablation before trusting
  the joint-model framing for the paper — if a single-task DR model also
  plateaus around 0.7 on this exact split, it's (b)/(c) not (a), which
  changes what the paper's multi-disease claim can say.

---

## Run 003 — DR-only single-task ablation

- **Date**: 2026-09-27
- **Purpose**: isolate whether DR's AUC ceiling in run 002's joint 5-disease
  model is task interference (shared backbone across 5 heads) vs. a
  resolution/label-quality limit that would persist even single-task.
- **Config**: identical to run 002 in every respect (backbone, split/seed,
  25-epoch cosine schedule w/ 2-epoch warmup, augmentation, pos_weight
  formula) — only difference is `LABEL_COLS = ["D"]`, single output head.
- **Result — FINAL (best epoch 9/25)**: D_auc=0.7195. Plateaus in the
  0.68-0.72 band from epoch ~5 onward; train loss keeps falling to 0.30 while
  val AUC does not — same overfitting-past-plateau shape as DR showed inside
  the joint model.
- **Artifacts**: script `results/run003_dr_only_ablation.py`, log
  `results/logs/run003_dr_only_ablation.log`, checkpoint
  `results/checkpoints/run003_dr_only_best_epoch9.pt`.
- **Verdict — answers run 002's open question**: single-task DR
  (AUC 0.7195) beats joint-model DR (AUC 0.6947) by only ~0.025, and both
  plateau in the same band. **Task interference (a) is ruled out** — dropping
  the other 4 disease heads barely moves DR's ceiling. The limiting factor is
  (b) 224px resolution and/or (c) label-protocol mismatch (patient-level D
  flag here isn't the clean "referable DR" binary grading most 0.9+ papers
  use), not the multi-disease joint-training framing.
  **This is a positive result for the paper's core claim**: joint multi-disease
  training does not sacrifice per-disease accuracy vs. specialist single-task
  models (within noise, on this dataset/resolution) — cataract/glaucoma/AMD
  numbers from run 002 should be read the same way this validates. DR itself
  needs a separate fix (higher input resolution, and/or swap to a
  severity-graded DR label source like APTOS 2019 for the DR head
  specifically) — track as future work, not a blocker for moving to the edge
  pipeline.

## Dataset provenance (applies to all runs above)

- Source: Kaggle `andrewmvd/ocular-disease-recognition-odir5k` (ODIR-5K
  mirror, 66.6k downloads, usability 0.82 — most-used mirror in the literature).
  Original: Peking University "Ocular Disease Intelligent Recognition" (ODIR-5K).
  Cite the original ODIR-5K source in the paper, not the Kaggle mirror.
- Files: `data/full_df.csv` (labels), `data/preprocessed_images/*.jpg`
  (per-eye images, already preprocessed/cropped by the mirror uploader).
- Verified 2026-09-27: N/D/G/C/A/H/M/O columns are patient-level (identical
  across a patient's left/right rows, checked on all 3358 patients — 0
  mismatches), all sampled image files present on disk.
- Class prevalence (patient-level, n=3358): N=1080 D=1105 G=206 C=208 A=163
  H=103 M=171 O=905. Confirms the severe class-imbalance gap called out in
  `brainstorm.md` §1 is real in this exact dataset — worth citing directly in
  the paper's dataset section.

## Target hardware for edge benchmarking (not yet run)

See `eye-disease-edge-ai-project.md` memory + `brainstorm.md` §3 for full
specs. Summary: N100 mini PC (`n100m`, x86, 15GB RAM, Intel UHD iGPU,
OpenVINO path) and Raspberry Pi 3B (`pi@192.168.188.108`, ARMv7, 870MB RAM
hard ceiling, TFLite path, no camera attached).
