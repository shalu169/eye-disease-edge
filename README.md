# Lightweight multi-disease retinal screening on low-cost edge hardware

Research code, experiment log, trained weights and benchmarking harness for a
single MobileNetV3-small network that screens **five retinal conditions from one
fundus photograph** — diabetic retinopathy (DR), glaucoma, cataract, age-related
macular degeneration (AMD) and hypertensive retinopathy — and runs **entirely
on-device** on hardware that costs tens to low hundreds of dollars.

The study trains on [ODIR-5K](#data-not-included-in-this-repository), evaluates on
a patient-disjoint held-out test set over three seeds, then deploys the model with
ONNX Runtime, OpenVINO and TensorFlow Lite on two tiers of cheap hardware — an
**Intel N100 mini PC** and a **Raspberry Pi 3B** — measuring export fidelity,
latency, memory, energy, INT8 quantization, probability calibration and
on-device Grad-CAM.

The manuscript is in preparation for *Biomedical Signal Processing and Control*
(Elsevier) and is **not part of this repository** — it will be released with the
camera-ready version. Authors: Shalini Agarwal, Aruna Bhat — Department of
Computer Science & Engineering, Delhi Technological University.

**Every number in the paper traces to a numbered run in
[`results/EXPERIMENTS.md`](results/EXPERIMENTS.md).** That file is the primary
record — read it before trusting anything summarised here.

---

## Headline results

Held-out test set (1025 images / 538 patients, never used for training or model
selection). Seed mean ± s.d. over 3 seeds. Run 008.

| Label | A: 512 px, per-eye DR | B: 224 px, patient DR | C: 512 px, patient DR |
|---|---|---|---|
| DR (per-eye labels) | 0.780 ± 0.032 | 0.708 ± 0.028 | 0.775 ± 0.031 |
| Glaucoma | 0.889 ± 0.012 | 0.860 ± 0.004 | 0.885 ± 0.008 |
| Cataract | 0.918 ± 0.023 | 0.905 ± 0.015 | 0.920 ± 0.009 |
| AMD | 0.852 ± 0.024 | 0.841 ± 0.022 | 0.878 ± 0.011 |
| Hypertensive ret. | 0.791 ± 0.021 | 0.694 ± 0.038 | 0.798 ± 0.009 |
| **Mean of 5** | **0.846 ± 0.017** | **0.800 ± 0.018** | **0.847 ± 0.004** |

Five findings worth knowing before reusing this code:

1. **Resolution is the lever that replicates.** 224 → 512 px lifts mean AUC
   0.800 → 0.846, DR by +0.073 (95% CI 0.047–0.099), hypertensive retinopathy by
   +0.097, glaucoma by +0.029 — CIs excluding zero on untouched data.
2. **Per-eye DR labels did *not* help.** Run 007 saw ~+0.03 on validation; on
   test, A vs C is +0.005 (CI −0.014 to 0.024). The run-007 gain was checkpoint
   selection noise. Per-eye labels remain the more *appropriate* target for a
   per-image screen, but the paper does not claim they improve accuracy.
3. **FP32 export is lossless on both tiers.** CPU runtimes match PyTorch to
   ~1e-5 (N100) and ~8.8e-5 (Pi), AUCs identical to four decimals.
4. **Avoid naive INT8.** Post-training quantization cost up to 0.23 AUC and
   broke calibration for little speed gain. See runs 004, 006, 011, 012.
5. **Class weighting makes outputs over-confident, and temperature scaling
   cannot fix it.** Platt scaling can (DR ECE 0.163 → 0.027) because it has a
   bias term; the `pos_weight` in the loss shifts logits by ≈ log(w).

On-device cost of the 512 px model (run 009):

| Tier | Backend | Model-only | End-to-end | Peak RSS | Energy/inference |
|---|---|---|---|---|---|
| Intel N100 | ORT / OpenVINO CPU | 8.3–8.4 ms | ~16 ms (≈60 img/s) | 222 MB | 120.7 / 127.7 mJ |
| Intel N100 | OpenVINO iGPU | 6.9 ms | — (144.6 inf/s sustained) | 617 MB | **69.6 mJ** (51.7 above idle) |
| Raspberry Pi 3B | TFLite, 4 threads | 0.51 s | ~0.74 s (≈1.35 img/s) | 65 MB | not measured |

Pi latencies were measured under an intermittent 600 MHz under-voltage cap —
roughly **2× pessimistic** versus a properly powered Pi 3B. Accuracy is
clock-independent.

**Grad-CAM** (run 010) runs on-device with no autodiff: the backbone emits the
576×16×16 feature map and NumPy computes the head and the CAM weights
analytically. It matches PyTorch autograd to 2.3e-4 (ONNX) / 7.2e-4 (TFLite),
passes Adebayo's model-randomisation sanity check, and beats a centre-prior
baseline on insertion (0.740 vs 0.593). But DR lesion localisation against
expert IDRiD masks is weak in absolute terms (pixel AUROC 0.657) and it
**never highlights microaneurysms**.

### Known limitations

Single dataset, no external validation. Mild NPDR remains the failure mode
(AUC 0.64–0.70). At 90% validation sensitivity, test specificity is low for DR
(0.47) and hypertensive retinopathy (0.29) — as a stand-alone screen this model
would refer many negatives. Resolution and label definition were chosen on the
validation set that also selects checkpoints; the test set guards the reported
numbers, not those design choices. Seed variation is large (DR s.d. 0.02–0.03).

---

## Repository layout

```
baseline.py                 Run 001/002 — first multi-label baseline (224 px)
dr_only_ablation.py         Run 003 — DR-only single-task ablation
abstract.md, brainstorm.md  Early planning notes (superseded by the manuscript)
results/
  EXPERIMENTS.md            PRIMARY RECORD — one entry per run: config,
                            hardware, split, metrics, artifact paths
  run{007..012}_*.py/.sh    Per-run script snapshots (exactly what was run)
  run0NN_ENTRY.md           Per-run write-ups
  checkpoints/*.pt          Trained weights (18)
  run008_splits/*.csv       train / val / test split definitions
  run008_probs/             Per-image probabilities + row order
  n100_run0NN/, pi_run0NN/  Raw device benchmark JSON + logs
  run010/, run011/          Grad-CAM and calibration outputs
  issues/                   Investigation notes
edge/
  export_onnx.py, export_tflite.py    Model export
  bench_n100.py, bench_pi.py          Device benchmark harnesses
  energy_n100.py                      RAPL energy measurement
  gradcam_numpy.py, gradcam_bench.py  Autodiff-free on-device Grad-CAM
  run_n100.sh, run_pi.sh              Device drivers
  artifacts/                          Exported ONNX / TFLite / fixtures (55)
docs/superpowers/           Mobile-app design spec and plan
```

The `edge/` scripts are the live, generalised versions (model, resolution,
manifest and reference passed as arguments); `results/run0NN_*.py` are frozen
snapshots of what each run actually executed. With no arguments, the `edge/`
scripts reproduce runs 004/005/006.

A separate branch, **`worktree-fundus-app-local-mode`**, holds a Flutter mobile
app (`mobile_app/`) that runs the same model on-device in Dart — including the
Grad-CAM implementation that `edge/gradcam_numpy.py` was ported from.

---

## Data (not included in this repository)

Datasets are deliberately not committed. Download them into `data/`:

**ODIR-5K** — training and evaluation. Kaggle mirror
`andrewmvd/ocular-disease-recognition-odir5k`; cite the original Peking
University *Ocular Disease Intelligent Recognition* release in any publication,
not the mirror.

```
data/full_df.csv                  Labels (N/D/G/C/A/H/M/O, patient-level)
data/preprocessed_images/*.jpg    6392 per-eye images, 512x512, disc-cropped
```

6392 images from 3358 patients. This project uses 5 of the 8 classes —
D, G, C, A, H; N (normal) and O (other) are out of scope. Labels are
patient-level and identical across a patient's two rows (verified on all 3358
patients). Prevalence is severely imbalanced: N=1080 D=1105 G=206 C=208 A=163
H=103 M=171 O=905.

**IDRiD** — expert pixel-level DR lesion masks, used only for Grad-CAM
localisation in run 010. Kaggle mirror
`aaryapatel98/indian-diabetic-retinopathy-image-dataset` (CC BY 4.0;
Porwal et al. 2018, *Data* 3(3):25).

```
data/idrid/raw/      Unpacked archive
data/idrid/prep512/  81 images + masks, preprocessed by results/run010_idrid_prep.py
```

---

## Environments

Three separate environments were used; pinned freezes are in
`results/env_{mac,n100,pi}_freeze.txt`.

**Training / analysis (Mac, Apple M5, MPS)** — `.venv`, Python 3.14:
`torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.30`, `numpy==2.5.3`,
`pandas==3.0.6`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `pillow==12.3.0`,
`grad-cam==1.5.7`, `matplotlib==3.11.2`, `opencv-python-headless==5.0.0.93`.

Run 010 additionally needs `.venv-run010` (Python 3.12, `onnxruntime==1.30.0`
matching the N100, `numpy==1.26.4` matching the Pi) and a TensorFlow environment
for the TFLite export (`onnx2tf==1.26.3`, TF 2.18.0, Python 3.12).

**Intel N100** (Debian 13, kernel 6.12) — Python 3.13: `onnxruntime==1.30.0`,
`openvino==2026.4.1`, `nncf==3.4.0`, `numpy==2.4.6`, `pillow==12.3.0`.

**Raspberry Pi 3B** (32-bit Raspbian 11, armv7l) — Python 3.9:
`tflite-runtime==2.13.0` (newest armv7/cp39 wheel; `ai-edge-litert` has no
armv7 build), `numpy==1.26.4` (pinned <2 — numpy 2.x fails to import without
`libopenblas.so.0`), `pillow==11.2.1`. Install system `libopenblas0`.

MPS, CUDA and CPU give slightly different results; the recorded runs used MPS
for training and **CPU only** for run 010.

---

## Reproducing the experiments

All commands run from the repository root. Long jobs should use
`nohup` + `caffeinate` — run 008's first launch died when its parent shell
was killed.

### Training

```bash
# Runs 001/002 — 224 px baseline, patient-level DR labels
.venv/bin/python baseline.py

# Run 003 — DR-only single-task ablation
.venv/bin/python dr_only_ablation.py

# Run 007 — resolution x DR-label-definition sweep
.venv/bin/python results/run007_train.py \
    --tag run007a_joint5_384 --img 384 --dlabel patient

# Run 008 — FINAL protocol: 3 models x 3 seeds, held-out test set.
# Writes results/run008_FINAL_CHECKPOINT.json. ~3 h on an M5.
nohup bash results/run008_run_all.sh > results/logs/run008_driver.log 2>&1 &
```

Run 008 first creates the splits (`results/run008_make_splits.py`:
`GroupShuffleSplit(test_size=0.2, random_state=2026)` applied to the original
training split, giving train 4097 / val 1270 / test 1025 with zero patient
overlap), then trains A/B/C × seeds 0–2, then evaluates the test set **once**
via `results/run008_eval_test.py`.

### Edge export and benchmarking

Export, then copy `edge/` plus the artifacts to each device (both expect
`~/eye-edge/`):

```bash
.venv/bin/python edge/export_onnx.py      # ONNX opset 17
.venv/bin/python edge/export_tflite.py    # via onnx2tf; needs the TF env
```

On the **N100** (`CONFIGS` defaults to `ort_fp32_cpu ov_fp32_cpu ov_fp32_gpu`):

```bash
OUT=run009/lat_512 ONNX=artifacts/run007c_eyeD512_fp32.onnx IMG=512 \
MANIFEST=artifacts/val_manifest_run009.csv \
REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
LABELS=D_eye,G,C,A,H nohup ./run_n100.sh > run009/lat_512.log 2>&1 &
```

On the **Pi 3B**:

```bash
OUT=run009/lat_512 MODEL=artifacts/run007c_eyeD512_fp32.tflite IMG=512 \
MANIFEST=artifacts/val_manifest_run009.csv \
REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
LABELS=D_eye,G,C,A,H N=300 CONFIGS="tfl_fp32_t4 tfl_fp32_t1" \
nohup ./run_pi.sh > run009/lat_512.log 2>&1 &
```

Energy on the N100 needs RAPL readable — this blocked run 005 entirely, and
sysfs permissions reset on every reboot:

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj \
               /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj \
               /sys/class/powercap/intel-rapl:0/intel-rapl:0:1/energy_uj
ENERGY_OUT=run009/energy_512 .venv/bin/python energy_n100.py all   # ~45 min
```

### Grad-CAM, calibration, quantization

```bash
# Run 010 — export, parity, sanity checks, faithfulness, IDRiD localisation,
# figures. Requires run008_FINAL_CHECKPOINT.json with updated_after_all_seeds.
.venv/bin/python results/run010_idrid_prep.py      # once, prepares IDRiD
nohup bash results/run010_run_final.sh > results/run010/log_final_driver.txt 2>&1 &

# Run 011 — temperature vs Platt scaling, on device and under INT8
.venv/bin/python results/run011_calibration.py
.venv/bin/python results/run011_int8_transfer.py

# Run 012 — OpenVINO iGPU INT8 collapse: diagnosis and fix (run on the N100)
.venv/bin/python results/run012_int8_variants.py
.venv/bin/python results/run012_min_repro.py
```

Run 012 traced the run-004 iGPU INT8 collapse (AUC 0.62) to an **OpenVINO
2026.4.1 GPU-plugin bug** in quantized depthwise `GroupConvolution` whose weight
`FakeQuantize` ranges vary along the kernel-width axis — not the driver, as
run 004 originally assumed. Excluding depthwise convs from quantization
(`ignored_scope=IgnoredScope(types=["GroupConvolution"])`) restores accuracy,
but INT8 still is not worth using.

---

## Conventions

| Artifact | Location |
|---|---|
| Checkpoints | `results/checkpoints/run{NNN}_*.pt` |
| Script snapshot for a run | `results/run{NNN}_*.py` |
| Raw stdout logs | `results/logs/run{NNN}_*.log` *(not committed)* |
| Environment freeze | `results/env_<machine>_freeze.txt` |
| Device results | `results/{n100,pi}_run{NNN}/` |

Update `results/EXPERIMENTS.md` every time a run finishes — do not let results
live only in terminal scrollback.

**Not committed** (see `.gitignore`): `data/`, virtualenvs, `results/logs/`,
and Flutter build output. Raw training logs are therefore absent; the per-epoch
history is preserved in `results/run*_summary.json`.

---

## Citing

The manuscript is unpublished. Until it appears, cite this repository and the
underlying datasets (ODIR-5K and, for the localisation analysis, IDRiD).

No licence file is present yet, so default copyright applies and the code
carries no reuse grant. Add a `LICENSE` if you intend others to build on this.
