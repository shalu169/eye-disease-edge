# Multi-Disease Retinal Screening on Affordable Edge Hardware — Brainstorm

Date: 2026-09-27

## 1. Motivation / Gap

- DR and glaucoma AI screening papers are dominant in the literature (high
  lab-reported accuracy, 92-98% sens/spec), but:
  - Real-world sensitivity drops to 51-86% outside controlled trials — poor
    cross-site/cross-camera generalization is the "most fundamental" limitation
    (Early Detection of Glaucoma and DR in Low-Resource Settings, PMC13467204).
  - Most edge-deployment papers (arXiv 2506.14834, ScienceDirect
    S1877050926014171) handle **one disease** on **one modality**.
  - Multi-label fundus models (ODIR-based: NASNetMobile, EfficientNet) chase
    accuracy only — no edge deployment numbers, no explainability, no
    uncertainty.
  - Explainable lightweight CNNs (PMC12387214, Grad-CAM/Grad-CAM++) exist but
    aren't benchmarked on real edge hardware or paired with calibrated
    uncertainty.
  - On-device frameworks (Sci Rep 2025, s41598-025-20990-y) run on-device but
    skip explainability/uncertainty and skip a human-factor validation.
  - IOP measurement outside clinic remains a flagged "genuine bottleneck" for
    glaucoma specifically.
  - **Nobody combines**: multi-disease + on-device XAI/UQ + real hardware
    benchmarking + field validation with non-specialist users, in one paper.

## 2. Working Research Question

Can a single lightweight, quantized multi-label model — deployed entirely
on-device — jointly screen DR, glaucoma, cataract, and AMD (± hypertensive
retinopathy) from one fundus photograph, with embedded calibrated uncertainty
quantification and pixel-level explainability (Grad-CAM/Grad-CAM++), at
accuracy matching specialist single-disease models and real-time throughput on
consumer-grade hardware — and does the confidence+explanation output
measurably improve triage decision accuracy/trust for non-specialist health
workers versus a black-box baseline?

Splits into two contributions:
1. **Systems contribution** — multi-disease lightweight model + on-device
   XAI/UQ, benchmarked for accuracy, latency, peak memory, power draw across
   two real affordability tiers.
2. **Human-factor contribution** — field/user study: does confidence +
   heatmap output change triage correctness/trust for non-ophthalmologist
   users vs. black-box output. Closest prior work: "Enhancing Community
   Vision Screening — AI Driven Retinal Photography for Patient Trust"
   (arXiv 2410.20309) — that paper doesn't do multi-disease + edge + UQ.

## 3. Target Hardware (already owned, measured 2026-09-27)

| | Pi 3B (`pi@192.168.188.108`) | N100 mini PC (`n100m`) |
|---|---|---|
| CPU | ARMv7 Cortex-A53, 4-core, 1.2GHz (currently undervoltage-capped ~600MHz) | x86_64, 4-core, up to 3.4GHz |
| RAM | 870MB (hard ceiling) | 15GB |
| GPU | none | Intel UHD (Alder Lake-N) iGPU, OpenVINO-capable |
| Storage | 59GB, 47GB free | 216GB NVMe, 199GB free |
| OS | Raspbian 11 (bullseye) | Debian 13 (trixie) |
| Python | 3.9.2 | 3.13.5 |
| Camera | **none detected** (`supported=0 detected=0`) | n/a (not needed if using dataset images) |
| Approx. price tier | ~$35 legacy SBC | ~$150-200 mini PC |
| Runtime path | TFLite (armv7 wheels) | OpenVINO INT8, iGPU/CPU |

This gives a genuine two-tier "affordability" axis for the paper instead of a
single arbitrary device — matches the low-resource-settings framing directly.

**Constraints to design around:**
- Pi's 870MB RAM is the binding constraint on model size — same ceiling class
  already hit in the KTS whisper.cpp project on this exact board. Expect only
  MobileNetV3-small / EfficientNet-lite0 class backbones (single-digit-MB
  quantized) to fit alongside OS + runtime + Grad-CAM buffers.
  → open question: does Grad-CAM/uncertainty overhead even fit in the Pi's
  remaining headroom, or is Pi limited to classification-only (explainability
  on N100 only)? Worth measuring early, before committing scope.
- Pi has no camera attached → the pipeline is inference-only on pre-captured
  dataset images for benchmarking; live capture demo (if wanted for the paper)
  needs either a Pi camera module or a phone-captured image handoff.
- N100 has no ML frameworks installed yet — first task is installing
  OpenVINO/ONNX Runtime + int8 quantization toolchain.

## 4. Candidate Datasets

- **ODIR-5K / ODIR-2019** — 8-class multi-label (DR, glaucoma, cataract, AMD,
  hypertension, myopia, other, normal), ~5000 patients, bilateral fundus pairs.
  Best fit for the multi-disease multi-label framing.
- **APTOS 2019** — DR severity grading, large, well-used baseline.
- **RIM-ONE / DRISHTI-GS / ORIGA** — glaucoma-specific fundus sets, useful for
  a glaucoma-only sanity baseline against specialist models.
- Possible cross-dataset generalization test: train on ODIR, eval on
  APTOS/RIM-ONE subsets to speak to the generalization gap directly (ties back
  to gap #1 in section 1).

## 5. Methodology Sketch

1. **Model**: multi-label CNN (MobileNetV3-small or EfficientNet-lite0
   backbone) + multi-head classifier, trained on ODIR-5K (+ supplementary
   single-disease sets for class balance).
2. **Compression**: post-training INT8 quantization — OpenVINO POT/NNCF path
   for N100, TFLite converter for Pi.
3. **Explainability**: Grad-CAM++ hook on last conv layer, computed on-device
   at inference time.
4. **Uncertainty**: calibrated confidence (temperature scaling or MC-dropout
   at inference, budget permitting given latency cost) — report
   ECE (expected calibration error) alongside accuracy.
5. **Benchmarking**: per-device fps, ms/image, peak RSS, power draw (measure
   via USB power meter or `powertop`/RAPL on N100; Pi lacks RAPL, would need
   inline USB meter), compared against Raspberry Pi/Jetson baselines reported
   in cited prior work.
6. **Field study**: recruit N non-ophthalmologist health workers, paired
   before/after or A/B on triage decisions with vs. without
   confidence+heatmap, measure decision accuracy + trust (Likert) +
   time-to-decision.

## 6. Risks / Open Questions to resolve with user

- Scope: is the field/human-factor study realistic to run (access to health
  workers, IRB/ethics approval needed for any human-subjects study, even
  informal)? If not feasible, paper could drop to systems-only contribution
  and cite arXiv 2410.20309's trust findings instead of replicating them.
- Does explainability/uncertainty need to run on the Pi at all, or is "Pi =
  classification-only baseline, N100 = full pipeline" an acceptable framing?
  This affects whether Pi's RAM ceiling becomes a paper finding rather than a
  blocker.
- Which journal exactly — IEEE JBHI/TBME favor systems+hardware rigor;
  Elsevier AIM/CBM favor methods+clinical-workflow rigor. Affects how much
  weight the human-factor arm needs.

## 7. Decisions (2026-09-27)

- **Target**: regular track (JBHI, or BSPC/CBM as backup), not the JBHI
  Oct-30 special issue — full scope (multi-disease training + two-device
  benchmark + field trust study + writing) isn't realistic in 33 days, and
  cutting the field study just to hit a deadline weakens the paper's second
  contribution. No hard deadline; do it properly.
- **Dev order**: Mac first (fast local iteration, train/debug baseline) →
  N100 (quantize + benchmark, has headroom to debug the INT8 pipeline) → Pi
  last (port once a working INT8 model is confirmed to fit Pi's 870MB
  budget).

## 8. Immediate next technical steps

1. ~~Set up local Python env on this Mac for model dev.~~ **Done 2026-09-27**:
   venv at `.venv` (Python 3.14.6), torch 2.14 + torchvision confirmed
   working with MPS acceleration on the M5. Also installed: scikit-learn,
   pandas, opencv-python-headless, timm, grad-cam, kaggle CLI.
2. **Blocked**: pull ODIR-5K — needs a Kaggle API token
   (`~/.kaggle/kaggle.json`), not present on this Mac. Waiting on user to
   provide it.
3. ~~Once dataset is in: quick multi-label baseline (non-quantized) on Mac to
   confirm feasibility~~ **Done 2026-09-27**: `baseline.py`, MobileNetV3-small
   (timm, pretrained), ODIR-5K `andrewmvd/ocular-disease-recognition-odir5k`
   (6392 per-eye images / 3358 patients, split by patient ID — no leakage),
   5 labels (D/G/C/A/H), 5 epochs on Mac M5/MPS. Val AUC:
   D(DR)=0.63, G(glaucoma)=0.85, C(cataract)=0.94, A(AMD)=0.79, H(hyp.)=0.65
   (H noisy — only ~20 val positives). **Feasibility confirmed** on a
   genuinely edge-sized backbone. DR AUC is weak and worth digging into
   before scaling up (see below) — everything else is untuned proof-of-concept,
   not a paper-ready number.
4. Install OpenVINO + NNCF on N100, quantize + benchmark there.
5. Port to Pi once a working INT8 model is under Pi's memory budget.

**Open question on DR's weak AUC (0.63 vs. literature's usual 0.9+):**
likely just under-training (5 epochs, no LR schedule, no stronger aug) rather
than a real ceiling — cataract/glaucoma already look strong with the same
setup. Needs more epochs + a proper train/val curve before concluding
anything, not worth guessing further without running it.
