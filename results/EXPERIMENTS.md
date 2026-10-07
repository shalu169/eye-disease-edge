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

---

## Run 004 — N100 edge benchmark of run002 (ONNX Runtime / OpenVINO, FP32 + INT8, CPU + iGPU)

- **Date**: 2026-10-02
- **Purpose**: first on-device numbers for the paper — does the run002 model
  keep its accuracy when exported and run on the ~$150 N100 tier, how fast is
  it, and what does INT8 post-training quantization cost/buy.
- **Model**: run002 checkpoint (`results/checkpoints/run002_baseline_tuned_best_epoch15.pt`),
  exported on the Mac to ONNX opset 17 (`edge/artifacts/run002_mnv3s_fp32.onnx`,
  6.1 MB, md5 `6dc9448de15b3bca9cbf07898a42be22`, identical on N100).
  PyTorch CPU reference on the Mac reproduces run002 exactly (mean AUC 0.8215).
- **INT8**: OpenVINO IR via NNCF 3.4.0 `nncf.quantize`, preset MIXED, 300
  calibration images sampled from the *train* split (`random_state=0`,
  `edge/artifacts/calib_manifest.csv`). No accuracy-aware tuning, no ignored
  scope. IR: `edge/artifacts/run002_mnv3s_int8.{xml,bin}` (2.5 MB total).
- **Eval set**: same 1270-image val split as runs 001-003
  (`edge/artifacts/val_manifest.csv`). Preprocessing re-implemented in
  NumPy/PIL to match `baseline.py` `val_tfm` (PIL bilinear resize 224x224,
  ImageNet normalize).
- **Hardware**: N100 mini PC (`n100m`, hostname `extraeye`), Intel N100 4C
  up to 3.4 GHz, governor `performance`, 15 GB RAM, Intel UHD iGPU
  (device id 0x46d1), Debian 13.6, kernel 6.12.107, freshly booted, no other
  workloads (no containers running). iGPU driver: Debian `intel-opencl-icd`
  22.43.24595.41.
- **Env**: Python 3.13.5, onnxruntime 1.30.0, openvino 2026.4.1, nncf 3.4.0,
  numpy 2.4.6, pillow 12.3.0. Full freeze: `results/env_n100_freeze.txt`.
- **Method**: one process per config (so peak RSS is attributable to it).
  Accuracy over full val split at batch 1. Latency: batch 1, 50 warmup +
  500 timed runs on one fixed preprocessed image, model-only (excludes JPEG
  decode/resize, which is reported separately as `preprocess_ms_mean`).
  OpenVINO compiled with `PERFORMANCE_HINT=LATENCY`, default threading.
  Single run per config, no repeats.
- **Results**:

  | config | mean AUC | D | G | C | A | H | max \|Δp\| vs torch | latency mean / p95 (ms) | preprocess (ms) | peak RSS (MB) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | ORT FP32 CPU | 0.8214 | 0.6947 | 0.8722 | 0.9376 | 0.8575 | 0.7452 | 6.3e-6 | 2.06 / 2.05 | 5.19 | 196 |
  | OV FP32 CPU | 0.8215 | 0.6947 | 0.8722 | 0.9376 | 0.8575 | 0.7453 | 6.2e-6 | 2.43 / 2.44 | 4.60 | 238 |
  | OV INT8 CPU | 0.8050 | 0.6673 | 0.8422 | 0.9137 | 0.8583 | 0.7436 | 0.92 | 1.97 / 1.98 | 5.17 | 242 |
  | OV FP32 iGPU | 0.8215 | 0.6944 | 0.8729 | 0.9375 | 0.8578 | 0.7450 | 0.059 | 3.03 / 3.08 | 4.36 | 613 |
  | OV INT8 iGPU | 0.6226 | 0.4650 | 0.5809 | 0.8444 | 0.6598 | 0.5627 | 0.999 | 3.24 / 3.31 | 4.44 | 657 |

- **Artifacts**: per-config JSON + stdout in `results/n100_run004/`
  (`results_<config>.json`, `log_<config>.txt`); scripts
  `results/run004_export_onnx.py`, `results/run004_bench_n100.py`
  (working copies in `edge/`); export summary
  `edge/artifacts/export_summary.json`; PyTorch reference probabilities
  `edge/artifacts/torch_cpu_val_probs.npy`.
- **Verdict**:
  - FP32 deployment is lossless: ORT and OpenVINO CPU match PyTorch to ~1e-6
    in probability, AUCs identical to run002. Paper can claim "no accuracy
    loss from export" for FP32 on N100.
  - Model is tiny for this CPU: ~2 ms/image model-only, so preprocessing
    (~5 ms JPEG decode + resize in Python) dominates. End-to-end ≈ 7 ms/image,
    i.e. well over 100 images/s single-stream — far beyond clinical need.
  - iGPU is *slower* than CPU at batch 1 (3.0 vs 2.1 ms) and uses ~3x RAM; its
    0.059 max prob shift is consistent with the GPU plugin's default FP16
    inference precision, AUC unaffected. Not worth it for this model.
  - Naive INT8 PTQ is a bad trade on N100: −0.0165 mean AUC (DR −0.027,
    glaucoma −0.030, cataract −0.024) and large per-image probability shifts
    (max 0.92, mean 0.12 — would break calibrated-uncertainty claims) for only
    ~0.1-0.5 ms saved. MobileNetV3's hard-swish/SE blocks are known to be
    PTQ-sensitive. **INT8 iGPU is broken (AUC 0.62)** — treat as a
    runtime/driver bug (old Debian OpenCL driver 22.43), not a model result;
    do not report as a number without investigating.
    **Update (run 012)**: not the driver. An OpenVINO GPU-plugin NaN in INT8
    depthwise convolutions; fixed by keeping depthwise convs in float. See Run 012.
  - **Not measured yet**: energy (RAPL `energy_uj` is root-only and `test` has
    no passwordless sudo), multi-run variance, thermal behaviour under
    sustained load.
  - **Next**: (1) energy via RAPL once root access is sorted; (2) if INT8 is
    needed for the Pi's RAM budget, retry with accuracy-aware PTQ / ignored
    scope on SE + classifier, or QAT; (3) Pi 3B TFLite run on the same
    val manifest for the cross-tier comparison.

---

## Run 005 — N100 energy per inference via RAPL (BLOCKED, not yet measured)

- **Date**: 2026-10-02
- **Status**: **no energy numbers yet.** RAPL `energy_uj` on the N100 is still
  root-only (`-r-------- root`). Polled every 60 s for 40 min (18:26-19:10 CEST,
  log `results/n100_run005/rapl_poll.log`), never became readable. User `test`
  has no passwordless sudo. Script is written, tested and deployed; run it
  once the permissions are fixed (commands below).
- **Purpose**: energy per inference (total and above-idle) for the four
  working run004 configs, to support the paper's edge-efficiency claim.
  `ov_int8_gpu` excluded (broken in run004, AUC 0.62).
- **Hardware**: same as run004, N100 (`n100m`). RAPL zones: `intel-rapl:0`
  = package-0, `intel-rapl:0:0` = core, `intel-rapl:0:1` = uncore (iGPU).
  `max_energy_range_uj` = 262143328850 for all three. Package power limits:
  PL1 (long_term) 10 W / 28 s window, PL2 (short_term) 25 W. Governor
  `performance`. Thermal sensors: `x86_pkg_temp` + coretemp (package, 4 cores).
- **Env**: unchanged from run004 (`~/eye-edge/.venv`: onnxruntime 1.30.0,
  openvino 2026.4.1).
- **Method (planned, implemented in `edge/energy_n100.py`)**:
  one process per config, plus separate idle baseline processes before
  (`idle pre`) and after (`idle post`) all configs. Per config: model load,
  page-cache warm of all 1270 val JPEGs, 50 warmup model runs + 50 warmup
  e2e runs, then two modes:
  - `model`: sustained batch-1 inference on one fixed preprocessed image;
  - `e2e`: per iteration `preprocess()` (JPEG decode + PIL bilinear resize +
    normalize, from `bench_n100.py`) + inference, cycling through the val images.
  Each mode: 3 repeats x 60 s, 30 s sleep (cooldown) before each repeat.
  Energy counters (package/core/uncore) sampled about every 1 s inside the loop;
  deltas integrated with wraparound correction via `max_energy_range_uj`.
  Recorded per repeat: energy (J), avg power (W), power during the second half
  of the window (steady state after the PL1 28 s window), inferences/s,
  mJ/inference per zone, per-second package power trace, process CPU time,
  system-wide CPU busy fraction (detects interference), CPU temps and
  frequencies before/after. Above-idle energy = (P_load − mean P_idle) /
  inferences/s. `summary.json` reports mean ± std (ddof=1) over repeats.
  Runtime is about 45 min.
- **Validation done**: `selftest` (wraparound arithmetic; synthetic counter
  with 7 J range wrapping every ~1.4 s recovers 5.0/2.0/0.5 W exactly, idle
  and busy loops). Full `all` pipeline ran on the N100 with the stub
  reader (`ENERGY_STUB=1`, 4 s x 2 repeats), and all JSONs and the summary
  came out. Real models loaded and ran. Stub-run throughputs (4 s windows,
  sampling loop included, *not* paper numbers): model-only ORT FP32 ~495/s,
  OV FP32 CPU ~406/s, OV INT8 CPU ~513/s, OV FP32 iGPU ~330/s; e2e
  128-141/s for all configs (preprocessing dominates, as in run004).
  These line up with run004 latencies.
- **Things to watch when it runs**: ORT CPU showed ~100% system-wide
  CPU busy at batch 1 (default intra-op thread pool spinning on all 4 cores),
  while iGPU showed ~25% (one core busy-waiting). So ORT's power will include
  spin-wait cost. This is the default deployment behaviour, but mention it in
  the paper. Package temp rose ~42→59 °C within 4 s of load, so check the
  recorded temps/second-half power for PL1 throttling.
- **To run** (after the user grants read access; sysfs perms reset on every reboot):
  ```
  # on the N100, as root:
  sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj \
                 /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj \
                 /sys/class/powercap/intel-rapl:0/intel-rapl:0:1/energy_uj
  # then (as test):
  ssh n100m 'cd ~/eye-edge && mkdir -p run005 && ENERGY_OUT=run005 nohup .venv/bin/python energy_n100.py all > run005/run_all.log 2>&1 &'
  # after about 45 min:
  scp 'n100m:eye-edge/run005/*' results/n100_run005/
  ```
  Subzones are optional, and the script skips them if they can't be read. Package is required.
- **Artifacts**: script `results/run005_energy_n100.py` (working copy
  `edge/energy_n100.py`, identical to `~/eye-edge/energy_n100.py`, md5
  `f10953efe8125d4b496b3453ee2c9523`); poll log `results/n100_run005/rapl_poll.log`.
- **Verdict**: none yet. The energy measurement is blocked on root
  permissions, not on tooling.

- **Update 2026-10-03**: superseded. RAPL access was granted on 2026-10-03 and the energy protocol in this entry was run as part of run 009 (224 and 512 px); see the Run 009 entry.

---

## Run 006 — Raspberry Pi 3B edge benchmark of run002 (TFLite FP32 / dynamic-range INT8 / full-integer INT8, 4 vs 1 threads)

- **Date**: 2026-10-02
- **Purpose**: the low-cost tier of the cross-tier comparison. Does the run002
  model keep its accuracy as TFLite on an ARMv7 Pi 3B, how fast is it, does it
  fit in the 870 MB RAM ceiling, and what does INT8 buy/cost? Methodology
  mirrors run 004 (N100) so the numbers are directly comparable.
- **Model**: run002 checkpoint via the *same ONNX file* benchmarked in run 004
  (`edge/artifacts/run002_mnv3s_fp32.onnx`, md5 `6dc9448de15b3bca9cbf07898a42be22`),
  converted on the Mac (CPU only) with onnx2tf 1.26.3 + TensorFlow 2.18.0
  (Python 3.12.14, the `.venv-export` venv of the `fundus-app-local-mode`
  worktree, used read-only). Full model, logits output `[1,5]`, input NHWC
  `[1,224,224,3]` float32. Unlike the mobile-app export (backbone only),
  this includes conv_head + classifier.
  - **FP32**: `edge/artifacts/run002_mnv3s_fp32.tflite`, 6.1 MB,
    md5 `0f1d05227e5d68f1de4daacdd6aebcc0`.
  - **DRQ (dynamic-range INT8)**: int8 weights, float activations
    (`Optimize.DEFAULT`, no representative data). `run002_mnv3s_drq.tflite`,
    1.7 MB, md5 `e5ad7f91ddef5f886cb333ccac20b4bc`.
  - **INT8 (full-integer PTQ)**: int8 weights + activations
    (`TFLITE_BUILTINS_INT8`), float32 I/O kept so preprocessing is identical.
    Calibrated on the 300 *train* images in `edge/artifacts/calib_manifest.csv`
    (the same images NNCF used in run 004; asserted zero overlap with val).
    `run002_mnv3s_int8.tflite`, 1.8 MB, md5 `dc4d23bfc04a3d9dd256088010f57e28`.
  - Both quantized models use **per-tensor weights for the single dense layer**
    (`_experimental_disable_per_channel_quantization_for_dense_layers=True`).
    Without this, TF 2.18 emits `FULLY_CONNECTED` op v12, which
    tflite-runtime 2.13.0 cannot load ("Didn't find op for builtin opcode
    'FULLY_CONNECTED' version '12'"). Mac-side AUC is essentially unchanged by
    this (DRQ 0.8164 → 0.8164, INT8 0.6002 → 0.5994; first-attempt summary kept
    in `results/logs/run006_tflite_export_summary_attempt1_fcv12.json`).
- **Mac pre-ship verification** (TF 2.18 `tf.lite.Interpreter`, full val split,
  `edge/artifacts/tflite_export_summary.json`): FP32 vs torch max |Δp|
  1.1e-5, mean 6.2e-7, AUCs identical to run002 (mean 0.8215). DRQ 0.8164.
  INT8 **0.5994**, i.e. broken (see INT8 diagnosis below).
- **Eval set**: same 1270-image val split (`edge/artifacts/val_manifest.csv`),
  same preprocessing as run 004 (PIL bilinear resize 224x224, ImageNet
  normalize), transposed to NHWC.
- **Hardware**: Raspberry Pi 3B (`pi@192.168.188.108`, hostname `raspberrypi`),
  BCM2837 4x Cortex-A53, running **32-bit** Raspbian 11 (armv7l), kernel
  6.1.21-v7+, 870 MB RAM (~700 MB `MemAvailable` before each run; the 99 MB
  swap was already 100 % used by other processes before the run), uptime 14 d,
  the board's own picam/LXDE desktop left running, kts whisper sweep **not**
  running (load avg 0.28 at start).
  **Power**: `get_throttled` = `0x50005` (bit 0 under-voltage *now*, bit 2
  throttled *now*, sticky bits 16/18 "has occurred") before and after every
  config. The ARM clock read
  600 MHz (half the rated 1.2 GHz) at every before/after sample. The cap is
  intermittent: dmesg cycles "Undervoltage detected / Voltage normalised"
  every few seconds, and one sample right after the suite (after 60 s idle)
  read `0x50000` @ 1.2 GHz. The clock was only sampled before/after each
  config, not continuously, so brief 1.2 GHz excursions during timing can't
  be ruled out (the int8_t1 distribution with mean < p50 hints at this). Treat
  all latencies as **"measured under an intermittent 600 MHz under-voltage cap,
  roughly 2x pessimistic vs a properly powered Pi 3B"**. Accuracy is
  clock-independent. SoC temp stayed 49-59 °C; the soft-temperature-limit
  bits (0x8 / 0x80000) were never set, so the throttling is power, not thermal.
- **Env (Pi)**: Python 3.9.2 in a venv at `~/eye-edge/.venv`.
  - `tflite-runtime==2.13.0`: PyPI wheel
    `tflite_runtime-2.13.0-cp39-cp39-manylinux2014_armv7l.whl`
    (sha256 `dcba7f3b…94616f6`). It is the newest armv7/cp39 build (pip index
    lists 2.7.0-2.13.0; `ai-edge-litert` has no armv7 wheel).
  - `numpy==1.26.4` (piwheels). The first resolve pulled numpy 2.0.2, which
    failed to import (missing `libopenblas.so.0`), so numpy was pinned below 2.
  - `pillow==11.2.1` (piwheels; 11.3.0 has no armv7 wheel and failed to build
    from source).
  - System package installed for this run: `libopenblas0` 0.3.13+ds-3+rpi1+deb11u1
    (apt, needed by the piwheels numpy).
  - No pandas/sklearn on the Pi: AUC is a rank (Mann-Whitney) implementation
    validated against `sklearn.roc_auc_score` to 1e-16 on the Mac (incl. ties).
    Pi AUCs were **re-computed with sklearn on the Mac** from the saved
    probabilities and match every JSON value exactly.
  - Full freeze: `results/env_pi_freeze.txt`.
- **Method**: one process per config, run sequentially by `run_pi.sh` under
  nohup with a 60 s idle gap between configs. Accuracy over the full val split
  at batch 1. Latency: batch 1, **20 warmup + 200 timed** runs on one fixed
  preprocessed image (run 004 used 50 + 500; reduced because the Pi is ~60x
  slower), model-only, with `preprocess_ms_mean` reported separately.
  `Interpreter(num_threads=4|1)`, default delegates (XNNPACK applies to the
  float graph). Peak RSS = `ru_maxrss`, cross-checked with `/usr/bin/time -v`
  (agrees to <1 MB). Single run per config, no repeats.
- **Results** (all values from `results/pi_run006/results_<config>.json`):

  | config | mean AUC | D | G | C | A | H | max \|Δp\| vs torch | mean \|Δp\| | latency mean / p50 / p95 / p99 (ms) | preprocess (ms) | load (s) | peak RSS (MB) |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | TFL FP32, 4 thr | 0.8215 | 0.6947 | 0.8722 | 0.9376 | 0.8575 | 0.7453 | 1.8e-5 | 7.4e-7 | 130.4 / 130.4 / 140.3 / 148.3 | 93.2 | 0.054 | 48.2 |
  | TFL FP32, 1 thr | 0.8214 | 0.6947 | 0.8722 | 0.9376 | 0.8575 | 0.7452 | 1.7e-5 | 7.4e-7 | 172.6 / 174.8 / 175.9 / 181.9 | 90.2 | 0.053 | 43.6 |
  | TFL DRQ, 4 thr | 0.8166 | 0.6819 | 0.8608 | 0.9344 | 0.8525 | 0.7535 | 0.699 | 0.055 | 165.4 / 164.9 / 195.9 / 212.0 | 93.4 | 0.057 | 39.3 |
  | TFL DRQ, 1 thr | 0.8166 | 0.6818 | 0.8604 | 0.9343 | 0.8531 | 0.7533 | 0.699 | 0.055 | 165.1 / 168.7 / 169.3 / 170.8 | 90.2 | 0.057 | 40.5 |
  | TFL INT8, 4 thr ⚠ | 0.5926 | 0.4944 | 0.6157 | 0.7247 | 0.5340 | 0.5941 | 0.983 | 0.189 | 78.3 / 78.1 / 78.5 / 79.4 | 93.3 | 0.060 | 37.8 |
  | TFL INT8, 1 thr ⚠ | 0.5861 | 0.4963 | 0.6144 | 0.7137 | 0.5281 | 0.5778 | 0.977 | 0.189 | 151.1 / 162.3 / 162.5 / 162.6 | 88.1 | 0.059 | 38.4 |

  ⚠ = accuracy-broken model; latency/RSS only meaningful as "what full-int8
  kernels cost on this CPU".
  Pi vs Mac TFLite (same file) max |Δp|: FP32 1.3e-5; DRQ 0.36; INT8 0.46-0.48.
  The quantized kernels are not bit-identical across x86/TF 2.18 and
  ARM/runtime 2.13, and the DRQ probabilities even differ slightly between 1
  and 4 threads.
- **INT8 diagnosis (Mac, CPU)**: full-integer PTQ breaks the model on train
  images too (calib[:100]: int8 vs fp32 mean |Δp| 0.198), so this is
  quantization error, not distribution shift. The TFLite QuantizationDebugger
  (64 train images, `edge/artifacts/tflite_int8_layer_debug.csv`) flags the
  SE-block global-average-pool `MEAN` ops (single-layer RMSE up to 12.7x the
  int8 step, against a median of 0.29 over 112 tensors). Two bounded repair
  attempts:
  (a) all 10 `MEAN` ops left in float (`run002_mnv3s_int8mixed.tflite`;
      verified via op dump: 10 float MEAN + 11 Q/DQ pairs). Val AUC 0.6002,
      unchanged.
  (b) the 14 tensors with RMSE/scale > 0.5 left in float (probe only, file
      not kept; `results/run006_int8sel_probe.py`, log
      `results/logs/run006_int8sel_probe.log`). Train mean |Δp| 0.200, unchanged.
  Conclusion: the error is spread through the network and accumulates, and
  no small set of layers explains it. TFLite's symmetric per-tensor
  activation PTQ (no bias correction) fails on this MobileNetV3-small, whereas
  NNCF MIXED on the same calibration set lost only 0.0165 AUC (run 004).
  Probe (b) used the pre-fix converter (per-channel FC). `int8mixed` was
  rebuilt after the fix, but came out byte-identical (md5
  `9abbce61e93ffbc17fbc5ad3c08325d1`), so the debugger path ignores the
  per-tensor-FC flag. Neither was run on the Pi.
- **Artifacts**: per-config JSON, stdout + `/usr/bin/time -v` logs, and val
  probabilities in `results/pi_run006/` (`results_<config>.json`,
  `log_<config>.txt`, `probs_<config>.npy`, `suite.log`); scripts
  `results/run006_export_tflite.py`, `results/run006_diagnose_tflite_int8.py`,
  `results/run006_bench_pi.py`, `results/run006_run_pi.sh`,
  `results/run006_int8sel_probe.py` (working copies in `edge/`); Mac logs
  `results/logs/run006_*.log`; export summary
  `edge/artifacts/tflite_export_summary.json`; Mac TFLite val probabilities
  `edge/artifacts/tflite_mac_{fp32,drq,int8,int8mixed}_val_probs.npy`; Pi env
  `results/env_pi_freeze.txt`. On the Pi everything is in `~/eye-edge/`.
- **Verdict**:
  - **FP32 TFLite on the Pi 3B is lossless**: max |Δp| 1.8e-5 vs PyTorch, AUCs
    identical to run002 (0.8215; the 1-thread H 0.7452 is a rounding-boundary
    flip). The paper can claim "no accuracy loss from export" on both tiers.
  - **RAM is a non-issue**: peak RSS 38-48 MB for every variant, about 4-6 %
    of the 870 MB board and under 7 % of the ~700 MB available. FP32 and INT8
    both fit with a large margin, so INT8 is **not needed** for memory on this
    model.
  - **Speed (600 MHz cap)**: FP32 4 threads 130 ms model-only + 93 ms
    preprocessing ≈ 225 ms/image end-to-end, about 4.4 images/s
    single-stream. That is clinically ample for screening. It is about 63x
    slower model-only than the N100 (ORT FP32 2.06 ms) and preprocessing
    is about 18x slower (93 vs 5.2 ms). Threads help little (1→4 threads:
    173→130 ms, 1.32x).
  - **DRQ is a bad trade on the Pi**: −0.005 mean AUC, large per-image shifts
    (max |Δp| 0.70), and *slower* than FP32 at 4 threads (165 vs 130 ms)
    because the hybrid kernels don't use multithreading/XNNPACK here (t1 ≈ t4).
    The only benefit is file size (1.7 vs 6.1 MB).
  - **Full-integer INT8 would be the fastest (78 ms, 1.7x over FP32) but is
    accuracy-broken (AUC 0.59, D below chance)**. Do not report it as a model
    result. It is a PTQ-method failure (see diagnosis), not a runtime bug:
    Mac and Pi agree it's broken.
  - **Caveats**: single run per config; clock sampled only before/after each
    config under an intermittently capping supply; no energy measurement; 200
    timed iterations vs 500 on the N100.
- **Next**: (1) fix the Pi's power supply/cable and repeat FP32 t4/t1 at
  a stable 1.2 GHz, with continuous `measure_clock` logging during timing,
  for a non-pessimistic number; (2) if INT8 speed matters for the
  paper, get a usable int8 TFLite via QAT or a PTQ path with bias
  correction (e.g. AI Edge Torch / PT2E quantizer, or NNCF → ONNX QDQ → TFLite),
  and keep the per-tensor-FC constraint for runtime 2.13; (3) speed up the
  Pi-side preprocessing (93 ms is ~42 % of end-to-end: PIL `draft()`
  JPEG downscale-on-decode or a smaller stored image), checking parity against
  the torch reference; (4) energy per image via a USB power meter.

---

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

---

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

---

## Run 009 — Edge re-benchmark at 512 px (N100 ORT/OpenVINO CPU+iGPU, Pi 3B TFLite) + N100 RAPL energy, 224 vs 512

> Phase 1 = latency/RAM/energy (val split, run007c/run007b/run002 weights); Phase 2 = held-out test-set parity of
> the run008 final checkpoint (section at the end). All numbers below are generated from the result JSONs by
> `results/run009_aggregate.py` (output: `results/run009_tables.md`). That script re-computes every AUC with
> sklearn from the saved per-image probabilities (15/15 configs match the JSON) and re-computes max |Δp| vs the
> PyTorch reference.

- **Date**: 2026-10-03
- **Purpose**: run 007 recommends the 512 px per-eye-D joint model as the paper's main model. 512 px costs about 5.2× the
  FLOPs of 224 px, and its edge latency, RAM and energy had not been measured. This run measures them on both tiers.
  It also produces the first N100 energy numbers (run 005 was blocked on RAPL permissions) for the 512 model and
  for run002 at 224. INT8 is skipped because runs 004/006 showed naive PTQ is harmful.
- **Models (Phase 1; latency, RAM and energy don't depend on the final weights)**:
  - 512: `results/checkpoints/run007c_joint5_eyeD_512_s1_best_epoch22.pt` (md5 `fff98cdd984211b690210d31ce14bc13`), exported on
    the Mac on CPU with `edge/export_onnx.py` (opset 17): `edge/artifacts/run007c_eyeD512_fp32.onnx` (6.1 MB, md5
    `8a8baa44ed6fad52a82f5b0af90fdac7`, identical on N100). TFLite FP32 via onnx2tf 1.26.3 / TF 2.18.0:
    `edge/artifacts/run007c_eyeD512_fp32.tflite` (6.1 MB, md5 `ff69fbe3780cd309c56403bf66dd6413`, identical on Pi), input NHWC `[1,512,512,3]`.
  - 384 (extra, a fallback data point; 512 turned out to fit, see below): `run007b_joint5_eyeD_384_best_epoch17.pt` →
    `run007b_eyeD384_fp32.onnx` (md5 `1c1aed32dea8447c69987d14a2cc97e9`), `.tflite` (md5 `f20f01262ae0e4f466690555b7ff444f`).
  - 224: the unchanged run002 files from runs 004/006 (`run002_mnv3s_fp32.onnx` / `.tflite`), re-run in the same session.
- **Preprocessing**: as `results/run007_common.py` / run007 `val_tfm`: PIL RGB, bilinear `Resize((S,S))`, ToTensor,
  ImageNet normalize (NCHW for ONNX, NHWC for TFLite). The stored images are already 512×512, so at 512 the resize
  is a no-op. The preprocess time is JPEG decode plus normalize at 4× the pixels of 224.
- **Eval set (Phase 1)**: the run001-007 1270-image val split (`edge/artifacts/val_manifest_run009.csv` = same files/order as
  `val_manifest.csv`, plus the per-eye `D_eye` column). The 512/384 models are scored with D = per-eye label (`D_eye`),
  and patient-level D is reported separately as "D patient". The 224 run002 model is scored with patient D, as in runs 004/006.
- **Mac parity (CPU)**: PyTorch CPU reference reproduces run007c exactly (D_eye 0.8131, D patient 0.7832, G 0.9090,
  C 0.9364, A 0.8493, H 0.8245) and run007b_384 (D_eye 0.7626, D patient 0.7339).
  - ONNX Runtime 1.30.0 (throwaway Mac venv, the repo venv has no ORT), full val 1270: max |Δp| 1.2e-5, mean 5.6e-7, AUCs
    identical (`results/n100_run009/mac_onnx_parity/`).
  - TFLite (TF 2.18 interpreter), full val 1270: 512 max |Δp| 3.0e-5, mean 1.2e-6, and 384 max 9.8e-6, mean 5.9e-7. AUCs identical
    (`edge/artifacts/run007c_eyeD512_tflite_export_summary.json`, `run007b_eyeD384_tflite_export_summary.json`).
- **Hardware / env**: unchanged from runs 004/005/006. Package lists are identical to `results/env_n100_freeze.txt` /
  `env_pi_freeze.txt`.
  - N100: up since 2026-10-02 21:48 IST, governor `performance`, kernel 6.12.107, nothing else running (load 0.00 at start).
  - Pi: up 15 d, governor `ondemand`, LXDE/picam desktop left running as in run 006, 99 MB swap already 100 % used by
    other processes before the run.
  - RAPL `energy_uj` (package, core, uncore) was readable by `test`. The energy driver checked this immediately before
    starting (`results/n100_run009/energy_all.out`).
- **Method**: identical latency protocol to runs 004/006, using the generalised scripts (model, size, manifest and reference are now arguments):
  - Setup: one process per config, then accuracy/parity over the manifest at batch 1, then model-only latency on one fixed preprocessed image.
  - Warmup and timed runs: N100 50 warmup + 500 timed. Pi 20 warmup + 200 timed.
  - Recorded: `preprocess_ms_mean` (decode + resize + normalize), model load time and `ru_maxrss` peak RSS.
  - N100: all 1270 val images. Pi: the first 300 val images (`--n 300`), to keep the Pi passes to about 5-9 min per config. Latency is unaffected.
  - Pi: `/usr/bin/time -v` cross-check of RSS. A background sampler logged `get_throttled`, `measure_clock arm`, `measure_temp`, MemAvailable and SwapFree every 5 s for the whole suite (`vcgencmd_samples.tsv`). `free -m` was recorded before and after every config, with a 60 s gap between configs.
- **Energy method**: `edge/energy_n100.py` (run 005 protocol, unchanged apart from model arguments):
  - Order: idle baseline process before and after, then per config `model` (fixed input) and `e2e` (preprocess + infer cycling through the 1270 val JPEGs, page-cache warm).
  - Repeats: 3 × 60 s each, 30 s sleep before each.
  - Sampling: RAPL package/core/uncore sampled about every 1 s with wrap correction.
  - Configs: ort_fp32_cpu, ov_fp32_cpu, ov_fp32_gpu. The 512 pass ran first (05:23-05:59 IST), then the 224 pass (05:59-06:35 IST), each with its own idle pre/post. Nothing else ran on the N100: idle system busy fraction was ≤0.0004, and my status checks were a few ssh logins only.
  - Above-idle = (P_load − mean P_idle of that pass) / inf/s.

- **Results — latency / RSS / parity** (val manifest, batch 1; from `results/{n100,pi}_run009/lat_<res>/results_<config>.json`):

  | device | res | config | n | mean AUC (5) | D AUC | D patient | max \|Δp\| | mean \|Δp\| | latency mean / p50 / p95 / p99 (ms) | preprocess (ms) | e2e ≈ (ms) | load (s) | peak RSS (MB) |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | N100 | 224 | ORT FP32 CPU | 1270 | 0.8214 | D 0.6947 | – | 6.3e-6 | 4.7e-7 | 2.07 / 2.03 / 2.06 / 2.92 | 4.97 | 7.0 | 0.068 | 196.6 |
  | N100 | 224 | OV FP32 CPU | 1270 | 0.8215 | D 0.6947 | – | 6.2e-6 | 4.7e-7 | 2.43 / 2.43 / 2.45 / 2.46 | 4.46 | 6.9 | 0.243 | 238.3 |
  | N100 | 224 | OV FP32 iGPU | 1270 | 0.8215 | D 0.6944 | – | 5.9e-2 | 3.7e-3 | 3.04 / 3.03 / 3.07 / 3.09 | 4.50 | 7.5 | 3.539 | 614.2 |
  | N100 | 384 | ORT FP32 CPU | 1270 | 0.8489 | D_eye 0.7626 | 0.7339 | 1.2e-5 | 5.2e-7 | 4.69 / 4.68 / 4.71 / 4.72 | 7.86 | 12.5 | 0.064 | 209.5 |
  | N100 | 384 | OV FP32 CPU | 1270 | 0.8489 | D_eye 0.7626 | 0.7339 | 7.9e-6 | 4.2e-7 | 5.07 / 5.07 / 5.09 / 5.10 | 7.22 | 12.3 | 0.242 | 247.5 |
  | N100 | 384 | OV FP32 iGPU | 1270 | 0.8488 | D_eye 0.7628 | 0.7340 | 5.3e-2 | 3.5e-3 | 4.99 / 4.99 / 5.04 / 5.05 | 6.83 | 11.8 | 3.471 | 614.3 |
  | N100 | 512 | ORT FP32 CPU | 1270 | 0.8665 | D_eye 0.8131 | 0.7832 | 1.2e-5 | 5.6e-7 | 8.29 / 8.24 / 8.28 / 9.69 | 7.60 | 15.9 | 0.057 | 222.1 |
  | N100 | 512 | OV FP32 CPU | 1270 | 0.8665 | D_eye 0.8131 | 0.7832 | 1.1e-5 | 3.8e-7 | 8.40 / 8.40 / 8.42 / 8.44 | 7.66 | 16.1 | 0.245 | 257.0 |
  | N100 | 512 | OV FP32 iGPU | 1270 | 0.8666 | D_eye 0.8130 | 0.7834 | 6.5e-2 | 3.0e-3 | 6.93 / 6.93 / 6.98 / 7.00 | 6.94 | 13.9 | 3.675 | 617.4 |
  | Pi 3B | 224 | TFL FP32 4 thr | 300 | 0.8177 | D 0.7101 | – | 1.0e-5 | 7.8e-7 | 132.9 / 132.2 / 145.1 / 163.9 | 93.5 | 226.4 | 0.055 | 47.3 |
  | Pi 3B | 224 | TFL FP32 1 thr | 300 | 0.8177 | D 0.7101 | – | 1.1e-5 | 7.8e-7 | 164.6 / 175.0 / 176.8 / 184.7 | 91.6 | 256.3 | 0.054 | 43.3 |
  | Pi 3B | 384 | TFL FP32 4 thr | 300 | 0.8583 | D_eye 0.8436 | 0.7726 | 6.8e-5 | 3.6e-6 | 314.3 / 313.5 / 335.2 / 348.8 | 184.3 | 498.6 | 0.055 | 55.0 |
  | Pi 3B | 384 | TFL FP32 1 thr | 300 | 0.8583 | D_eye 0.8436 | 0.7726 | 6.8e-5 | 3.6e-6 | 483.5 / 492.2 / 498.8 / 500.1 | 180.3 | 663.8 | 0.056 | 51.8 |
  | Pi 3B | 512 | TFL FP32 4 thr | 300 | 0.8563 | D_eye 0.8424 | 0.7967 | 8.8e-5 | 6.1e-6 | 512.5 / 510.6 / 535.5 / 554.4 | 226.9 | 739.4 | 0.055 | 64.5 |
  | Pi 3B | 512 | TFL FP32 1 thr | 300 | 0.8563 | D_eye 0.8424 | 0.7967 | 8.7e-5 | 6.1e-6 | 856.6 / 870.7 / 879.9 / 896.1 | 221.9 | 1078.5 | 0.062 | 64.5 |

  Pi AUCs are on 300 images, so they are not comparable to the 1270-image N100 AUCs. On those same 300 images the
  PyTorch reference gives identical AUCs for every Pi config. Pi vs Mac TFLite (same file, same 300 images): 512 max
  |Δp| 6.7e-5, 224 1.0e-5. `/usr/bin/time -v` max RSS on the Pi: 224 48.4/44.4 MB, 384 56.3/53.1, 512 66.0/66.0 (t4/t1).
  The e2e column is model-only mean + preprocess mean, computed, not measured.

- **224 vs 512 summary** (224 from runs 004/006 as published, with the same-session 224 re-run in brackets):

  | device / config | latency 224 (ms) | latency 512 (ms) | ×  | preprocess 224 → 512 (ms) | e2e 224 → 512 (ms) | peak RSS 224 → 512 (MB) |
  |---|---|---|---|---|---|---|
  | N100 ORT FP32 CPU | 2.06 [2.07] | 8.29 | 4.0 | 5.19 [4.97] → 7.60 | 7.3 → 15.9 | 196 [197] → 222 |
  | N100 OV FP32 CPU | 2.43 [2.43] | 8.40 | 3.5 | 4.60 [4.46] → 7.66 | 7.0 → 16.1 | 238 [238] → 257 |
  | N100 OV FP32 iGPU | 3.03 [3.04] | 6.93 | 2.3 | 4.36 [4.50] → 6.94 | 7.4 → 13.9 | 613 [614] → 617 |
  | Pi TFL FP32 4 thr | 130.4 [132.9] | 512.5 | 3.9 | 93.2 [93.5] → 226.9 | 224 → 739 | 48.2 [47.3] → 64.5 |
  | Pi TFL FP32 1 thr | 172.6 [164.6] | 856.6 | 5.0 | 90.2 [91.6] → 221.9 | 263 → 1079 | 43.6 [43.3] → 64.5 |

- **Results — N100 energy** (RAPL package, mean ± std over 3 × 60 s repeats, from
  `results/n100_run009/energy_{512,224}/summary.json`; idle = 6 repeats from pre + post idle processes of that pass):

  | res | config | mode | W load | W load, 2nd half | W idle | inf/s | mJ/inf total | mJ/inf above idle | pkg °C after |
  |---|---|---|---|---|---|---|---|---|---|
  | 512 | ORT FP32 CPU | model | 11.53 ± 0.21 | 10.09 ± 0.35 | 2.59 ± 0.08 | 95.6 ± 3.8 | 120.7 ± 2.5 | 93.6 ± 1.4 | 69 |
  | 512 | ORT FP32 CPU | e2e | 10.99 ± 0.01 | 10.66 ± 0.03 | 2.59 ± 0.08 | 59.1 ± 0.5 | 186.1 ± 1.7 | 142.3 ± 1.3 | 75 |
  | 512 | OV FP32 CPU | model | 11.50 ± 0.01 | 9.93 ± 0.00 | 2.59 ± 0.08 | 90.1 ± 0.3 | 127.7 ± 0.3 | 99.0 ± 0.3 | 79 |
  | 512 | OV FP32 CPU | e2e | 11.08 ± 0.01 | 10.53 ± 0.04 | 2.59 ± 0.08 | 54.6 ± 0.4 | 202.9 ± 1.3 | 155.5 ± 1.0 | 80 |
  | 512 | OV FP32 iGPU | model | 10.07 ± 0.01 | 10.09 ± 0.01 | 2.59 ± 0.08 | 144.6 ± 0.2 | 69.6 ± 0.2 | 51.7 ± 0.2 | 76 |
  | 512 | OV FP32 iGPU | e2e | 9.81 ± 0.01 | 9.84 ± 0.01 | 2.59 ± 0.08 | 74.0 ± 0.1 | 132.6 ± 0.3 | 97.6 ± 0.2 | 75 |
  | 224 | ORT FP32 CPU | model | 11.32 ± 0.17 | 10.48 ± 0.42 | 2.61 ± 0.05 | 428.5 ± 16.2 | 26.4 ± 0.6 | 20.3 ± 0.4 | 69 |
  | 224 | ORT FP32 CPU | e2e | 9.70 ± 0.01 | 9.73 ± 0.01 | 2.61 ± 0.05 | 139.3 ± 0.6 | 69.6 ± 0.4 | 50.9 ± 0.3 | 72 |
  | 224 | OV FP32 CPU | model | 11.31 ± 0.03 | 10.07 ± 0.11 | 2.61 ± 0.05 | 346.2 ± 2.4 | 32.7 ± 0.1 | 25.1 ± 0.1 | 76 |
  | 224 | OV FP32 CPU | e2e | 10.66 ± 0.02 | 10.69 ± 0.02 | 2.61 ± 0.05 | 141.6 ± 0.1 | 75.3 ± 0.1 | 56.8 ± 0.1 | 77 |
  | 224 | OV FP32 iGPU | model | 9.31 ± 0.02 | 9.34 ± 0.03 | 2.61 ± 0.05 | 333.1 ± 0.9 | 28.0 ± 0.1 | 20.1 ± 0.1 | 73 |
  | 224 | OV FP32 iGPU | e2e | 9.00 ± 0.01 | 9.02 ± 0.02 | 2.61 ± 0.05 | 128.3 ± 0.0 | 70.2 ± 0.1 | 49.8 ± 0.1 | 73 |

  - Idle package power: 512 pass 2.587 ± 0.076 W, 224 pass 2.610 ± 0.048 W (n = 6 each). Idle core about 0.016 W, uncore 0.000 W.
  - The uncore zone (iGPU) reads 0 W in every CPU config and 0.28-1.25 W in the iGPU configs, so the counter works.
  - Power limits: PL1 10 W / 28 s, PL2 25 W.
  - Process CPU: ORT uses 4.0 cores in both modes (its intra-op pool spins, as seen in run 005).
    OV CPU uses 3.8-4.0 cores for model-only and about 2.4 cores for e2e. iGPU uses 1.0 core (one core busy-waiting, system busy 0.25).

- **Throttling / thermal**:
  - **N100 is PL1-limited under sustained load.** CPU configs start at 12-14 W (first 10 s) and settle at 10.0 W in
    the last 10 s of every 60 s window. All-core frequency after the window is 2.57-2.89 GHz (vs 3.4 GHz boost).
    So sustained model-only throughput is below the single-shot latency benchmark:
    - 512 ORT: 95.6 inf/s ≈ 10.5 ms vs 8.29 ms latency.
    - 224 ORT: 428.5 ≈ 2.33 ms vs 2.07.
    - iGPU is unaffected: 512 at 144.6 inf/s ≈ 6.9 ms, equal to its latency. Its package power sits at about 10 W from the start.
    The "W load" column averages over the PL2 → PL1 transition. "2nd half" is the steady state.
  - Package temperature peaked at 80 °C. The 30 s cooldown did not return the package to idle temperature (42 → 62 °C
    before-repeat over the course of a pass), so later configs in a pass start warmer. Run order was ORT → OV CPU →
    iGPU in both passes.
  - **Pi**: under-voltage throttled throughout, as in run 006.
    - At 4 threads `get_throttled` was `0x50005` @ 600 MHz in 100 % of 5-s samples at 512 (67/67) and 384 (44/44), and 95 % at 224 (19/20, one `0x50000` @ 1.2 GHz sample).
    - At 1 thread, 96-98 % of samples were at 600 MHz, with rare 1.2 GHz excursions (2/101 at 512). This explains the 1-thread mean < p50 pattern.
    - Before/after of every config: `0x50005` @ 600 MHz, except 224 t1 "after" `0x50000` @ 1.2 GHz.
    - SoC temperature 42-54 °C, soft-temp bits never set.
    - All Pi latencies are **under a ~600 MHz under-voltage cap**, roughly 2× pessimistic vs a properly powered Pi 3B.
    - The dmesg "undervoltage" line count stayed at 1560 for the whole session (the ring buffer doesn't track it usefully), so the sampler is the evidence.
    - Full table: `results/run009_tables.md`.
  - **Pi memory**: MemAvailable never fell below 666 MB during any config (≈700 MB before). Peak RSS of the 512 process is
    64.5 MB (`time -v` 66.0 MB), about 7.4-7.6 % of the 870 MB board. Swap was 0 MB free throughout, but it was already full
    from other processes before the run, and it never needed to grow. The 512 model **fits with a large margin**.

- **Phase 2: final model, held-out test-set parity**:
  - Checkpoint: from `results/run008_FINAL_CHECKPOINT.json` (`updated_after_all_seeds: true`, seen 13:03 CEST). That is
    `results/checkpoints/run008A_joint5_eyeD_512_s2_best_epoch21.pt` (md5 `c94c9a7b1067e18b8aaeec64b61953d3`), the median-validation
    seed of 3, selected without using the test set.
  - Exported exactly as in Phase 1: `edge/artifacts/run008final_fp32.onnx` (md5 `091fd482568d8bcdbe57dc66a2e3991c`) and
    `run008final_fp32.tflite` (md5 `282fbcc31ef06048fc3710677d655c89`). md5 is identical on the devices.
  - Test set: `results/run008_splits/test.csv`, 1025 images (copied byte-identical to `edge/artifacts/test_manifest_run008.csv`).
    Disjoint from the old val split, with 55 images overlapping the old run004 calibration list (irrelevant: no INT8 here).
    Only the 1025 test JPEGs were copied to the devices. The md5-of-md5s over all 1025 files is identical to the Mac on both.
  - The PyTorch CPU reference is computed on the Mac on the same images (`edge/artifacts/run008final_torch_cpu_test_probs.npy`,
    `run008final_export_summary.json`). D is scored on per-eye labels (`D_eye`), G/C/A/H on ODIR patient labels, and
    patient-level D is reported separately.
  - Devices: N100 full test set for ort_fp32_cpu, ov_fp32_cpu, plus ov_fp32_gpu as an extra. Pi tfl_fp32_t4 on the full test set. Same
    scripts and latency protocol as Phase 1. AUCs are re-computed with sklearn on the Mac from the saved probabilities and
    match every JSON value.

  | runtime | n | D_eye | G | C | A | H | mean (5) | D patient | max \|Δp\| vs torch | mean \|Δp\| vs torch |
  |---|---|---|---|---|---|---|---|---|---|---|
  | PyTorch CPU (Mac, reference) | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | – | – |
  | ONNX Runtime CPU (Mac) | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | 2.2e-5 | 6.4e-7 |
  | TFLite FP32 (Mac, TF 2.18) | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | 7.0e-5 | 3.0e-6 |
  | **N100 ORT FP32 CPU** | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | 2.5e-5 | 6.6e-7 |
  | **N100 OV FP32 CPU** | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | 1.0e-5 | 4.4e-7 |
  | N100 OV FP32 iGPU (extra) | 1025 | 0.8083 | 0.8913 | 0.8979 | 0.8625 | 0.8129 | 0.8546 | 0.7697 | 6.4e-2 | 3.4e-3 |
  | **Pi 3B TFLite FP32, 4 thr** | 1025 | 0.8093 | 0.8916 | 0.8973 | 0.8616 | 0.8126 | 0.8545 | 0.7709 | 1.3e-4 | 5.8e-6 |

  - Latency of the final model in the same runs:
    - N100: ORT 8.31 ms, OV CPU 8.47 ms, iGPU 6.96 ms model-only. Preprocess 6.7-8.8 ms. RSS 223 / 257 / 614 MB.
    - Pi t4: 512.8 ms model-only + 226.4 ms preprocess. RSS 65.4 MB (`time -v` 67.0 MB).
    - These are identical within noise to the Phase 1 run007c numbers, as expected for the same architecture.
  - Pi throttling during the test run: 172/172 samples `0x50005` at 600 MHz, 42.9-56.9 °C, MemAvailable ≥ 663 MB.
  - **Result**: every FP32 CPU edge runtime reproduces the PyTorch test-set AUCs to all 4 decimals on every label. The
    final-model accuracy claims hold on-device on both tiers. The iGPU (FP16 internally) moves individual AUCs by ≤0.001
    (D_eye −0.0010, D patient −0.0012). That is within noise but not bit-faithful, so report CPU as the reference deployment.
    The Pi's 1.3e-4 max |Δp| is the largest shift seen, still about 4 orders below anything that changes a rank-based metric.

- **Verdict (Phase 1)**:
  - **512 px is deployable on both tiers at FP32 with no accuracy loss from export.**
    - ORT / OV CPU match PyTorch to ~1e-5 (max |Δp| 1.1-1.2e-5 on 1270 val images, AUCs identical).
    - TFLite on the Pi matches to 8.8e-5 (Mac TFLite 3.0e-5). Slightly larger than at 224, which is expected with 4× more pixels in the reductions. Still 3 orders of magnitude below any AUC-relevant shift.
  - **N100**: 512 costs 3.5-4.0× the CPU latency of 224 (8.3-8.4 ms model-only, ~16 ms e2e ≈ 60 img/s) and +25 MB RSS.
    - The **iGPU becomes the fastest and most efficient backend at 512**: 6.9 ms, 144.6 inf/s sustained, 69.6 mJ/inf total and 51.7 mJ/inf above idle. That is 42-45 % less energy per inference than CPU (120.7 / 127.7 mJ).
    - The iGPU costs about 2.8× the RAM (617 MB) and FP16-level probability shifts (max |Δp| 0.065, mean 0.003, AUC within ±0.0002).
    - At 224, ORT CPU and the iGPU are tied on energy (26.4 vs 28.0 mJ/inf).
    - Energy per inference at 512 is 4.6× (ORT, 120.7 vs 26.4 mJ) and 2.5× (iGPU) that of 224. End-to-end it is 2.7× (ORT, 186 vs 70 mJ) and 1.9× (iGPU, 133 vs 70 mJ), because JPEG decode and normalize are a fixed per-image cost.
    - Even the worst case (OV CPU e2e, 203 mJ/image) is about 0.06 Wh per 1000 screened eyes.
  - **Pi 3B**: 512 is **slow but practical** for screening: 512 ms model-only + 227 ms preprocess ≈ 0.74 s/image
    end-to-end at 4 threads (≈1.35 img/s) under the 600 MHz cap, vs 0.23 s at 224.
    - Model-only 3.9× (t4) / 5.2× (t1) of 224, close to the FLOP ratio.
    - Threads help more at 512 (1.67× t1→t4) than at 224 (1.24×).
    - Preprocessing becomes 31 % of e2e and is the obvious optimisation target (NumPy float normalize of 786k pixels at 600 MHz).
    - 384 px is a middle point: 314 ms model + 184 ms preprocessing ≈ 0.50 s, 55 MB. It is **not needed as a fallback**: 512 fits in RAM and is well under 1 s/image.
  - INT8 not run (by design). Pi energy not measured (no power meter).
  - Devices released at 13:25 CEST (this file is the signal). Nothing is left running on either device.
- **Issues / caveats**:
  - Single latency run per config. Pi parity on the val split used 300 images only. Phase 2 test parity is full (1025).
  - N100 sustained throughput is PL1-limited, so latency-bench numbers are best-case single-stream.
  - The previous (stopped) agent left an un-finished export (`results/logs/run009_export_onnx_run007c_512.log`, no summary). The ONNX was regenerated from scratch here.
  - No ORT in the Mac repo venv. Mac ORT parity used a throwaway `uv` venv at `/tmp/run009-ort` (onnxruntime 1.30.0, numpy 2.5.3, pillow 12.3.0).
  - `/usr/bin/time` does not exist on the N100. RSS there is `ru_maxrss` only, as in run 004.
- **Artifacts**:
  - Generalised scripts (model, size, manifest and reference as arguments; no-argument defaults reproduce runs 004/005/006): `edge/{export_onnx,export_tflite,bench_n100,energy_n100,bench_pi}.py`, `edge/run_pi.sh`, new `edge/run_n100.sh`.
  - Script snapshots: `results/run009_{export_onnx,export_tflite,bench_n100,energy_n100,bench_pi,aggregate}.py`, `results/run009_run_{pi,n100}.sh`. The device-side drivers are inside the result dirs (`lat_all.sh`, `lat_384.sh`, `energy_all.sh`).
  - N100: `results/n100_run009/` with:
    - `lat_{224,384,512}/` (results/log/probs per config) and suite logs;
    - `energy_{224,512}/` (per-config and idle JSONs with per-second power traces, logs, `summary.json`);
    - `mac_onnx_parity/`.
  - Pi: `results/pi_run009/lat_{224,384,512}/` (results/log incl. `time -v`/probs per config, `vcgencmd_samples.tsv`) and suite logs.
  - Phase 2: `results/{n100,pi}_run009/test_final/` + `test_final.log`, `results/n100_run009/mac_onnx_parity_test/`, launch script `results/n100_run009/phase2_launch.sh`, `edge/artifacts/run008final_*`.
  - Mac logs: `results/logs/run009_export_*.log`.
  - Tables: `results/run009_tables.md`.
  - Export summaries: `edge/artifacts/run007c_eyeD512_{export,tflite_export}_summary.json`, `run007b_eyeD384_*`.

---

## Run 010 — Grad-CAM: reference vs edge implementation, sanity checks, faithfulness, lesion localisation, on-device cost

- **Date**: 2026-10-03
- **Status**: DONE on the final model. The development pass on the run007c
  checkpoint is kept in `results/run010/tables_dev.md` and the `*_dev*` files.
  It gave the same qualitative picture, except that faithfulness against the
  centre-bias baseline was weaker (see Verdict).
- **Purpose**: the paper claims an on-device Grad-CAM path that needs no
  automatic differentiation on the device. Literature (Adebayo 2018, Arun 2021,
  Saporta 2022, Sayres 2019, Ghassemi 2021) warns that saliency maps can be
  independent of the model, unfaithful, and poor at small lesions. This run
  tests (1) that the edge implementation is numerically the same Grad-CAM as
  the PyTorch reference, (2) Adebayo's randomisation sanity checks, (3)
  deletion/insertion faithfulness against random and centre-bias baselines,
  (4) localisation against expert pixel-level DR lesion masks (IDRiD), and
  (5) its latency/memory cost on the N100 and the Pi 3B.
- **Model**: the run 008 final checkpoint from `results/run008_FINAL_CHECKPOINT.json`
  (`updated_after_all_seeds: true`, median-validation seed):
  `results/checkpoints/run008A_joint5_eyeD_512_s2_best_epoch21.pt` (md5
  `c94c9a7b1067e18b8aaeec64b61953d3`). It is a 512 px joint 5-label model with
  per-eye DR, and it never saw the run 008 TEST split. The full pipeline ran
  unattended via `results/run010_run_final.sh` (12:59-15:27 CEST, Mac CPU only).
  Development work used `run007c_joint5_eyeD_512_s1_best_epoch22.pt`, whose
  training set contains the TEST images, so its test-set numbers are not
  reported as results.
- **Grad-CAM definition** (`results/run010_common.py`): target layer = output
  of `forward_features` (= `blocks[5]`, 1x1 ConvBnAct 96→576, 16x16 at 512 px),
  i.e. the last conv feature map before GAP. Per label c: weights = spatial mean
  of d logit_c / dA (pre-sigmoid logit), map = ReLU(Σ_k w_k A_k), bilinear
  upsample to 512 (align_corners=False), divided by the per-map maximum for
  display. All metrics are invariant to that scaling.
- **Edge path** (`edge/gradcam_numpy.py`): the backbone (ONNX or TFLite) outputs
  only the 576x16x16 feature map. NumPy computes GAP → conv_head (1x1 = dense
  1024x576) → hardswish → classifier (5x1024) → sigmoid, and the Grad-CAM weights
  analytically: d logit_c/d pooled = (W2[c] ⊙ hardswish'(z)) W1, divided by HW
  (the gradient is the same at every position because GAP is linear). Bilinear
  upsampling uses two interpolation matrices that reproduce torch exactly. The
  idea is ported from the Flutter app's `lib/inference/gradcam.dart` in the
  `fundus-app-local-mode` worktree (read only; nothing there was modified).
  Exports: `results/run010_export_edge.py` (torch 2.14 ONNX opset 17: backbone,
  full model, head weights `.npz`) and `results/run010_export_tflite.py`
  (onnx2tf 1.26.3 + TF 2.18.0 from the worktree's `.venv-export`, read only,
  CPU). Final artifacts: `edge/artifacts/run010_final_backbone_fp32.onnx` (3.7 MB,
  md5 `2f1573abbfef08f3f5a17beed9e1ecc3`), `run010_final_full_fp32.onnx` (6.1 MB,
  `ccb496758103a0ec337ff04c103030a2`), `run010_final_backbone_fp32.tflite`
  (3.7 MB, `b448d16361063b27e1fda50b4b6387c3`, output NHWC [1,16,16,576]),
  `run010_final_full_fp32.tflite` (6.1 MB, `aedea11f938de6d1c6075184403c8c1a`),
  `run010_final_head.npz` (`9829700c114722c134275d3eb3555d06`). **Note:** onnx2tf
  rewrites its input ONNX in place (the simplified graph), so the md5s in
  `run010_final_export.json` (taken right after the torch export) differ from the
  files on disk. Every ONNX number in this entry (Mac parity, faithfulness, N100)
  was measured on the rewritten files listed here, and they pass parity.
- **Environments (Mac, CPU only; MPS not used)**: `.venv` (Python 3.14,
  torch 2.14.0, timm 1.0.30, grad-cam 1.5.7, numpy 2.5.3, scipy 1.18.1,
  scikit-learn 1.9.1, matplotlib 3.11.2, opencv 5.0.0) and a new
  `.venv-run010` (Python 3.12.14, onnxruntime 1.30.0 = the N100 version,
  numpy 1.26.4 = the Pi version, torch 2.14.1, timm 1.0.30; freeze in
  `results/run010/env_venv_run010_freeze.txt`). onnxruntime was not in `.venv`,
  and the shared training venv was left untouched.
- **IDRiD provenance**: Indian Diabetic Retinopathy Image Dataset (Porwal et al.
  2018, Data 3(3):25, doi 10.3390/data3030025), "A. Segmentation" part: 81
  images (54 train + 27 test, 4288x2848) with pixel masks for microaneurysms (81
  images), haemorrhages (80), hard exudates (81), soft exudates (40) and the
  optic disc (81). Source: Kaggle mirror
  `aaryapatel98/indian-diabetic-retinopathy-image-dataset` (dataset id 472549,
  uploaded 2020-01-11, 7967 downloads), downloaded 2026-10-03 with the
  kaggle CLI and the existing token. Zip sha256
  `68c20658a41c695828377bfc347b97b0d9d658f0dd1ce72609d4d0cec7330059`
  (`data/idrid/zip.sha256`, Kaggle metadata in
  `data/idrid/kaggle_dataset-metadata.json`). **Licence: CC BY 4.0.** The zip
  contains the original `LICENSE.txt` ("IDRiD dataset (c) by Prasanna Porwal,
  Samiksha Pachade and Manesh Kokare … licensed under a Creative Commons
  Attribution 4.0 International License") and the full CC-BY-4.0 text, and the
  Kaggle page lists the same licence. No login-gated source was used. Cite
  porwal2018indian, not the mirror.
- **IDRiD preprocessing** (`results/run010_idrid_prep.py`, ODIR-like): FOV =
  grey > 15. The disc is cut at the top and bottom in every IDRiD image (81/81),
  so the square crop is taken around the disc: side = horizontal FOV extent
  (3410-3802 px), vertical centre = mean row of the widest FOV rows. The parts
  outside the image are padded black (like ODIR's inscribed-circle crop, but
  with black bands where IDRiD's disc is truncated). The image is then resized to
  512 (PIL bilinear). Masks get the identical crop/pad. Each output pixel's
  lesion fraction is computed with the PIL BOX filter, and the pixel counts as
  lesion if that fraction is > 0, so microaneurysms (~7x downscale) are kept.
  Mean mask area in the 512 FOV: MA 0.27 %, HE 1.74 %, EX 1.89 %, SE 0.62 %. The
  mask alignment was checked visually (`paper/figs/run010_idrid_overlays.png`
  top row).
- **Methods**:
  - *Reference cross-check* (`results/run010_check_reference.py`): 64 seeded test
    images × 5 labels against `pytorch_grad_cam.GradCAM(target_layers=[model.blocks[-1]])`.
  - *Parity* (`results/run010_parity_edge.py`): 300 seeded test images × 5
    labels, ONNX Runtime 1.30.0 and TFLite (TF 2.18 interpreter), FP32. The
    NumPy head is also run on the torch feature map, which isolates the head
    maths.
  - *Sanity checks* (`results/run010_sanity.py`): 100 seeded test images × 5
    labels. Cascading re-initialisation from the top (classifier → conv_head →
    blocks.5 … blocks.0 → stem) to the architecture's own init: weights of a
    fresh `timm.create_model(..., pretrained=False)`, seed 0, BN reset to
    γ=1, β=0, μ=0, σ²=1. Independent re-initialisation of blocks.5 only.
    Similarity: Spearman ρ (128x128 subsample) and SSIM (σ=1.5) of the 512 maps.
    Reference levels: the same label's map of a different image (similarity
    from generic fundus layout alone) and the centre-bias map.
    *Data randomisation* (model trained on permuted labels,
    `results/run010_train_randlabels.py`, written but **not run**): at 512 px it
    needs about 30 min on MPS, which run 008 B/C were using throughout, or about
    11 h on CPU (measured 0.32 s/image per training step).
  - *Faithfulness* (`results/run010_faithfulness.py`): every positive TEST image
    per label (D = per-eye DR, 285; G 62; C 62; A 42; H 34; 485 pairs on 448
    images). Pixels are ranked by the saliency map. Deletion/insertion is done at
    33 fractions (0, 1/32 … 1), with blur fill (Gaussian σ=10 px, primary) and
    ImageNet-mean fill. The target label's sigmoid probability is tracked, and the
    AUC is a trapezoid over the fraction. Baselines: smooth random (U(0,1) on
    16x16, upsampled; 1 draw per image) and centre bias (Gaussian, σ=0.25·512).
    95 % bootstrap CIs (2000 resamples over images) and paired Wilcoxon tests.
    Perturbed images were scored with the exported full-model ONNX in ORT
    (checked against torch, max logit diff < 1e-3, asserted). It is about 4x
    faster than torch CPU.
  - *Localisation* (`results/run010_localisation.py`): DR Grad-CAM on the 81
    IDRiD images, scored inside the FOV. Pointing game, strict and with ±15 px
    tolerance. Energy pointing game (EPG = share of the map's mass inside
    lesions; chance = lesion area fraction). Pixel AUROC and AP (every 2nd
    row/column). Baselines: smooth random (mean over 50 draws per image), centre
    bias, and the *glaucoma* Grad-CAM as a wrong-class control. Results are
    reported for the union of the four lesion types and for each type.
  - *Figures* (`results/run010_figures.py`): selection is a seeded uniform random
    draw within each category. The operating threshold per label = 90 %
    sensitivity on VALIDATION (`run008_common.thr_at_sens`). Selections are
    logged in `results/run010/figure_selection_final.json` and
    `figure_idrid_selection_final.json`.
  - *On-device cost* (`edge/gradcam_bench.py`, driver `edge/run010_device.sh`,
    bundle `edge/artifacts/run010_bundle_final/` sent over ssh with tar,
    `COPYFILE_DISABLE=1`, md5s checked on arrival). This ran only after
    `results/run009_ENTRY.md` existed. One process per config:
    `plain` = full model; `head` = backbone + NumPy head (probabilities only);
    `gradcam1` = + Grad-CAM for DR (512 px, normalised); `gradcam5` = all 5
    labels. Timing is batch 1 on one fixed preprocessed image, excluding JPEG
    decode/resize, which run 009 measures. Peak RSS = `ru_maxrss` (Pi
    cross-checked with `/usr/bin/time -v`). Before timing, each config checks
    its output against the Mac PyTorch-autograd probabilities and maps of 3
    test images in `run010_final_fixture.npz` (maps stored as float16).
    N100 (`n100m`, ~/eye-edge venv: Python 3.13.5, ORT 1.30.0, OpenVINO 2026.4.1,
    numpy 2.4.6; governor performance): 50 warmup + 500 timed runs. Runtimes:
    ORT default, ORT with `session.intra_op.allow_spinning=0`, and OpenVINO CPU
    (`PERFORMANCE_HINT=LATENCY`). Pi 3B (~/eye-edge venv: Python 3.9.2,
    tflite-runtime 2.13.0, numpy 1.26.4): 4 threads, 10 warmup + 100 timed runs.
    `get_throttled`, clock and temperature were recorded before/after each
    config, and the 30 s idle gap between configs is the same on both devices.
    Nothing was installed on either device.

- **Results (final model; every table generated by `results/run010_tables.py` from the JSONs)**:

  ### Reference Grad-CAM vs pytorch-grad-cam
  
  64 test images x 5 labels = 320 maps. Low-res (16x16, min-max) max |diff| 1.5e-06; 512 px library-style post-processing max |diff| 1.4e-06; our display map vs library output max |diff| 1.5e-06, min Spearman 0.9999999999. Grad-CAM forward vs plain forward prob max |diff| 1.2e-06.
  
  ### Edge Grad-CAM parity vs PyTorch autograd (Mac)
  
  | runtime | images / maps | feature map max abs diff | prob max diff, plain full model | prob max diff, backbone + NumPy head | prob max diff, NumPy head on torch features | heatmap max abs diff | mean of per-map max diff | heatmap max diff, NumPy head on torch features | min Spearman | argmax identical |
  |---|---|---|---|---|---|---|---|---|---|---|
  | onnx (1.30.0) | 300 / 1500 | 1.4e-03 (1.8e-05 rel) | 1.1e-05 | 1.1e-05 | 3.9e-07 | 2.3e-04 | 8.4e-06 | 1.4e-05 | 0.99999997 | 0.999 |
  | tflite (tensorflow 2.18.0) | 300 / 1500 | 3.9e-03 (5.4e-05 rel) | 6.8e-05 | 6.8e-05 | 3.9e-07 | 7.2e-04 | 2.9e-05 | 1.3e-05 | 0.99983390 | 0.999 |
  
  ### Sanity checks (Adebayo 2018): Spearman rho / SSIM of randomised vs original map
  
  100 test images, all 5 labels. Median over images; per-label Spearman medians.
  
  | condition | mean abs Δp | Spearman (all) | SSIM (all) | D | G | C | A | H | all-zero maps |
  |---|---|---|---|---|---|---|---|---|---|
  | cascade → classifier | 0.322 | 0.075 | 0.483 | 0.15 | 0.04 | 0.04 | 0.12 | 0.10 | 0 |
  | cascade → conv_head | 0.644 | 0.022 | 0.406 | -0.07 | -0.15 | 0.13 | 0.14 | 0.05 | 0 |
  | cascade → blocks.5 | 0.704 | -0.038 | 0.335 | -0.29 | -0.10 | 0.08 | 0.06 | 0.06 | 0 |
  | cascade → blocks.4 | 0.401 | -0.002 | 0.308 | 0.27 | -0.06 | -0.04 | -0.07 | 0.01 | 0 |
  | cascade → blocks.3 | 0.359 | -0.013 | 0.374 | -0.04 | -0.02 | -0.05 | 0.02 | 0.01 | 0 |
  | cascade → blocks.2 | 0.413 | 0.002 | 0.390 | 0.07 | -0.07 | -0.07 | 0.04 | 0.03 | 0 |
  | cascade → blocks.1 | 0.754 | 0.009 | 0.215 | 0.28 | -0.08 | -0.07 | 0.10 | -0.05 | 0 |
  | cascade → blocks.0 | 0.431 | 0.007 | 0.266 | 0.10 | -0.12 | -0.05 | 0.14 | 0.00 | 0 |
  | cascade → stem | 0.424 | -0.045 | 0.322 | -0.11 | -0.07 | -0.00 | -0.03 | -0.02 | 0 |
  | independent: blocks.5 only | 0.377 | 0.067 | 0.432 | -0.08 | 0.11 | 0.13 | 0.12 | 0.05 | 0 |
  | reference: different image, same label | – | 0.144 | 0.531 | 0.21 | 0.16 | 0.21 | 0.07 | 0.09 | 0 |
  | reference: centre-bias map | – | 0.176 | 0.139 | 0.39 | 0.22 | 0.05 | 0.14 | 0.12 | 0 |
  
  Class specificity (same image, other label's map), Spearman median: D_vs_G -0.006, D_vs_C 0.104, D_vs_A 0.168, D_vs_H 0.124; all label pairs median 0.087 [IQR -0.085, 0.220].
  
  ### Deletion / insertion AUC, blur fill (test positives; deletion lower = better, insertion higher = better)
  
  | label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | D | 285 | 0.69 | 0.331 | 0.470 | 0.294 | -0.139 [-0.156, -0.122] (p=7.0e-36, win 0.85) | +0.037 [0.021, 0.051] (p=6.0e-07, win 0.39) | 0.769 | 0.474 | 0.606 | +0.295 [0.281, 0.310] (p=1.7e-48, win 1.00) | +0.163 [0.151, 0.176] (p=4.4e-48, win 0.98) |
  | G | 62 | 0.72 | 0.238 | 0.522 | 0.270 | -0.284 [-0.329, -0.240] (p=1.9e-11, win 0.94) | -0.031 [-0.077, 0.014] (p=5.1e-01, win 0.48) | 0.843 | 0.532 | 0.625 | +0.312 [0.267, 0.357] (p=7.6e-12, win 1.00) | +0.218 [0.182, 0.258] (p=7.6e-12, win 1.00) |
  | C | 62 | 0.73 | 0.737 | 0.754 | 0.906 | -0.018 [-0.046, 0.009] (p=2.9e-01, win 0.56) | -0.170 [-0.215, -0.123] (p=7.6e-12, win 1.00) | 0.877 | 0.755 | 0.813 | +0.122 [0.086, 0.160] (p=7.3e-10, win 0.95) | +0.064 [0.030, 0.100] (p=3.5e-02, win 0.55) |
  | A | 42 | 0.67 | 0.041 | 0.187 | 0.112 | -0.147 [-0.190, -0.108] (p=4.5e-13, win 1.00) | -0.071 [-0.095, -0.049] (p=4.5e-13, win 1.00) | 0.406 | 0.175 | 0.446 | +0.231 [0.185, 0.282] (p=2.3e-12, win 0.98) | -0.040 [-0.080, 0.002] (p=1.9e-01, win 0.50) |
  | H | 34 | 0.38 | 0.045 | 0.163 | 0.116 | -0.117 [-0.157, -0.080] (p=1.6e-09, win 0.97) | -0.070 [-0.092, -0.048] (p=1.7e-07, win 0.88) | 0.467 | 0.177 | 0.197 | +0.290 [0.228, 0.353] (p=1.2e-10, win 1.00) | +0.270 [0.207, 0.334] (p=1.6e-09, win 0.94) |
  | all | 485 | 0.68 | 0.326 | 0.467 | 0.341 | -0.141 [-0.155, -0.128] (p=5.0e-57, win 0.85) | -0.015 [-0.030, -0.001] (p=2.0e-01, win 0.56) | 0.740 | 0.471 | 0.593 | +0.269 [0.256, 0.284] (p=1.2e-80, win 0.99) | +0.147 [0.134, 0.160] (p=1.1e-63, win 0.88) |
  
  ### Deletion / insertion AUC, mean fill (test positives; deletion lower = better, insertion higher = better)
  
  | label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | D | 285 | 0.69 | 0.196 | 0.373 | 0.204 | -0.178 [-0.192, -0.164] (p=3.2e-46, win 0.96) | -0.009 [-0.022, 0.005] (p=4.0e-01, win 0.50) | 0.618 | 0.376 | 0.452 | +0.242 [0.222, 0.261] (p=1.3e-43, win 0.91) | +0.167 [0.149, 0.184] (p=3.7e-39, win 0.88) |
  | G | 62 | 0.72 | 0.080 | 0.123 | 0.145 | -0.043 [-0.064, -0.024] (p=1.0e-04, win 0.69) | -0.066 [-0.094, -0.039] (p=3.4e-05, win 0.68) | 0.532 | 0.121 | 0.611 | +0.412 [0.367, 0.457] (p=7.6e-12, win 1.00) | -0.079 [-0.130, -0.029] (p=6.5e-03, win 0.37) |
  | C | 62 | 0.73 | 0.198 | 0.120 | 0.768 | +0.078 [0.048, 0.111] (p=2.8e-05, win 0.34) | -0.570 [-0.607, -0.533] (p=7.6e-12, win 1.00) | 0.301 | 0.124 | 0.606 | +0.178 [0.140, 0.219] (p=8.3e-11, win 0.89) | -0.305 [-0.357, -0.253] (p=2.5e-11, win 0.05) |
  | A | 42 | 0.67 | 0.069 | 0.131 | 0.096 | -0.063 [-0.108, -0.023] (p=4.2e-02, win 0.62) | -0.027 [-0.050, -0.006] (p=2.0e-01, win 0.52) | 0.380 | 0.118 | 0.435 | +0.262 [0.219, 0.307] (p=1.5e-11, win 0.93) | -0.055 [-0.106, 0.001] (p=7.2e-02, win 0.43) |
  | H | 34 | 0.38 | 0.011 | 0.024 | 0.041 | -0.012 [-0.020, -0.006] (p=1.9e-06, win 0.88) | -0.029 [-0.041, -0.018] (p=1.2e-09, win 0.97) | 0.141 | 0.023 | 0.098 | +0.118 [0.087, 0.150] (p=5.8e-10, win 0.94) | +0.043 [0.013, 0.072] (p=3.0e-03, win 0.76) |
  | all | 485 | 0.68 | 0.157 | 0.264 | 0.248 | -0.106 [-0.120, -0.092] (p=2.6e-45, win 0.81) | -0.091 [-0.109, -0.072] (p=2.6e-13, win 0.62) | 0.513 | 0.264 | 0.466 | +0.249 [0.233, 0.264] (p=5.4e-75, win 0.92) | +0.047 [0.027, 0.069] (p=5.8e-09, win 0.66) |
  
  ### IDRiD localisation of the DR Grad-CAM (81 images, FOV only)
  
  Model on IDRiD: mean p(DR) 0.948, median 0.998, 97.5% with p >= 0.5. Grad-CAM energy on the optic disc 0.019 (centre map 0.029, OD area 0.026). Spearman(p(DR), EPG any-lesion) = 0.258 (p=2.0e-02).
  
  | lesions | n img | area frac (chance EPG) | area frac +15px (≈chance PG-tol) | method | PG strict | PG ±15 px | EPG | pixel AUROC | AP |
  |---|---|---|---|---|---|---|---|---|---|
  | any (MA∪HE∪EX∪SE) | 81 | 0.0418 | 0.352 | Grad-CAM DR | 0.14 | 0.79 | 0.0704 | 0.657 | 0.0811 |
  |  |  |  |  | random | 0.04 | 0.32 | 0.0417 | 0.499 | 0.0432 |
  |  |  |  |  | centre | 0.09 | 0.58 | 0.0482 | 0.575 | 0.0657 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.10 | 0.30 | 0.0475 | 0.518 | 0.0591 |
  | microaneurysms | 81 | 0.0027 | 0.137 | Grad-CAM DR | 0.01 | 0.46 | 0.0057 | 0.622 | 0.0078 |
  |  |  |  |  | random | 0.00 | 0.13 | 0.0027 | 0.499 | 0.0030 |
  |  |  |  |  | centre | 0.00 | 0.20 | 0.0030 | 0.580 | 0.0038 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.02 | 0.11 | 0.0025 | 0.515 | 0.0036 |
  | haemorrhages | 80 | 0.0174 | 0.132 | Grad-CAM DR | 0.10 | 0.59 | 0.0362 | 0.708 | 0.0629 |
  |  |  |  |  | random | 0.02 | 0.12 | 0.0174 | 0.502 | 0.0191 |
  |  |  |  |  | centre | 0.01 | 0.14 | 0.0162 | 0.477 | 0.0195 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.00 | 0.12 | 0.0140 | 0.505 | 0.0218 |
  | hard exudates | 81 | 0.0189 | 0.184 | Grad-CAM DR | 0.02 | 0.20 | 0.0247 | 0.640 | 0.0303 |
  |  |  |  |  | random | 0.01 | 0.16 | 0.0189 | 0.499 | 0.0201 |
  |  |  |  |  | centre | 0.06 | 0.42 | 0.0268 | 0.711 | 0.0517 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.06 | 0.15 | 0.0273 | 0.535 | 0.0340 |
  | soft exudates | 40 | 0.0062 | 0.035 | Grad-CAM DR | 0.00 | 0.03 | 0.0087 | 0.569 | 0.0113 |
  |  |  |  |  | random | 0.00 | 0.03 | 0.0062 | 0.502 | 0.0083 |
  |  |  |  |  | centre | 0.03 | 0.05 | 0.0050 | 0.397 | 0.0081 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.03 | 0.10 | 0.0080 | 0.498 | 0.0180 |
  
  Paired differences Grad-CAM DR minus baseline (mean [95% bootstrap CI], Wilcoxon p):
  
  - any (MA∪HE∪EX∪SE): EPG vs random +0.029 [0.023, 0.034] p=2.7e-12; AUROC vs random +0.158 [0.134, 0.182] p=1.8e-13; EPG vs centre +0.022 [0.015, 0.029] p=8.2e-09; AUROC vs centre +0.082 [0.046, 0.118] p=3.5e-05; EPG vs gradcam_G +0.023 [0.003, 0.039] p=7.7e-08; AUROC vs gradcam_G +0.139 [0.107, 0.169] p=2.0e-10
  - microaneurysms: EPG vs random +0.003 [0.002, 0.004] p=5.6e-14; AUROC vs random +0.122 [0.102, 0.142] p=1.7e-13; EPG vs centre +0.003 [0.002, 0.003] p=1.5e-12; AUROC vs centre +0.042 [0.014, 0.068] p=2.3e-03; EPG vs gradcam_G +0.003 [0.002, 0.004] p=2.6e-12; AUROC vs gradcam_G +0.107 [0.084, 0.132] p=8.3e-11
  - haemorrhages: EPG vs random +0.019 [0.015, 0.023] p=5.6e-13; AUROC vs random +0.206 [0.175, 0.236] p=1.3e-13; EPG vs centre +0.020 [0.016, 0.024] p=1.1e-13; AUROC vs centre +0.231 [0.192, 0.272] p=7.5e-13; EPG vs gradcam_G +0.022 [0.017, 0.028] p=4.6e-13; AUROC vs gradcam_G +0.203 [0.163, 0.245] p=4.6e-12
  - hard exudates: EPG vs random +0.006 [0.002, 0.009] p=9.0e-05; AUROC vs random +0.141 [0.108, 0.172] p=3.6e-10; EPG vs centre -0.002 [-0.008, 0.003] p=4.1e-01; AUROC vs centre -0.071 [-0.117, -0.027] p=2.9e-03; EPG vs gradcam_G -0.003 [-0.024, 0.011] p=1.2e-03; AUROC vs gradcam_G +0.104 [0.057, 0.151] p=7.4e-05
  - soft exudates: EPG vs random +0.003 [0.000, 0.005] p=4.2e-01; AUROC vs random +0.067 [-0.010, 0.139] p=9.7e-02; EPG vs centre +0.004 [0.001, 0.006] p=1.8e-02; AUROC vs centre +0.173 [0.078, 0.269] p=9.4e-04; EPG vs gradcam_G +0.001 [-0.005, 0.005] p=1.1e-01; AUROC vs gradcam_G +0.071 [-0.037, 0.181] p=1.7e-01
  
  ### On-device cost: n100, OPENBLAS_NUM_THREADS=1 (primary)
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | ort (onnxruntime 1.30.0) | plain | 7.5 / 7.5 / 7.5 | – | – | 108.4 | 9.1e-06 | – |
  | ort (onnxruntime 1.30.0) | head | 8.1 / 8.1 / 8.2 | 0.8 | 0.7 | 107.9 | 9.2e-06 | – |
  | ort (onnxruntime 1.30.0) | gradcam1 | 9.2 / 9.2 / 9.2 | 1.8 | 1.7 | 117.5 | 9.2e-06 | 2.4e-04 |
  | ort (onnxruntime 1.30.0) | gradcam5 | 12.4 / 12.3 / 12.4 | 4.8 | 4.9 | 127.8 | 9.2e-06 | 2.5e-04 |
  | ort_nospin (onnxruntime 1.30.0) | plain | 7.9 / 7.9 / 7.9 | – | – | 106.6 | 9.1e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | head | 9.3 / 9.3 / 9.3 | 0.9 | 1.4 | 109.6 | 9.2e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | gradcam1 | 9.7 / 9.7 / 9.7 | 1.8 | 1.8 | 115.5 | 9.2e-06 | 2.4e-04 |
  | ort_nospin (onnxruntime 1.30.0) | gradcam5 | 12.9 / 12.8 / 12.9 | 4.8 | 5.0 | 130.6 | 9.2e-06 | 2.5e-04 |
  | ov (openvino 2026.4.1) | plain | 8.2 / 8.2 / 8.2 | – | – | 147.9 | 2.1e-06 | – |
  | ov (openvino 2026.4.1) | head | 9.0 / 9.0 / 9.0 | 0.8 | 0.8 | 151.4 | 1.9e-06 | – |
  | ov (openvino 2026.4.1) | gradcam1 | 10.1 / 10.1 / 10.2 | 1.8 | 1.9 | 160.5 | 1.9e-06 | 2.4e-04 |
  | ov (openvino 2026.4.1) | gradcam5 | 13.0 / 12.9 / 13.0 | 4.4 | 4.8 | 174.6 | 1.9e-06 | 2.5e-04 |
  
  ### On-device cost: n100, OpenBLAS default threads (first pass)
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | ort (onnxruntime 1.30.0) | plain | 7.4 / 7.4 / 7.5 | – | – | 107.2 | 9.1e-06 | – |
  | ort (onnxruntime 1.30.0) | head | 32.6 / 32.0 / 48.0 | 5.8 | 25.2 | 116.7 | 9.2e-06 | – |
  | ort (onnxruntime 1.30.0) | gradcam1 | 48.5 / 48.0 / 72.0 | 23.9 | 41.1 | 125.6 | 9.2e-06 | 2.4e-04 |
  | ort (onnxruntime 1.30.0) | gradcam5 | 86.3 / 96.0 / 112.0 | 57.5 | 78.9 | 139.0 | 9.2e-06 | 2.5e-04 |
  | ort_nospin (onnxruntime 1.30.0) | plain | 8.0 / 7.9 / 7.9 | – | – | 108.2 | 9.1e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | head | 13.2 / 13.2 / 14.7 | 0.7 | 5.3 | 115.1 | 9.2e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | gradcam1 | 15.6 / 15.7 / 16.8 | 1.5 | 7.6 | 123.4 | 9.2e-06 | 2.4e-04 |
  | ort_nospin (onnxruntime 1.30.0) | gradcam5 | 17.4 / 17.5 / 19.1 | 4.0 | 9.4 | 137.6 | 9.2e-06 | 2.5e-04 |
  | ov (openvino 2026.4.1) | plain | 8.3 / 8.2 / 8.3 | – | – | 147.8 | 2.1e-06 | – |
  | ov (openvino 2026.4.1) | head | 22.7 / 21.5 / 34.1 | 1.0 | 14.4 | 157.5 | 1.9e-06 | – |
  | ov (openvino 2026.4.1) | gradcam1 | 24.1 / 21.9 / 38.1 | 2.0 | 15.8 | 166.8 | 1.9e-06 | 2.4e-04 |
  | ov (openvino 2026.4.1) | gradcam5 | 26.6 / 25.1 / 38.8 | 4.2 | 18.3 | 180.7 | 1.9e-06 | 2.5e-04 |
  
  ### On-device cost: pi, OPENBLAS_NUM_THREADS=1
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | tflite (tflite-runtime 2.13.0) | plain | 547.0 / 546.9 / 569.6 | – | – | 63.3 | 1.2e-04 | – |
  | tflite (tflite-runtime 2.13.0) | head | 553.5 / 553.6 / 573.4 | 13.5 | 6.5 | 63.1 | 1.2e-04 | – |
  | tflite (tflite-runtime 2.13.0) | gradcam1 | 597.4 / 599.6 / 621.9 | 61.5 | 50.4 | 70.3 | 1.2e-04 | 2.5e-04 |
  | tflite (tflite-runtime 2.13.0) | gradcam5 | 693.0 / 693.8 / 720.6 | 155.8 | 146.0 | 81.2 | 1.2e-04 | 3.9e-04 |

- **Device conditions**:
  - **N100**: idle before every config (load avg ≤ 0.26, 3.2 GHz). The suite was
    run twice. Pass 1 used numpy's default OpenBLAS threading
    (`results/n100_run010/openblas_default/`). Pass 2, the primary one, used
    `OPENBLAS_NUM_THREADS=1` (`results/n100_run010/`). A first attempt failed
    before timing because `/usr/bin/time` is not installed on the N100 (rc 127,
    `out_attempt1_no_usr_bin_time/`); the driver now skips it there.
  - **Pi**: `get_throttled` was 0x50005 or 0x50000 and the ARM clock was
    600 MHz at every before/after sample (14:03-14:09 CEST), 48-55 °C. As in
    runs 006/009, all Pi latencies are under the ~600 MHz under-voltage cap,
    roughly 2x pessimistic.
  - **Pi first attempt** (`results/pi_run010/out_attempt1_openblas_default/`),
    default OpenBLAS threads: `plain` and `head` completed. `gradcam1` died with
    SIGSEGV after 74 s. `gradcam5` was still running after 16 min at ~390 % CPU
    (about 70 s expected) and was killed. 10-iteration probes with 1 and 4
    OpenBLAS threads both completed and gave the same maps, so the failure
    appears only in long runs with 4 OpenBLAS threads competing with the 4
    TFLite/XNNPACK threads on 4 cores. The cause was not pinned down. The final
    Pi suite used `OPENBLAS_NUM_THREADS=1`, and all 4 configs completed.

- **Artifacts**:
  - Scripts: `results/run010_common.py`, `run010_check_reference.py`,
    `run010_export_edge.py`, `run010_export_tflite.py`, `run010_parity_edge.py`,
    `run010_sanity.py`, `run010_train_randlabels.py` (not run),
    `run010_faithfulness.py`, `run010_idrid_prep.py`, `run010_localisation.py`,
    `run010_make_device_fixture.py`, `run010_figures.py`, `run010_tables.py`
    (generates every table above from the JSONs), `run010_run_final.sh`. Edge
    code: `edge/gradcam_numpy.py`, `edge/gradcam_bench.py`, `edge/run010_device.sh`.
  - Results (`results/run010/`, `*_final*`; development pass `*_dev*`):
    `reference_vs_pytorch_grad_cam_final.json`, `parity_{onnx,tflite}_final.json`,
    `sanity_final.json`, `faithfulness_final.json` (+ `_per_image.json`,
    `_curves.npz`), `localisation_final.json` (+ `_per_image.csv`),
    `idrid_manifest.csv`, `figure_selection_final.json`,
    `figure_idrid_selection_final.json`, `probs_final_{val,test}.npy`,
    `tables_final.md`, `log_*_final.txt`, `log_final_driver.txt`,
    `env_venv_run010_freeze.txt`.
  - Device results: `results/n100_run010/` and `results/pi_run010/`
    (`results_<runtime>_<mode>.json`, `log_*.txt`, `suite.log`, failed
    attempts in sub-directories). On the devices: `~/eye-edge/run010/`.
  - Edge artifacts: `edge/artifacts/run010_final_*`,
    `edge/artifacts/run010_bundle_final/`.
  - Figures (PDF + PNG preview): `paper/figs/run010_gradcam_examples`,
    `run010_idrid_overlays`, `run010_deletion_insertion`,
    `run010_sanity_cascade`. Colour map viridis (perceptually uniform, CVD-safe),
    each with a colourbar. Colour = positive Grad-CAM evidence for the named
    label scaled to that image's maximum: 0 (dark purple) = none, 1 (yellow) =
    strongest. It is not a probability. Example selection (seed 2026, uniform
    within each category; candidates: TP D 253, G 55, C 50, A 34, H 34; DR FP 360;
    other-disease FP 1451; mild-NPDR misses 22): `4354_left` (DR TP, moderate
    NPDR), `1294_right` (G), `726_right` (C), `608_left` (A), `4214_right` (H),
    `3431_right` (DR FP), `3110_right` (glaucoma FP), `4542_right` (mild NPDR
    missed). IDRiD overlays (seed 2027): IDRiD_01, 12, 32, 50.
  - IDRiD data: `data/idrid/` (zip, sha256, Kaggle metadata, `prep512/`).

- **Verdict (final model)**:
  - **The edge Grad-CAM is the same Grad-CAM, and no autodiff is needed on the
    device.**
    - The reference matches pytorch-grad-cam to 1.5e-6.
    - Mac, 300 test images x 5 labels: the ONNX backbone + NumPy head gives
      heatmaps within 2.3e-4 of PyTorch autograd (max abs diff, maps in [0,1])
      and probabilities within 1.1e-5. TFLite: 7.2e-4 (99th percentile 1.6e-4)
      and 6.8e-5. The hottest pixel is identical in 1499 of 1500 maps for both.
    - The NumPy maths alone (on torch features) adds ≤ 1.4e-5. The remaining
      error is the backbone runtime's.
    - On the devices, against the stored Mac reference: N100 maps ≤ 2.5e-4,
      probabilities ≤ 9.2e-6 (ORT) / 1.9e-6 (OpenVINO); Pi maps ≤ 3.9e-4,
      probabilities ≤ 1.2e-4. These diffs include the float16 storage of the
      reference maps.
  - **Passes the model-randomisation sanity check (Adebayo).**
    - Re-initialising only the classifier drops the median Spearman ρ between
      the randomised and original map to 0.075. That is already below the 0.144
      between the same label's maps of two *different* images.
    - Every deeper cascade stage stays between −0.045 and 0.022. Randomising only
      the Grad-CAM layer (blocks.5) gives 0.067.
    - SSIM stays at 0.22-0.48, below the different-image level of 0.53. On these
      smooth, mostly empty maps SSIM is dominated by background, so ρ is the
      informative metric.
    - The maps are class-specific: the median ρ between two labels' maps of the
      same image is 0.087.
    - The data-randomisation test was **not run** (compute).
  - **Faithful: clearly better than random, mostly better than a centre prior.**
    Blur fill, all 485 positive (image, label) pairs:
    - Insertion AUC: 0.740 vs 0.471 (random) and 0.593 (centre). Δ vs centre
      +0.147 [0.134, 0.160], Grad-CAM wins on 88 % of pairs.
    - Deletion AUC: 0.326 vs 0.467 (random) and 0.341 (centre). Δ vs centre
      −0.015 [−0.030, −0.001], paired Wilcoxon p = 0.20, so deletion does not
      separate Grad-CAM from the centre prior.
    - Per label:
      - **DR deletion is worse than centre** (0.331 vs 0.294, Δ +0.037 [0.021, 0.051]).
      - For cataract, deletion does not separate Grad-CAM from random (Δ −0.018,
        p = 0.29). This fits cataract being a global media opacity.
      - AMD insertion ties with centre (Δ −0.040 [−0.080, 0.002]).
    - Mean-colour fill is confounded for cataract: a grey disc looks like an
      opaque lens, so deleting the centre keeps p(cataract) high (centre
      deletion AUC 0.768). Blur is the primary fill.
    - The development checkpoint (trained on these test images) had *not* beaten
      the centre prior on insertion (Δ −0.009 [−0.024, 0.006]). Faithfulness
      depends on the model as well as the method.
  - **DR lesion localisation: above every baseline overall, weak in absolute
    terms, worst for microaneurysms.** 81 IDRiD images, external data; the model
    gives 97.5 % of them p(DR) ≥ 0.5.
    - Any-lesion pixel AUROC is 0.657, against 0.499 (random), 0.575 (centre)
      and 0.518 (wrong-class glaucoma map).
    - EPG is 0.070, 1.7x the lesion area fraction of 0.042.
    - The map's maximum hits a lesion in 11/81 images (0.14), against 0.04
      (random), 0.09 (centre) and 0.10 (glaucoma map).
    - The ±15 px pointing game (0.79) must be read against its chance level: the
      dilated lesions cover 35 % of the FOV.
    - Haemorrhages localise best: AUROC 0.708, pointing game 8/81 (0.10).
    - Hard exudates: AUROC 0.640, *below* the centre map (0.711, Δ −0.071
      [−0.117, −0.027]). Exudates cluster around the macula near the image
      centre.
    - Soft exudates (n = 40): AUROC 0.569, not different from random (Δ +0.067
      [−0.010, 0.139]).
    - **Microaneurysms**: the pointing game hits 1/81, EPG 0.0057 (2.1x the
      0.0027 chance level), AUROC 0.622. Grad-CAM at the 16x16 feature resolution
      (32 px cells at 512 px) cannot point at lesions a few pixels wide, as
      Saporta 2022 found for small pathologies.
    - Localisation correlates weakly with confidence (Spearman ρ between p(DR)
      and EPG = 0.258, p = 0.020).
    - The DR map puts 1.9 % of its mass on the optic disc, against a disc area of
      2.6 %, so there is no optic-disc shortcut on IDRiD. On ODIR, several
      sampled maps peak at the optic disc (DR FP, glaucoma FP, mild-NPDR miss)
      or at the fovea (DR TP).
  - **On-device cost is small on the N100 and moderate on the Pi**, with
    `OPENBLAS_NUM_THREADS=1`:
    - **N100, ORT** (plain 7.5 ms, 108 MB): Grad-CAM for DR adds +1.7 ms
      (9.2 ms) and all 5 labels add +4.9 ms (12.4 ms). Peak RSS +9 / +19 MB.
      The probability-only backbone + NumPy head path costs +0.7 ms.
    - **N100, OpenVINO**: +1.9 / +4.8 ms on 8.2 ms.
    - **Pi 3B, TFLite, 4 threads, 600 MHz cap** (plain 547 ms, 63 MB): +50 ms for
      DR (597 ms, +9 %) and +146 ms for 5 labels (693 ms, +27 %). The NumPy part
      itself is 62 / 156 ms. Peak RSS +7 / +18 MB (70 / 81 MB, well inside the
      870 MB board).
    - **Deployment caveat**: with numpy's default multi-threaded OpenBLAS next to
      the inference runtime, the N100 overhead grew to +79 ms (ORT, 5 labels;
      +9.4 ms with ORT thread spinning off; +18 ms OpenVINO). On the Pi it gave a
      crash and a stall. The NumPy Grad-CAM must be single-threaded
      (`OPENBLAS_NUM_THREADS=1`) when it runs next to an inference runtime.
  - **Paper framing**: Grad-CAM can be offered on both devices at little extra
    cost and is verified to be the reference Grad-CAM. It passes the randomisation
    check, and its relevance ranking is faithful to the model, beating random and,
    on insertion, a centre prior. It is **not** a lesion detector. It overlaps
    haemorrhages above chance, does no better than a centre prior on exudates,
    and does not show microaneurysms, the lesions that define mild NPDR, the
    model's weakest grade. With Sayres 2019 (heatmaps hurt readers on no-DR
    cases) and Ghassemi 2021, the map should be presented as a coarse "where the
    model looked" aid next to a positive prediction, not as evidence that the
    prediction is right.

- **Limitations**:
  - No data-randomisation test. One randomisation seed. One random-map draw per
    image in faithfulness (50 per image in localisation).
  - IDRiD is one Indian clinic and one camera, with the disc truncated at
    top/bottom (black bands after padding). Every image has DR, so the DR map is
    never evaluated on negatives. Masks are binarised with an any-coverage rule
    at 512 px, which slightly enlarges small lesions.
  - Pixel-wise deletion/insertion creates out-of-distribution images (the ROAR
    argument, Hooker 2019). Two fills are reported, with no retraining. Whole
    images are ranked, including the black corners.
  - The figure's operating thresholds (90 % validation sensitivity) are very low
    for H (0.00054) and A (0.039), because pos_weight training compresses those
    probabilities. So the "correct positive" H/A examples are correct only at
    those thresholds.
  - Grad-CAM only (no Grad-CAM++, integrated gradients or Score-CAM), and only
    the last-conv target layer (the only one available on the device without a
    second backbone output).
  - Pi latencies are under the under-voltage cap. Single run per config. The Pi
    clock was sampled before/after each config only.

---

## Run 011 — Confidence calibration: temperature vs Platt scaling, on device, under INT8

- **Date**: 2026-10-03
- **Purpose**: check whether the run 008 models' probabilities can be
  shown to a health worker as confidence, whether a post-hoc fix works,
  whether the fix survives export to the N100/Pi, and whether INT8 breaks it.
- **Method** (`results/run011_calibration.py`):
  - Inputs: run 008 per-image probabilities on validation and test
    (`results/run008_probs/`), all 9 models.
  - Per label, fitted on VALIDATION negative log-likelihood, applied
    unchanged to TEST: (i) temperature scaling σ(z/T) (Guo et al. 2017);
    (ii) Platt scaling σ(a·z + b). Platt was added after temperature
    scaling failed (see verdict): the loss's positive-class weight w shifts
    logits by roughly log w, which needs a bias term.
  - Test metrics: ECE (15 equal-width bins), adaptive ECE (15 equal-mass
    bins), Brier, NLL, mean predicted probability vs prevalence. AUC is
    unchanged by both maps (monotonic).
  - Final model (A s2): patient-clustered bootstrap (1000) of ECE and of the
    ECE change. The bootstrap distribution of ECE is biased upward
    (resampling duplicates add binning noise), so for DR the Platt point
    estimate falls below its own percentile interval. Report the CI of the
    change, which is unaffected.
  - Device check: validation-fitted Platt maps applied to the final model's
    test probabilities from the N100 (ONNX Runtime CPU, OpenVINO CPU/iGPU)
    and the Pi (TFLite FP32, 4 threads) from run 009.
  - INT8 (development, `results/run011_int8_transfer.py`): 224 px run002
    model, Pi run 006 TFLite probabilities on the 1270-image validation set.
    A Platt map fitted to FP32 outputs is applied unchanged to the
    dynamic-range and full-integer INT8 outputs of the same images.
- **Results — final model (A s2), test set**:

  | label | prevalence | mean p raw → Platt | T | Platt (a, b) | ECE raw | ECE temp. | ECE Platt | ΔECE Platt−raw, 95 % CI | Brier raw → Platt | NLL raw → Platt |
  |---|---|---|---|---|---|---|---|---|---|---|
  | DR (per-eye) | 0.278 | 0.429 → 0.283 | 1.32 | (0.68, −1.02) | 0.163 | 0.163 | 0.027 | [−0.143, −0.094] | 0.184 → 0.144 | 0.558 → 0.447 |
  | Glaucoma | 0.060 | 0.153 → 0.069 | 1.58 | (0.51, −1.75) | 0.097 | 0.114 | 0.021 | [−0.086, −0.054] | 0.078 → 0.038 | 0.277 → 0.148 |
  | Cataract | 0.060 | 0.110 → 0.055 | 1.12 | (0.68, −2.02) | 0.055 | 0.056 | 0.016 | [−0.049, −0.024] | 0.044 → 0.027 | 0.181 → 0.121 |
  | AMD | 0.041 | 0.099 → 0.048 | 1.46 | (0.45, −2.01) | 0.058 | 0.084 | 0.012 | [−0.055, −0.028] | 0.049 → 0.024 | 0.197 → 0.106 |
  | Hypertensive ret. | 0.033 | 0.075 → 0.031 | 1.40 | (0.33, −2.21) | 0.049 | 0.063 | 0.009 | [−0.054, −0.018] | 0.045 → 0.029 | 0.174 → 0.120 |

  Adaptive ECE raw → Platt: DR 0.162 → 0.021, G 0.104 → 0.020, C 0.062 →
  0.012, A 0.067 → 0.014, H 0.059 → 0.016.

  Across seeds (test ECE, mean ± s.d. of 3 seeds), raw → Platt:
  A: DR 0.148 ± 0.047 → 0.034 ± 0.006; G 0.161 → 0.020; C 0.129 → 0.017;
  A 0.174 → 0.017; H 0.129 → 0.011. B (224 px): DR 0.114 → 0.032; G 0.135
  → 0.018; C 0.072 → 0.013; A 0.093 → 0.022; H 0.099 → 0.009. C (512
  patient): DR 0.116 → 0.032; G 0.111 → 0.021; C 0.116 → 0.017; A 0.079 →
  0.014; H 0.075 → 0.009. Temperature scaling left ECE unchanged or worse
  for every group and label.

  Device check (Platt-calibrated test ECE; max |Δ calibrated p| vs reference):
  N100 ONNX Runtime CPU and OpenVINO CPU and Pi TFLite FP32 give identical
  ECE to four decimals (DR 0.0266, G 0.0206, C 0.0164, A 0.0125, H 0.0095);
  max difference 2.5e-5 (N100) and 1.4e-4 (Pi). N100 iGPU: DR 0.0281, others
  within 0.0012; max difference 0.032.

  INT8 (development, validation, Platt map fitted on FP32, ECE):

  | variant | DR | G | C | A | H |
  |---|---|---|---|---|---|
  | FP32 (in-sample) | 0.025 | 0.025 | 0.014 | 0.009 | 0.012 |
  | dynamic-range INT8 | 0.029 | 0.015 | 0.015 | 0.015 | 0.006 |
  | full-integer INT8 | 0.080 | 0.045 | 0.051 | 0.018 | 0.009 |

- **Artifacts**: `results/run011/calibration.json` (all models, all
  metrics, temperatures, Platt parameters, bootstrap, device check, raw INT8
  ECE), `results/run011/int8_platt_transfer_dev.json`, scripts
  `results/run011_calibration.py`, `results/run011_int8_transfer.py`, log
  `results/logs/run011_calibration.log`, figure
  `paper/figs/run011_reliability.{pdf,png}` (final model, test,
  reliability curves raw / temperature / Platt per label).
- **Verdict**:
  - Raw outputs are badly over-confident towards disease: mean predicted
    probability is 1.5-2.6× the prevalence, as expected from the
    positive-class weights in the loss.
  - **Temperature scaling does not fix this** and makes it worse for
    glaucoma, AMD and hypertensive retinopathy: one temperature cannot
    remove a constant logit offset, and the NLL-optimal T > 1 pushes
    probabilities further towards 0.5.
  - **Platt scaling fixes it**: test ECE falls by 0.04-0.14 per label, with
    CIs excluding zero for every label, and mean predicted probability
    lands within 0.002-0.009 of prevalence. It costs two operations per
    label on the device.
  - The calibrated outputs survive deployment exactly on every CPU runtime
    (identical ECE to four decimals on N100 and Pi).
  - Full-integer INT8 breaks the calibration map: DR ECE triples (0.025 →
    0.080) and cataract nearly quadruples. Dynamic-range INT8 roughly
    preserves it. This reinforces the FP32 recommendation.
- **Limitations**: one dataset; calibration fitted on the 1270-image
  validation set, which also selected checkpoints; INT8 analysis is on the
  development model and in-sample for FP32.

---

## Run 012 — N100 iGPU: driver upgrade, INT8 collapse diagnosed and fixed

- **Date**: 2026-10-03
- **Purpose**: run 004 found OpenVINO INT8 on the N100 iGPU collapsing to
  mean AUC 0.6226 (CPU INT8 0.805) and attributed it, without evidence, to
  the old Debian OpenCL driver (intel-opencl-icd 22.43). The paper footnote
  repeated that. Test the driver hypothesis, re-measure iGPU latency with a
  current driver, find the real cause, and fix it.
- **Driver setup (no sudo, nothing system-wide changed)**: Intel
  compute-runtime 26.35.39758.10 (2026-09-17) with IGC 2.41.5 and gmmlib
  22.10.0, official .debs downloaded from GitHub, all five sha256 sums
  matched Intel's published sums. Unpacked with `dpkg-deb -x` into
  `~/neo/root` on the N100. A private ICD file (`~/neo/vendors/intel-neo.icd`)
  plus `OCL_ICD_VENDORS` and `LD_LIBRARY_PATH` (`~/neo/env.sh`) make the
  ocl-icd loader use it. Each result JSON records the libigdrcl.so path
  actually mapped into the process, so old and new runs are verified to use
  different drivers. With the new driver OpenVINO reports the device as
  "Intel(R) UHD Graphics" instead of the raw id "Intel(R) Graphics [0x46d1]".
  Old Debian stack: intel-opencl-icd 22.43.24595.41, libigc1
  1.0.12504.6 (bookworm), libigdgmm12 22.7.2.
- **Env**: as run 004/009 (OpenVINO 2026.4.1, NNCF 3.4.0, ONNX Runtime
  1.30.0, Python 3.13.5). `bench_n100.py` gained an optional
  `OV_GPU_PRECISION` override (`INFERENCE_PRECISION_HINT`) and logs the GPU
  name, inference precision and driver path. Defaults unchanged.
- **Method**:
  1. Same protocol as run 004 (224 px run002 model, 1270-image validation
     set, batch 1, 50 warm-up + 500 timed) and run 009 (512 px final model,
     1025-image test set) for the iGPU configs under old and new driver, each
     at the plugin default (f16) and forced f32 inference precision.
  2. Re-quantize with NNCF variants (MIXED/PERFORMANCE preset × target
     device ANY/GPU), same 300 calibration images as run 004, each on CPU and
     GPU (`results/run012_int8_variants.py`).
  3. Layer-wise localisation (`results/run012_layer_diff.py`): every
     Convolution/GroupConvolution/Add/Multiply/activation/ReduceMean/MatMul
     output exposed as a model output; CPU vs GPU at f32 on 3 images,
     relative error per node. FP32 ONNX as control.
  4. NNCF workarounds targeted at the localised layer type.
- **Results**:

  Driver and precision (224 px on validation, 512 px on test; latency
  model-only mean ms):

  | config | old driver 22.43: mean AUC / latency | new driver 26.35: mean AUC / latency |
  |---|---|---|
  | 224 FP32 iGPU, default f16 | 0.8215 / 3.02 | 0.8215 / 2.75 |
  | 224 FP32 iGPU, f32 | 0.8214 / 3.62 | 0.8214 / 3.24 |
  | 224 INT8 iGPU, default | 0.6226 / 3.25 | 0.6226 / 3.06 |
  | 224 INT8 iGPU, f32 | 0.6124 / 3.48 | 0.6124 / 3.08 |
  | 512 FP32 iGPU, default f16 (test) | 0.8546 / 6.94 | 0.8546 / 7.04 |
  | 512 FP32 iGPU, f32 (test) | 0.8545 / 9.70 | 0.8545 / 9.17 |

  At f32 the iGPU matches the PyTorch reference (max |Δp| 7.7e-6 at 224,
  1.2e-5 at 512); the 0.06 probability shift seen in runs 004/009 is the
  plugin's default f16 precision. CPU INT8 re-check: 0.8050, 1.94 ms
  (= run 004).

  NNCF variants, new driver (mean val AUC / latency ms):

  | variant | CPU | iGPU |
  |---|---|---|
  | MIXED, target ANY (= run 004) | 0.8050 / 1.94 | 0.6226 / 2.99 |
  | MIXED, target GPU | 0.8050 / 1.93 | 0.6226 / 3.01 |
  | PERFORMANCE, target ANY | 0.7831 / 1.92 | 0.6078 / 2.93 |
  | PERFORMANCE, target GPU | 0.7831 / 1.93 | 0.6078 / 2.88 |
  | per-tensor activations | 0.5969 / 1.91 | 0.5238 / 2.88 |
  | first depthwise conv kept in float | 0.8038 / 1.92 | 0.7382 / 2.99 |
  | **all depthwise convs kept in float** | **0.8043 / 1.86** | **0.8037 / 2.83** |

  Localisation: in the INT8 IR, CPU and GPU agree to <1e-7 through the
  stem and its hard-swish. The first divergence is the first depthwise
  convolution (`blocks.0.0.conv_dw`, 3×3, stride 2, 16 channels). On the GPU
  its output contains NaN in exactly 2 of 16 channels (channels 14 and 15,
  all 56×56 positions, at f16 and f32 alike), on every image tested; the CPU
  output is finite. The NaNs propagate to the classifier (max relative error
  2.05 downstream). The NaN channels are not the ones with degenerate
  quantization ranges (channels 3, 4, 12 have [0, ≤0.29]), which points to
  the GPU plugin's INT8 depthwise kernel rather than the quantization
  parameters. FP32 control: no node above 5 % except the fused classifier
  MatMul, an artefact of exposing an output inside a fused op.
- **Minimal reproduction (added 2026-10-03, `results/run012_min_repro.py`,
  `results/n100_run012/min_repro.json`)**: the failing layer rebuilt in
  isolation with the original constants (activation FakeQuantize ->
  GroupConvolution 3x3 with weight FakeQuantize), CPU vs GPU at f32, new driver.

  | variant (one change at a time) | GPU NaN channels | rel. error on finite values |
  |---|---|---|
  | as produced by NNCF (weight ranges shape [16,1,1,1,3]) | 14, 15 | 4.7e-2 |
  | weight ranges collapsed to per-channel [16,1,1,1,1] | none | 0 |
  | weight ranges per-tensor | none | 0 |
  | weight FakeQuantize removed | none | 0 |
  | activation FakeQuantize removed | 14, 15 | 4.7e-2 |
  | stride 1 instead of 2 | 14, 15 | 4.7e-2 |
  | layer tiled to 32 channels | 30, 31 | 4.4e-2 |

  **The trigger is the weight FakeQuantize range shape.** NNCF quantized the
  depthwise weights ([16,1,1,3,3] after a Reshape) with ranges of shape
  [16,1,1,1,3]: one range per output channel *and* per kernel column, instead
  of the usual per-output-channel [16,1,1,1,1]. This is valid under
  FakeQuantize's numpy broadcasting, and the CPU plugin computes it
  correctly. The GPU plugin computes it wrongly: every channel is off (about
  5 % relative error) and the last two output channels are NaN, whatever the
  channel count. That pattern is consistent with the GPU kernel indexing the
  48-value range tensor as if it held one value per channel, reading wrong
  values everywhere and past the end for the last channels. This last part
  is an inference from the symptoms, not verified in the OpenVINO source.
  The unusual per-column range shape is itself likely an NNCF quirk with
  5-D group-convolution weights.
- **Plugin fix (added 2026-10-04)**:
  - **Root cause in source**: `src/plugins/intel_gpu/src/graph/graph_optimizer/prepare_quantization.cpp`,
    `prepare_scale_shift_opt`. Its helper `get_offset_safe` addressed the
    FakeQuantize range tensors as `b*p[0] + f*p[1] + y*p[2] + x*p[3]`. For a
    5D (bfzyx) range layout `p[2]` and `p[3]` are the z and y pitches, so x was
    multiplied by the y pitch. The loop also never visited z or w. For ranges
    `[16,1,1,1,3]` the read offset was `3b+3x` instead of `3b+x`, so channels
    14-15 read past the 48-element buffer: NaN, exactly as observed. The code
    is identical in the 2026.4.1 tag and current master.
  - **Patch** (`plugin_fix/ov_quantize_5d_fix.patch`, +113/−30):
    `get_offset_safe` wraps each coordinate to the range tensor's size and
    calls `layout::get_linear_offset`; the loop covers b, f, w, z, y, x. New
    unit test `quantize_gpu.quantize_levels_255_5d_ranges_vary_along_x`.
  - **Build**: OpenVINO tag 2026.4.1 (commit e213a147257, same as the
    installed wheel) plus the patch, built on the N100 (GPU plugin and
    `ov_gpu_unit_tests`; CPU/NPU plugins, oneDNN, Python and frontends off).
    The patched `libopenvino_intel_gpu_plugin.so` (sha256 b95e2c53…) was
    placed in a private copy of the installed openvino package (`~/ovtest`,
    used via PYTHONPATH). The real environment is unchanged.
  - **Validation**:
    - Standalone reproducer: NaN channels [14, 15] → none, error vs CPU 1.0e-7.
    - FakeQuantize-only tests: all channels wrong → all correct.
    - Full INT8 model (224 px, val): mean AUC 0.6226 → 0.8012 (CPU INT8 0.8050).
    - FP32 on the iGPU unchanged (0.8215).
    - Unit tests: all 109 `quantize_gpu*` tests pass, including the new one.
      Broader filter (quantize/fusings/prepare_quantization; 3,181 tests run
      before the run was stopped as too slow, plus the remaining 364 enabled
      `*quant*` tests): 12 failures.
    - All 12 failures also fail with the unpatched build, so they are
      pre-existing: a dynamic-shape `get_tensor()` exception in several
      dynamic quantized convolution tests, plus one fused conv and one
      pooling test. Likely due to this GPU or the reduced build without oneDNN.
    - The new test fails without the fix (value error at x=2) and passes
      with it. After restoring the patch, the rebuilt plugin is byte-identical
      to the validated one.
  - **Re-measured iGPU rows** (patched plugin, compute-runtime 26.35, idle
    N100, same protocol as runs 004/009; `n100_run012/remeasure/`):

    | config | mean AUC | latency mean / p95 (ms) | preprocess (ms) | peak RSS (MB) |
    |---|---|---|---|---|
    | 224 FP32 iGPU (val) | 0.8215 | 2.74 / 2.78 | 4.34 | 687 |
    | 224 INT8 iGPU (val) | 0.8012 | 3.01 / 3.04 | 4.38 | 711 |
    | 512 FP32 iGPU (test) | 0.8546 | 7.02 / 7.07 | 6.80 | 672 |

    The paper's Tables 4 (latency) and 6 (224 edge benchmark) now use these
    iGPU numbers. The energy table (run 009) was measured with the stock
    plugin and the Debian driver and is labelled as such.
- **Energy re-measured (added 2026-10-04, `n100_run012/energy/`)**: run 009's
  exact protocol (`energy_n100.py`, idle baseline, 3 × 60 s, model-only and
  end-to-end), with compute-runtime 26.35 and the patched plugin package
  (PYTHONPATH=~/ovtest). The GPU runs' runtime strings record the neo driver
  path. RAPL readable throughout. Idle package 2.66 ± 0.03 W (224 pass),
  2.60 ± 0.11 W (512 pass).

  | input | runtime | mode | W | inf/s | mJ/inf | mJ above idle | run 009 mJ/inf (old driver) |
  |---|---|---|---|---|---|---|---|
  | 224 | ORT CPU | model | 11.33 ± 0.17 | 426.5 ± 14.4 | 26.6 ± 0.5 | 20.3 ± 0.3 | 26.4 |
  | 224 | OV CPU | model | 11.31 ± 0.03 | 343.6 ± 1.2 | 32.9 ± 0.0 | 25.2 ± 0.0 | 32.7 |
  | 224 | OV iGPU | model | 8.54 ± 0.04 | 370.2 ± 0.4 | 23.1 ± 0.1 | 15.9 ± 0.1 | 28.0 |
  | 512 | ORT CPU | model | 11.53 ± 0.21 | 96.0 ± 3.8 | 120.2 ± 2.5 | 93.1 ± 1.4 | 120.7 |
  | 512 | OV CPU | model | 11.50 ± 0.01 | 89.4 ± 0.1 | 128.6 ± 0.1 | 99.5 ± 0.1 | 127.7 |
  | 512 | OV iGPU | model | 9.18 ± 0.02 | 143.2 ± 0.1 | 64.1 ± 0.2 | 46.0 ± 0.2 | 69.6 |
  | 224 | ORT CPU | e2e | 9.77 | 137.8 | 70.9 | 51.6 | 69.6 |
  | 224 | OV CPU | e2e | 10.74 | 139.7 | 76.9 | 57.9 | 75.3 |
  | 224 | OV iGPU | e2e | 8.50 | 119.0 | 71.4 | 49.1 | 70.2 |
  | 512 | ORT CPU | e2e | 11.00 | 59.3 | 185.6 | 141.8 | 186.1 |
  | 512 | OV CPU | e2e | 11.09 | 54.0 | 205.3 | 157.2 | 202.9 |
  | 512 | OV iGPU | e2e | 9.27 | 73.0 | 126.9 | 91.3 | 132.6 |

  The new driver lowers model-only iGPU energy by 8-18 % (power 9.3 → 8.5 W
  at 224, 10.1 → 9.2 W at 512). End-to-end iGPU and all CPU figures change by
  less than 5 %. The paper's energy table (Table 7) and the graphical
  abstract now use these numbers.
- **Artifacts**: `results/n100_run012/` (per-config JSON/logs for the driver
  matrix, `int8_variants/results.json`, `int8_variants/results_workarounds.json`,
  `layer_diff_{base,fp32}.{csv,json}`, `setup_and_driver.txt` with the
  driver env, ICD file and package checksums); scripts
  `results/run012_{bench_n100,int8_variants,layer_diff}.py` (working copies
  in `edge/`). Quantized IRs remain on the N100 in
  `~/eye-edge/run012/int8_variants/`.
- **Verdict**:
  - **The driver was not the cause.** The INT8 iGPU collapse is identical
    under Intel's current compute-runtime and under the 2022 Debian driver,
    and identical at f16 and f32 precision. The run 004 attribution and the
    paper footnote were wrong and are corrected.
  - **Cause**: the OpenVINO 2026.4.1 GPU plugin mis-computes a quantized
    depthwise GroupConvolution whose weight FakeQuantize has ranges varying
    along the kernel-width axis (shape [C,1,1,1,3], as NNCF emitted here):
    about 5 % error in every channel and NaN in the last two. The CPU plugin
    is correct on the same graph. Reproduced in a single isolated layer;
    independent of driver, precision, stride, activation quantization, NNCF
    preset and target device. Worth reporting to OpenVINO (GPU plugin) and
    possibly NNCF (range shape).
  - **Fix**: excluding depthwise convolutions from quantization
    (`ignored_scope=IgnoredScope(types=["GroupConvolution"])`) restores iGPU
    INT8 to CPU-INT8 accuracy (0.8037 vs 0.8043). It is still not useful:
    at 2.83 ms it is no faster than the iGPU in f16 (2.75 ms) and slower than
    the CPU in FP32 (2.06 ms), and it keeps INT8's accuracy and calibration
    cost relative to FP32 (−0.018 mean AUC).
  - **iGPU latency is not a driver fault either.** The new driver shaves
    about 9 % at 224 px (3.02 → 2.75 ms) and changes 512 px by +1 % (6.94 →
    7.04 ms) at the default precision. At 224 px the iGPU remains slower than
    the CPU because per-inference dispatch overhead dominates a model this
    small. At 512 px it is the fastest backend, as in run 009.
  - Paper/run 009 numbers measured with the old driver stand: the new
    driver changes no AUC and moves iGPU latency by −9 % to +1 %.
- **Limitations**: one OpenVINO version (2026.4.1); an older OpenVINO was not
  tried, so whether this is a regression is unknown. Layer diff on 3 images.
  Single latency run per config.

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

## Target hardware for edge benchmarking (N100: run 004; Pi: run 006)

See `eye-disease-edge-ai-project.md` memory + `brainstorm.md` §3 for full
specs. Summary: N100 mini PC (`n100m`, x86, 15GB RAM, Intel UHD iGPU,
OpenVINO path) and Raspberry Pi 3B (`pi@192.168.188.108`, ARMv7, 870MB RAM
hard ceiling, TFLite path, no camera attached).
