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
