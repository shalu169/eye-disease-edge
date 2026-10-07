## Run 009 — Edge re-benchmark at 512 px (N100 ORT/OpenVINO CPU+iGPU, Pi 3B TFLite) + N100 RAPL energy, 224 vs 512

> DRAFT (Phase 1 only). Phase 2 (test-set parity of the run008 final checkpoint) goes into
> `results/run009_ENTRY.md`. All numbers below are generated from the result JSONs by
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
- **Issues / caveats**:
  - Single latency run per config. Pi parity on 300 images only in Phase 1.
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
  - Mac logs: `results/logs/run009_export_*.log`.
  - Tables: `results/run009_tables.md`.
  - Export summaries: `edge/artifacts/run007c_eyeD512_{export,tflite_export}_summary.json`, `run007b_eyeD384_*`.
