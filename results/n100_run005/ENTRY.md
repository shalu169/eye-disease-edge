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
