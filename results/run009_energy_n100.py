"""
Energy per inference for the run002 model on the N100 mini PC (run 005).

Energy source: Intel RAPL via powercap sysfs (package-0 plus its core/uncore
subzones). The counters are cumulative microjoules that wrap at
max_energy_range_uj; we sample every ~SAMPLE_S seconds during a run so that at
most one wrap can occur between samples, and integrate the deltas.

For each config (one process per config) and for an idle baseline:
  - model mode: sustained batch-1 inference loop on ONE fixed preprocessed input
  - e2e mode:   per iteration, JPEG decode + resize + normalize (bench_n100.preprocess)
                + inference, cycling through the val images
Each mode runs REPEATS times for DURATION seconds with a COOLDOWN sleep before
each repeat (so every repeat starts with a similar thermal / PL2 budget state).
Per repeat we record package/core/uncore energy, wall time, iterations, a
per-second package power timeseries, CPU temps before/after, process CPU time
and system-wide CPU busy fraction (to detect interference).

Usage on the N100 (from ~/eye-edge, needs read access to energy_uj):
  .venv/bin/python energy_n100.py all                 # idle(pre) + 4 configs + idle(post) + summary
  .venv/bin/python energy_n100.py idle <tag>          # one idle baseline process
  .venv/bin/python energy_n100.py run <config>        # one config process
  .venv/bin/python energy_n100.py summarize           # aggregate results_*.json -> summary.json
  .venv/bin/python energy_n100.py selftest            # stub-RAPL test of the integration logic
Options (env vars): ENERGY_DURATION_S (60), ENERGY_REPEATS (3), ENERGY_COOLDOWN_S (30),
  ENERGY_STUB=1 (use synthetic RAPL counter, for testing without root), ENERGY_OUT (".")
Run 009+: model/size/manifest flags are the same as bench_n100.py (defaults = run002 at 224),
plus --configs (comma list, default all of CONFIGS); `all` passes them through to every step:
  ENERGY_OUT=run009/energy_512 .venv/bin/python energy_n100.py all --configs ort_fp32_cpu,ov_fp32_cpu,ov_fp32_gpu \
      --onnx artifacts/run007c_eyeD512_fp32.onnx --img 512 --manifest artifacts/val_manifest_run009.csv \
      --ref artifacts/run007c_eyeD512_torch_cpu_val_probs.npy --label-cols D_eye,G,C,A,H
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

DURATION_S = float(os.environ.get("ENERGY_DURATION_S", 60))
REPEATS = int(os.environ.get("ENERGY_REPEATS", 3))
COOLDOWN_S = float(os.environ.get("ENERGY_COOLDOWN_S", 30))
OUT = os.environ.get("ENERGY_OUT", ".")
STUB = os.environ.get("ENERGY_STUB") == "1"
SAMPLE_S = 1.0
WARMUP = 50
CONFIGS = ["ort_fp32_cpu", "ov_fp32_cpu", "ov_int8_cpu", "ov_fp32_gpu"]  # ov_int8_gpu skipped: broken (run004, AUC 0.62)

POWERCAP = "/sys/class/powercap"
ZONES = {"package": "intel-rapl:0", "core": "intel-rapl:0:0", "uncore": "intel-rapl:0:1"}


# ---------------------------------------------------------------- energy readers
class RaplReader:
    def __init__(self):
        self.zones, self.max_range = {}, {}
        for key, z in ZONES.items():
            base = f"{POWERCAP}/{z}"
            if not os.path.exists(f"{base}/energy_uj"):
                continue
            self.zones[key] = f"{base}/energy_uj"
            self.max_range[key] = int(open(f"{base}/max_energy_range_uj").read())
            name = open(f"{base}/name").read().strip()
            assert (key, name) in {("package", "package-0"), ("core", "core"), ("uncore", "uncore")}, (key, name)
        # package is mandatory; subzones are optional (skip if not readable)
        for key in list(self.zones):
            try:
                int(open(self.zones[key]).read())
            except PermissionError:
                if key == "package":
                    raise
                print(f"warning: {key} energy_uj not readable, skipping subzone", flush=True)
                del self.zones[key], self.max_range[key]

    def read(self):
        return {k: int(open(p).read()) for k, p in self.zones.items()}


class StubReader:
    """Synthetic counter: constant power, small max range so it wraps often."""

    def __init__(self, watts=None):
        self.watts = watts or {"package": 5.0, "core": 2.0, "uncore": 0.5}
        self.max_range = {k: 7_000_000 for k in self.watts}  # 7 J -> wraps every ~1.4 s at 5 W
        self.t0 = time.perf_counter()
        self.offset = {k: 6_000_000 for k in self.watts}  # start near the wrap point

    def read(self):
        dt = time.perf_counter() - self.t0
        return {k: int(self.offset[k] + w * dt * 1e6) % self.max_range[k] for k, w in self.watts.items()}


def delta_uj(e0, e1, max_range):
    d = e1 - e0
    return d + max_range if d < 0 else d


# ---------------------------------------------------------------- system probes
def read_temps():
    t = {}
    for z in sorted(os.listdir("/sys/class/thermal")) if os.path.isdir("/sys/class/thermal") else []:
        if z.startswith("thermal_zone"):
            try:
                t[open(f"/sys/class/thermal/{z}/type").read().strip()] = int(open(f"/sys/class/thermal/{z}/temp").read()) / 1000
            except OSError:
                pass
    hw = "/sys/class/hwmon"
    for h in sorted(os.listdir(hw)) if os.path.isdir(hw) else []:
        try:
            if open(f"{hw}/{h}/name").read().strip() != "coretemp":
                continue
            for f in sorted(os.listdir(f"{hw}/{h}")):
                if f.startswith("temp") and f.endswith("_input"):
                    label = open(f"{hw}/{h}/{f.replace('_input', '_label')}").read().strip()
                    t[f"coretemp {label}"] = int(open(f"{hw}/{h}/{f}").read()) / 1000
        except OSError:
            pass
    return t


def read_proc_stat():
    try:
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        idle = v[3] + v[4]
        return sum(v), idle
    except OSError:
        return None


def cpu_freqs_mhz():
    out = []
    for i in range(os.cpu_count() or 0):
        try:
            out.append(int(open(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq").read()) / 1000)
        except OSError:
            pass
    return out


def power_limits():
    base = f"{POWERCAP}/{ZONES['package']}"
    lim = {}
    for c in range(3):
        try:
            name = open(f"{base}/constraint_{c}_name").read().strip()
            lim[name] = {"power_limit_w": int(open(f"{base}/constraint_{c}_power_limit_uw").read()) / 1e6,
                         "time_window_s": int(open(f"{base}/constraint_{c}_time_window_us").read()) / 1e6}
        except OSError:
            pass
    return lim


# ---------------------------------------------------------------- core measurement
def measure(reader, step, duration=DURATION_S):
    """Run step(i) repeatedly for `duration` s (step=None -> idle sleep), integrating energy.

    Returns a dict with energies (J), wall time, iteration count and a per-sample
    package power timeseries.
    """
    temps_before, freq_before = read_temps(), cpu_freqs_mhz()
    stat0, ru0 = read_proc_stat(), resource.getrusage(resource.RUSAGE_SELF)
    acc = {k: 0 for k in reader.max_range}
    prev = reader.read()
    t0 = time.perf_counter()
    t_prev, next_sample, n, series = t0, t0 + SAMPLE_S, 0, []

    def sample(now):
        nonlocal prev, t_prev
        cur = reader.read()
        d = {k: delta_uj(prev[k], cur[k], reader.max_range[k]) for k in acc}
        for k in acc:
            acc[k] += d[k]
        series.append((round(now - t0, 3), round(d["package"] / 1e6 / (now - t_prev), 3)))
        prev, t_prev = cur, now

    while True:
        if step is None:
            time.sleep(max(0.0, min(next_sample, t0 + duration) - time.perf_counter()))
        else:
            step(n)
            n += 1
        now = time.perf_counter()
        if now - t0 >= duration:
            sample(now)
            break
        if now >= next_sample:
            sample(now)
            next_sample += SAMPLE_S
            if next_sample <= now:  # a single step overran a whole interval; resync
                next_sample = now + SAMPLE_S
    wall = time.perf_counter() - t0
    ru1, stat1 = resource.getrusage(resource.RUSAGE_SELF), read_proc_stat()
    max_gap = max(b[0] - a[0] for a, b in zip([(0.0, 0)] + series[:-1], series))
    res = {
        "wall_s": round(wall, 4),
        "iterations": n,
        "energy_j": {k: round(v / 1e6, 4) for k, v in acc.items()},
        "avg_power_w": {k: round(v / 1e6 / wall, 4) for k, v in acc.items()},
        "n_samples": len(series),
        "max_sample_gap_s": round(max_gap, 3),
        "package_power_series_w": series,
        "process_cpu_s": round((ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime), 3),
        "temps_before_c": temps_before,
        "temps_after_c": read_temps(),
        "cpu_freq_mhz_before": freq_before,
        "cpu_freq_mhz_after": cpu_freqs_mhz(),
    }
    if stat0 and stat1:
        tot, idle = stat1[0] - stat0[0], stat1[1] - stat0[1]
        res["system_cpu_busy_frac"] = round(1 - idle / tot, 4) if tot else None
    half = [p for t, p in series if t > duration / 2]
    res["package_power_w_second_half"] = round(float(np.mean(half)), 4) if half else None
    if n:
        res["inferences_per_s"] = round(n / wall, 3)
        res["mj_per_inference"] = {k: round(v / 1e3 / n, 4) for k, v in acc.items()}
    return res


def make_reader():
    return StubReader() if STUB else RaplReader()


def common_meta():
    return {"host": platform.node(), "python": platform.python_version(), "stub_reader": STUB,
            "duration_s": DURATION_S, "repeats": REPEATS, "cooldown_s": COOLDOWN_S, "sample_s": SAMPLE_S,
            "power_limits": power_limits(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def run_idle(tag):
    reader = make_reader()
    reps = []
    for r in range(REPEATS):
        time.sleep(COOLDOWN_S)
        m = measure(reader, None)
        print(f"idle[{tag}] rep {r}: {m['avg_power_w']} W, temps {m['temps_after_c'].get('x86_pkg_temp')} C, "
              f"sys busy {m.get('system_cpu_busy_frac')}", flush=True)
        reps.append(m)
    out = {"config": f"idle_{tag}", **common_meta(), "repeats_data": reps,
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    with open(f"{OUT}/results_idle_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)


def run_config(config):
    import pandas as pd
    import bench_n100 as bn
    from bench_n100 import make_infer, preprocess

    reader = make_reader()
    t = time.perf_counter()
    infer, runtime = make_infer(config)
    load_s = time.perf_counter() - t
    files = [f"{bn.IMG_DIR}/{f}" for f in pd.read_csv(bn.VAL_MANIFEST)["filename"]]
    for p in files:  # warm the page cache so e2e measures decode, not disk
        with open(p, "rb") as fh:
            fh.read()
    x = preprocess(files[0])
    for _ in range(WARMUP):
        infer(x)
    for i in range(WARMUP):
        infer(preprocess(files[i]))

    def model_step(i):
        infer(x)

    def e2e_step(i):
        infer(preprocess(files[i % len(files)]))

    data = {}
    for mode, step in (("model", model_step), ("e2e", e2e_step)):
        data[mode] = []
        for r in range(REPEATS):
            time.sleep(COOLDOWN_S)
            m = measure(reader, step)
            print(f"{config} {mode} rep {r}: {m['avg_power_w']['package']:.3f} W pkg, "
                  f"{m['inferences_per_s']:.1f} inf/s, {m['mj_per_inference']['package']:.2f} mJ/inf, "
                  f"pkg temp {m['temps_before_c'].get('x86_pkg_temp')}->{m['temps_after_c'].get('x86_pkg_temp')} C, "
                  f"sys busy {m.get('system_cpu_busy_frac')}", flush=True)
            data[mode].append(m)
    out = {"config": config, "runtime": runtime, **common_meta(), "onnx": bn.ONNX, "img": bn.IMG_SIZE,
           "manifest": bn.VAL_MANIFEST, "model_load_s": round(load_s, 3),
           "n_images_cycled": len(files), "warmup": WARMUP, "batch": 1, "modes": data,
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    with open(f"{OUT}/results_{config}.json", "w") as fh:
        json.dump(out, fh, indent=2)


# ---------------------------------------------------------------- aggregation
def ms(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            "n": int(len(a)), "values": [round(float(v), 4) for v in a]}


def summarize():
    idle_reps = []
    for tag in ("pre", "post"):
        p = f"{OUT}/results_idle_{tag}.json"
        if os.path.exists(p):
            idle_reps += json.load(open(p))["repeats_data"]
    idle_w = ms([m["avg_power_w"]["package"] for m in idle_reps])
    summary = {"idle_package_w": idle_w,
               "idle_by_zone_w": {z: ms([m["avg_power_w"][z] for m in idle_reps]) for z in idle_reps[0]["avg_power_w"]},
               "configs": {}}
    for config in CONFIGS:
        p = f"{OUT}/results_{config}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        summary["configs"][config] = {"runtime": r["runtime"], "peak_rss_mb": r["peak_rss_mb"]}
        for mode, reps in r["modes"].items():
            pw = [m["avg_power_w"]["package"] for m in reps]
            ips = [m["inferences_per_s"] for m in reps]
            summary["configs"][config][mode] = {
                "package_w": ms(pw),
                "package_w_second_half": ms([m["package_power_w_second_half"] for m in reps]),
                "zone_w": {z: ms([m["avg_power_w"][z] for m in reps]) for z in reps[0]["avg_power_w"]},
                "inferences_per_s": ms(ips),
                "mj_per_inference_total": ms([m["mj_per_inference"]["package"] for m in reps]),
                # above-idle: (P_load - mean P_idle) / throughput, per repeat
                "mj_per_inference_above_idle": ms([(p_ - idle_w["mean"]) / i * 1e3 for p_, i in zip(pw, ips)]),
                "process_cpu_cores": ms([m["process_cpu_s"] / m["wall_s"] for m in reps]),
                "system_cpu_busy_frac": ms([m["system_cpu_busy_frac"] for m in reps]),
                "pkg_temp_before_c": [m["temps_before_c"].get("x86_pkg_temp") for m in reps],
                "pkg_temp_after_c": [m["temps_after_c"].get("x86_pkg_temp") for m in reps],
            }
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"idle package: {idle_w['mean']:.3f} ± {idle_w['std']:.3f} W (n={idle_w['n']})")
    print(f"{'config':14s} {'mode':5s} {'pkg W':>14s} {'inf/s':>14s} {'mJ/inf total':>16s} {'mJ/inf >idle':>16s} {'temp after':>10s}")
    for c, d in summary["configs"].items():
        for mode in ("model", "e2e"):
            s = d[mode]
            f = lambda k: f"{s[k]['mean']:.3f}±{s[k]['std']:.3f}"
            print(f"{c:14s} {mode:5s} {f('package_w'):>14s} {f('inferences_per_s'):>14s} "
                  f"{f('mj_per_inference_total'):>16s} {f('mj_per_inference_above_idle'):>16s} {max(s['pkg_temp_after_c'] or [0])!s:>10s}")


def run_all(passthrough):
    env = dict(os.environ)
    py = sys.executable
    os.makedirs(OUT, exist_ok=True)
    steps = [["idle", "pre"]] + [["run", c] for c in CONFIGS] + [["idle", "post"]]
    for s in steps:
        print(f"=== {' '.join(s)} ({time.strftime('%H:%M:%S')})", flush=True)
        with open(f"{OUT}/log_{s[1] if s[0] == 'run' else 'idle_' + s[1]}.txt", "w") as log:
            rc = subprocess.run([py, __file__, *s, *passthrough], env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        print(open(log.name).read(), flush=True)
        if rc != 0:
            print(f"step {s} failed rc={rc}", flush=True)
    summarize()


# ---------------------------------------------------------------- self test
def selftest():
    # 1) wrap arithmetic
    assert delta_uj(10, 30, 100) == 20 and delta_uj(90, 5, 100) == 15
    # 2) stub integration over many wraps, idle and busy loops
    for step in (None, lambda i: sum(range(2000))):
        r = StubReader()
        m = measure(r, step, duration=5.0)
        for k, w in r.watts.items():
            assert abs(m["avg_power_w"][k] - w) / w < 0.01, (k, m["avg_power_w"][k], w)
        assert m["max_sample_gap_s"] < 1.5, m["max_sample_gap_s"]
        print("ok", "idle" if step is None else "busy", m["avg_power_w"], m["n_samples"], "samples",
              m.get("inferences_per_s"), "it/s", "gap", m["max_sample_gap_s"])
    # 3) a step longer than the wrap period would undercount -> sanity: sample gap reported
    print("selftest passed")


if __name__ == "__main__":
    import argparse

    import bench_n100

    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("cmd", choices=["selftest", "idle", "run", "summarize", "all"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    bench_n100.add_model_args(ap)
    a = ap.parse_args()
    CONFIGS = a.configs.split(",")
    bench_n100.configure(a)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "idle":
        run_idle(a.arg)
    elif a.cmd == "run":
        assert a.arg in CONFIGS, a.arg
        run_config(a.arg)
    elif a.cmd == "summarize":
        summarize()
    elif a.cmd == "all":
        run_all(sys.argv[2:])
