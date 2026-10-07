"""
Run 009: build the tables for results/run009_ENTRY.md directly from the result JSONs
(no hand-copied numbers). Also re-computes every AUC with sklearn from the saved
per-image probabilities and checks it against the JSON value, and recomputes
|dp| against the PyTorch CPU reference.

Usage (repo root): .venv/bin/python results/run009_aggregate.py > results/run009_tables.md
"""
import csv
import collections
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "results"
A = "edge/artifacts"
N1, PI = f"{R}/n100_run009", f"{R}/pi_run009"

SETS = {  # tag -> (manifest, torch ref, label cols)
    "512": (f"{A}/val_manifest_run009.csv", f"{A}/run007c_eyeD512_torch_cpu_val_probs.npy", ["D_eye", "G", "C", "A", "H"]),
    "384": (f"{A}/val_manifest_run009.csv", f"{A}/run007b_eyeD384_torch_cpu_val_probs.npy", ["D_eye", "G", "C", "A", "H"]),
    "224": (f"{A}/val_manifest.csv", f"{A}/torch_cpu_val_probs.npy", ["D", "G", "C", "A", "H"]),
}


def j(p):
    return json.load(open(p))


def check_probs(res_path, probs_path, key):
    """sklearn AUC from saved probs == JSON AUC, and |dp| vs torch recomputed."""
    man, ref, cols = SETS[key]
    r = j(res_path)
    assert os.path.basename(r["manifest"]) == os.path.basename(man), (r["manifest"], man)
    p = np.load(probs_path)
    n = len(p)
    df = pd.read_csv(man).iloc[:n]
    t = np.load(ref)[:n]
    auc = {c: round(float(roc_auc_score(df[c], p[:, i])), 4) for i, c in enumerate(cols)}
    ok = auc == r["val_auc"]
    d = np.abs(p - t)
    assert abs(d.max() - r["max_abs_prob_diff_vs_torch"]) < 1e-9, (res_path, d.max(), r["max_abs_prob_diff_vs_torch"])
    return ok, n


def fmt_lat(l):
    return f"{l['mean']:.2f} / {l['p50']:.2f} / {l['p95']:.2f} / {l['p99']:.2f}"


def latency_tables():
    print("## Latency / RSS / parity (val manifest, model-only latency, batch 1)\n")
    print("| device | res | config | n imgs | mean AUC (5) | D AUC (label of row) | D patient | max \\|Δp\\| | mean \\|Δp\\| | latency mean / p50 / p95 / p99 (ms) | preprocess mean (ms) | e2e ≈ (ms) | load (s) | peak RSS (MB) | sklearn AUC == JSON |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for dev, base, cfgs in (("N100", N1, ["ort_fp32_cpu", "ov_fp32_cpu", "ov_fp32_gpu"]),
                            ("Pi 3B", PI, ["tfl_fp32_t4", "tfl_fp32_t1"])):
        for res in ("224", "384", "512"):
            for c in cfgs:
                p = f"{base}/lat_{res}/results_{c}.json"
                if not os.path.exists(p):
                    continue
                r = j(p)
                ok, n = check_probs(p, f"{base}/lat_{res}/probs_{c}.npy", res)
                l = r["latency_ms"]
                d0 = list(r["val_auc"].items())[0]
                print(f"| {dev} | {res} | {c} | {n} | {r['mean_auc']:.4f} | {d0[0]} {d0[1]:.4f} | {r.get('D_patient_auc', '–')} | "
                      f"{r['max_abs_prob_diff_vs_torch']:.1e} | {r['mean_abs_prob_diff_vs_torch']:.1e} | {fmt_lat(l)} | "
                      f"{r['preprocess_ms_mean']:.2f} | {l['mean'] + r['preprocess_ms_mean']:.1f} | {r['model_load_s']:.3f} | {r['peak_rss_mb']:.1f} | {ok} |")
    print()


def pi_windows(d):
    w = {}
    for line in open(f"{d}.log"):
        m = re.match(r"=== (\S+) (start|exit \d+) (\S+)", line)
        if m:
            w.setdefault(m.group(1), {})["s" if m.group(2) == "start" else "e"] = dt.datetime.fromisoformat(m.group(3))
    rows = list(csv.DictReader(open(f"{d}/vcgencmd_samples.tsv"), delimiter="\t"))
    out = {}
    for c, se in w.items():
        sel = [r for r in rows if se["s"] <= dt.datetime.fromisoformat(r["time"]) <= se["e"]]
        cnt = collections.Counter((r["throttled"], int(r["arm_clock_hz"]) // 1_000_000) for r in sel)
        temps = [float(r["temp"].rstrip("'C")) for r in sel]
        out[c] = {"n": len(sel), "states": dict(cnt),
                  "frac_600": sum(v for (t, mhz), v in cnt.items() if mhz <= 600) / len(sel),
                  "frac_uv_now": sum(v for (t, mhz), v in cnt.items() if int(t, 16) & 0x1) / len(sel),
                  "temp": (min(temps), max(temps)),
                  "memavail_min_mb": min(int(r["mem_available_kb"]) for r in sel) // 1024,
                  "swapfree_min_mb": min(int(r["swap_free_kb"]) for r in sel) // 1024}
    return out


def pi_throttle_table(dirs):
    print("## Pi throttling samples (vcgencmd every 5 s during each config process)\n")
    print("| run | config | samples | states (throttled, MHz): count | frac at ≤600 MHz | frac under-voltage now | temp °C | MemAvailable min (MB) | SwapFree min (MB) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        if not os.path.exists(f"{d}/vcgencmd_samples.tsv"):
            continue
        for c, s in pi_windows(d).items():
            st = ", ".join(f"{k[0]}@{k[1]}: {v}" for k, v in sorted(s["states"].items()))
            print(f"| {os.path.basename(d)} | {c} | {s['n']} | {st} | {s['frac_600']:.2f} | {s['frac_uv_now']:.2f} | "
                  f"{s['temp'][0]:.1f}-{s['temp'][1]:.1f} | {s['memavail_min_mb']} | {s['swapfree_min_mb']} |")
    print()


def energy_table():
    print("## N100 energy (RAPL package; mean ± std over 3 × 60 s repeats)\n")
    print("| res | config | mode | pkg W (load) | pkg W 2nd half | W idle | inf/s | mJ/inf total | mJ/inf above idle | proc CPU cores | sys busy | pkg temp after (°C) | peak RSS (MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for res in ("224", "512"):
        p = f"{N1}/energy_{res}/summary.json"
        if not os.path.exists(p):
            continue
        s = j(p)
        iw = s["idle_package_w"]
        f = lambda x, k=3: f"{x['mean']:.{k}f} ± {x['std']:.{k}f}"
        for c, d in s["configs"].items():
            for mode in ("model", "e2e"):
                m = d[mode]
                print(f"| {res} | {c} | {mode} | {f(m['package_w'], 2)} | {f(m['package_w_second_half'], 2)} | {f(iw, 2)} | "
                      f"{f(m['inferences_per_s'], 1)} | {f(m['mj_per_inference_total'], 1)} | {f(m['mj_per_inference_above_idle'], 1)} | "
                      f"{f(m['process_cpu_cores'], 2)} | {f(m['system_cpu_busy_frac'], 2)} | {max(m['pkg_temp_after_c'])} | {d['peak_rss_mb']} |")
        print(f"\nidle {res} run: package {f(iw)} W (n={iw['n']}, values {iw['values']}); zones "
              + ", ".join(f"{z} {f(v)}" for z, v in s["idle_by_zone_w"].items()) + "\n")


def test_parity():
    """Phase 2: run008 final checkpoint on the held-out test set (results/run008_splits/test.csv)."""
    man, ref = f"{A}/test_manifest_run008.csv", f"{A}/run008final_torch_cpu_test_probs.npy"
    if not os.path.exists(ref):
        return
    cols = ["D_eye", "G", "C", "A", "H"]
    df, t = pd.read_csv(man), np.load(ref)
    print("## Phase 2: test-set parity, run008 final checkpoint (n = %d test images)\n" % len(df))
    print("| runtime | n | D_eye | G | C | A | H | mean (5) | D patient | max \\|Δp\\| vs torch | mean \\|Δp\\| vs torch | JSON AUC == sklearn |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [("PyTorch CPU (Mac, reference)", ref, None),
            ("ONNX Runtime CPU (Mac)", f"{N1}/mac_onnx_parity_test/probs_ort_fp32_cpu.npy", f"{N1}/mac_onnx_parity_test/results_ort_fp32_cpu.json"),
            ("TFLite FP32 (Mac, TF 2.18)", f"{A}/run008final_tflite_mac_fp32_probs.npy", None)]
    for c in ("ort_fp32_cpu", "ov_fp32_cpu", "ov_fp32_gpu"):
        rows.append((f"N100 {c}", f"{N1}/test_final/probs_{c}.npy", f"{N1}/test_final/results_{c}.json"))
    rows.append(("Pi 3B tfl_fp32_t4", f"{PI}/test_final/probs_tfl_fp32_t4.npy", f"{PI}/test_final/results_tfl_fp32_t4.json"))
    for name, pp, jp in rows:
        if not os.path.exists(pp):
            continue
        p = np.load(pp)
        assert len(p) == len(df), (pp, len(p))
        auc = {c: round(float(roc_auc_score(df[c], p[:, i])), 4) for i, c in enumerate(cols)}
        dpat = round(float(roc_auc_score(df["D"], p[:, 0])), 4)
        d = np.abs(p - t)
        ok = "–"
        if jp:
            r = j(jp)
            ok = str(auc == r["val_auc"] and abs(d.max() - r["max_abs_prob_diff_vs_torch"]) < 1e-9)
        print(f"| {name} | {len(p)} | " + " | ".join(f"{auc[c]:.4f}" for c in cols)
              + f" | {np.mean(list(auc.values())):.4f} | {dpat:.4f} | {d.max():.1e} | {d.mean():.1e} | {ok} |")
    print()
    print("Latency of the final model on the test run (same protocol):\n")
    print("| device | config | latency mean / p50 / p95 / p99 (ms) | preprocess (ms) | peak RSS (MB) |")
    print("|---|---|---|---|---|")
    for dev, base, cfgs in (("N100", N1, ["ort_fp32_cpu", "ov_fp32_cpu", "ov_fp32_gpu"]), ("Pi 3B", PI, ["tfl_fp32_t4"])):
        for c in cfgs:
            jp = f"{base}/test_final/results_{c}.json"
            if os.path.exists(jp):
                r = j(jp)
                print(f"| {dev} | {c} | {fmt_lat(r['latency_ms'])} | {r['preprocess_ms_mean']:.2f} | {r['peak_rss_mb']:.1f} |")
    print()


if __name__ == "__main__":
    latency_tables()
    pi_throttle_table([f"{PI}/lat_512", f"{PI}/lat_384", f"{PI}/lat_224", f"{PI}/test_final"])
    energy_table()
    test_parity()
