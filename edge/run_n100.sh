#!/bin/bash
# Latency/accuracy driver on the N100 (run 009+): one process per config, sequential,
# 20 s idle gap. Model/size/manifest via env vars passed through to bench_n100.py.
# From ~/eye-edge:
#   OUT=run009/lat_512 ONNX=artifacts/run007c_eyeD512_fp32.onnx IMG=512 \
#   MANIFEST=artifacts/val_manifest_run009.csv REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
#   LABELS=D_eye,G,C,A,H nohup ./run_n100.sh > run009/lat_512.log 2>&1 &
cd "$(dirname "$0")"
CONFIGS="${CONFIGS:-ort_fp32_cpu ov_fp32_cpu ov_fp32_gpu}"
mkdir -p "$OUT"
EXTRA=""
[ -n "$N" ] && EXTRA="--n $N"
[ -n "$IMGDIR" ] && EXTRA="$EXTRA --img-dir $IMGDIR"
echo "suite start $(date -Is) | load $(cut -d' ' -f1-3 /proc/loadavg) | governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) | pkg temp $(($(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)/1000)) C"
free -m
for c in $CONFIGS; do
  echo "=== $c start $(date -Is)"
  TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"
  $TIMEV .venv/bin/python bench_n100.py run "$c" --onnx "$ONNX" --img "$IMG" --manifest "$MANIFEST" \
    --ref "$REF" --label-cols "$LABELS" --out-dir "$OUT" $EXTRA > "$OUT/log_$c.txt" 2>&1
  echo "=== $c exit $? $(date -Is)"
  sleep 20
done
free -m
echo "suite end $(date -Is) | load $(cut -d' ' -f1-3 /proc/loadavg)"
echo SUITE_DONE
