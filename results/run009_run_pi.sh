#!/bin/bash
# Pi driver: one process per config, sequential, 60 s idle gap between configs so
# thermal state is comparable. Run under nohup from ~/eye-edge:
#   run 006 (defaults):  nohup ./run_pi.sh > suite.log 2>&1 &
#   run 009+: OUT=run009/lat_512 MODEL=artifacts/run007c_eyeD512_fp32.tflite IMG=512 \
#             MANIFEST=artifacts/val_manifest_run009.csv REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy \
#             LABELS=D_eye,G,C,A,H N=300 CONFIGS="tfl_fp32_t4 tfl_fp32_t1" nohup ./run_pi.sh > run009/lat_512.log 2>&1 &
# If OUT is set, a background sampler logs vcgencmd throttled/clock/temp + MemAvailable/SwapFree
# every SAMPLE_S (5) s to $OUT/vcgencmd_samples.tsv for the whole suite, and free -m is
# recorded before/after every config.
cd "$(dirname "$0")"
CONFIGS="${CONFIGS:-tfl_fp32_t4 tfl_fp32_t1 tfl_drq_t4 tfl_drq_t1 tfl_int8_t4 tfl_int8_t1}"
SAMPLE_S="${SAMPLE_S:-5}"
ARGS=()
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
[ -n "$IMG" ] && ARGS+=(--img "$IMG")
[ -n "$MANIFEST" ] && ARGS+=(--manifest "$MANIFEST")
[ -n "$REF" ] && ARGS+=(--ref "$REF")
[ -n "$LABELS" ] && ARGS+=(--label-cols "$LABELS")
[ -n "$N" ] && ARGS+=(--n "$N")
LOGDIR=.
if [ -n "$OUT" ]; then
  mkdir -p "$OUT"; LOGDIR="$OUT"; ARGS+=(--out-dir "$OUT")
  (
    echo -e "time\tthrottled\tarm_clock_hz\ttemp\tmem_available_kb\tswap_free_kb"
    while true; do
      echo -e "$(date -Is)\t$(vcgencmd get_throttled | cut -d= -f2)\t$(vcgencmd measure_clock arm | cut -d= -f2)\t$(vcgencmd measure_temp | cut -d= -f2)\t$(awk '/MemAvailable/{print $2}' /proc/meminfo)\t$(awk '/SwapFree/{print $2}' /proc/meminfo)"
      sleep "$SAMPLE_S"
    done
  ) > "$OUT/vcgencmd_samples.tsv" 2>&1 &
  SAMPLER=$!
  trap 'kill $SAMPLER 2>/dev/null' EXIT
fi
echo "suite start $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
echo "undervoltage dmesg lines at start: $(dmesg 2>/dev/null | grep -ci undervoltage)"
for c in $CONFIGS; do
  echo "=== $c start $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
  free -m
  /usr/bin/time -v .venv/bin/python bench_pi.py "$c" "${ARGS[@]}" > "$LOGDIR/log_$c.txt" 2>&1
  echo "=== $c exit $? $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
  free -m
  sleep 60
done
echo "undervoltage dmesg lines at end: $(dmesg 2>/dev/null | grep -ci undervoltage)"
echo "suite end $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
echo SUITE_DONE
