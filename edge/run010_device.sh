#!/bin/bash
# Run 010 device driver: one process per (runtime, mode), sequential, 30 s idle gap.
# Usage (inside the bundle dir on the device, under nohup):
#   bash run010_device.sh TAG PY "RUNTIMES" WARMUP ITERS THREADS
#   N100: bash run010_device.sh final ~/eye-edge/.venv/bin/python "ort ov" 50 500 0
#   Pi  : bash run010_device.sh final ~/eye-edge/.venv/bin/python "tflite" 10 100 4
set -u
TAG=$1; PY=$2; RTS=$3; WU=$4; IT=$5; TH=$6
mkdir -p out
log() { echo "[$(date '+%F %T')] $*" | tee -a out/suite.log; }
probe() {
  if command -v vcgencmd >/dev/null 2>&1; then
    log "$1 throttled=$(vcgencmd get_throttled) clock=$(vcgencmd measure_clock arm) temp=$(vcgencmd measure_temp)"
  else
    log "$1 load=$(cut -d' ' -f1-3 /proc/loadavg) mhz=$(grep -m1 MHz /proc/cpuinfo | awk '{print $4}')"
  fi
}
TIMEV=""; [ -x /usr/bin/time ] && TIMEV="/usr/bin/time -v"  # absent on the N100; RSS comes from ru_maxrss anyway
log "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-unset}"
log "start tag=$TAG runtimes=$RTS warmup=$WU iters=$IT threads=$TH host=$(hostname)"
for rt in $RTS; do
  for mode in plain head gradcam1 gradcam5; do
    cfg="${rt}_${mode}"
    probe "before $cfg"
    $TIMEV "$PY" gradcam_bench.py --runtime "$rt" --mode "$mode" --tag "$TAG" --threads "$TH" \
      --warmup "$WU" --iters "$IT" --out "out/results_${cfg}.json" > "out/log_${cfg}.txt" 2>&1
    log "end $cfg rc=$?"
    probe "after $cfg"
    sleep 30
  done
done
log "ALL DONE"
