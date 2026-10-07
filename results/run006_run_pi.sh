#!/bin/bash
# Run 006 driver on the Pi: one process per config, sequential, 60 s idle gap
# between configs so thermal state is comparable. Run under nohup from
# ~/eye-edge:   nohup ./run_pi.sh > suite.log 2>&1 &
cd "$(dirname "$0")"
CONFIGS="${CONFIGS:-tfl_fp32_t4 tfl_fp32_t1 tfl_drq_t4 tfl_drq_t1 tfl_int8_t4 tfl_int8_t1}"
echo "suite start $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
echo "undervoltage dmesg lines at start: $(dmesg 2>/dev/null | grep -ci undervoltage)"
for c in $CONFIGS; do
  echo "=== $c start $(date -Is)"
  /usr/bin/time -v .venv/bin/python bench_pi.py "$c" > "log_$c.txt" 2>&1
  echo "=== $c exit $? $(date -Is)"
  sleep 60
done
echo "undervoltage dmesg lines at end: $(dmesg 2>/dev/null | grep -ci undervoltage)"
echo "suite end $(date -Is) | $(vcgencmd measure_temp) $(vcgencmd get_throttled) $(vcgencmd measure_clock arm)"
echo SUITE_DONE
