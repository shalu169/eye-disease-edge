#!/bin/bash
# Run 008 driver: A (512 eye-D) x3 seeds -> B (224 patient-D) x3 -> C (512 patient-D) x3.
# Writes run008_FINAL_CHECKPOINT.json after A s0 and updates it after A s2.
# Usage (repo root): nohup bash results/run008_run_all.sh > results/logs/run008_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() {  # tag img dlabel seed
  echo "[$(date '+%F %T')] start $1"
  $PY results/run008_train.py --tag "$1" --img "$2" --dlabel "$3" --seed "$4" --workers 8 > "results/logs/$1.log" 2>&1
  rc=$?; echo "[$(date '+%F %T')] end $1 rc=$rc"; return $rc
}
for s in 0 1 2; do
  run run008A_joint5_eyeD_512_s$s 512 eye $s || exit 1
  if [ $s -eq 0 ]; then $PY results/run008_final_ckpt.py --mode first > results/logs/run008_final_ckpt_first.log 2>&1; echo "wrote FINAL_CHECKPOINT (first)"; fi
done
$PY results/run008_final_ckpt.py --mode all > results/logs/run008_final_ckpt_all.log 2>&1; echo "wrote FINAL_CHECKPOINT (all seeds)"
for s in 0 1 2; do run run008B_joint5_patientD_224_s$s 224 patient $s || exit 1; done
for s in 0 1 2; do run run008C_joint5_patientD_512_s$s 512 patient $s || exit 1; done
echo "[$(date '+%F %T')] ALL DONE"
