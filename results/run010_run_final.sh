#!/bin/bash
# Run 010 final pipeline on the run 008 final checkpoint (requires
# results/run008_FINAL_CHECKPOINT.json with "updated_after_all_seeds": true).
# Mac, CPU only. Usage (repo root): nohup bash results/run010_run_final.sh > results/run010/log_final_driver.txt 2>&1 &
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PYO=.venv-run010/bin/python
PYT=.claude/worktrees/fundus-app-local-mode/.venv-export/bin/python
TAG=final
L=results/run010
echo "[$(date '+%F %T')] start"
$PY -c "import json; d=json.load(open('results/run008_FINAL_CHECKPOINT.json')); assert d['updated_after_all_seeds'], d; print(d['checkpoint'])"
$PY results/run010_check_reference.py --ckpt final --n 64 > $L/log_check_reference_final.txt 2>&1
$PY results/run010_export_edge.py --ckpt final --tag $TAG > $L/log_export_final.txt 2>&1
$PYT results/run010_export_tflite.py --tag $TAG > $L/log_export_tflite_final.txt 2>&1
$PYO results/run010_parity_edge.py --runtime onnx --tag $TAG --ckpt final --n 300 > $L/log_parity_onnx_final.txt 2>&1
$PYT results/run010_parity_edge.py --runtime tflite --tag $TAG --ckpt final --n 300 > $L/log_parity_tflite_final.txt 2>&1
$PY results/run010_make_device_fixture.py --tag $TAG --ckpt final > $L/log_fixture_final.txt 2>&1
echo "[$(date '+%F %T')] parity done"
$PY results/run010_localisation.py --ckpt final > $L/log_localisation_final.txt 2>&1
$PY results/run010_sanity.py --ckpt final --n 100 $( [ -f results/checkpoints/run010_randlabels_512_final.pt ] && echo --randlabel-ckpt results/checkpoints/run010_randlabels_512_final.pt ) > $L/log_sanity_final.txt 2>&1
echo "[$(date '+%F %T')] sanity done"
$PYO results/run010_faithfulness.py --ckpt final --onnx edge/artifacts/run010_${TAG}_full_fp32.onnx > $L/log_faithfulness_final.txt 2>&1
$PY results/run010_figures.py --ckpt final > $L/log_figures_final.txt 2>&1
echo "[$(date '+%F %T')] ALL DONE"
