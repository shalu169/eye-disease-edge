#!/bin/bash
# usage: phase2.sh <ckpt>
set -e
cd /Users/piuaggawal/Documents/my_workspace/eye-disease-edge-ai
CK=$1; TAG=run008final
.venv/bin/python edge/export_onnx.py --ckpt $CK --img 512 --tag $TAG --manifest edge/artifacts/test_manifest_run008.csv --ref-name test > results/logs/run009_export_onnx_$TAG.log 2>&1
CUDA_VISIBLE_DEVICES= .claude/worktrees/fundus-app-local-mode/.venv-export/bin/python edge/export_tflite.py --onnx edge/artifacts/${TAG}_fp32.onnx --img 512 --tag $TAG --manifest edge/artifacts/test_manifest_run008.csv --ref edge/artifacts/${TAG}_torch_cpu_test_probs.npy --label-cols D_eye,G,C,A,H --variants fp32 > results/logs/run009_export_tflite_$TAG.log 2>&1
cd edge
COPYFILE_DISABLE=1 tar cf - artifacts/${TAG}_fp32.onnx artifacts/${TAG}_torch_cpu_test_probs.npy | ssh n100m "cd ~/eye-edge && tar xf - 2>/dev/null; md5sum artifacts/${TAG}_fp32.onnx; (OUT=run009/test_final ONNX=artifacts/${TAG}_fp32.onnx IMG=512 MANIFEST=artifacts/test_manifest_run008.csv REF=artifacts/${TAG}_torch_cpu_test_probs.npy LABELS=D_eye,G,C,A,H nohup ./run_n100.sh > run009/test_final.log 2>&1 &)"
COPYFILE_DISABLE=1 tar cf - artifacts/${TAG}_fp32.tflite artifacts/${TAG}_torch_cpu_test_probs.npy | ssh pi@192.168.188.108 "cd ~/eye-edge && tar xf - 2>/dev/null; md5sum artifacts/${TAG}_fp32.tflite; (OUT=run009/test_final MODEL=artifacts/${TAG}_fp32.tflite IMG=512 MANIFEST=artifacts/test_manifest_run008.csv REF=artifacts/${TAG}_torch_cpu_test_probs.npy LABELS=D_eye,G,C,A,H CONFIGS=tfl_fp32_t4 nohup ./run_pi.sh > run009/test_final.log 2>&1 &)"
md5 -r artifacts/${TAG}_fp32.onnx artifacts/${TAG}_fp32.tflite
echo PHASE2_LAUNCHED
