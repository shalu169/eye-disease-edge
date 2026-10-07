#!/bin/bash
cd ~/eye-edge
OUT=run009/lat_384 MODEL=artifacts/run007b_eyeD384_fp32.tflite IMG=384 MANIFEST=artifacts/val_manifest_run009.csv REF=artifacts/run007b_eyeD384_torch_cpu_val_probs.npy LABELS=D_eye,G,C,A,H N=300 CONFIGS="tfl_fp32_t4 tfl_fp32_t1" ./run_pi.sh > run009/lat_384.log 2>&1
echo done > run009/lat_384.done
