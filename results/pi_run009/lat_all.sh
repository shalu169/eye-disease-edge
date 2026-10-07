#!/bin/bash
cd ~/eye-edge
OUT=run009/lat_512 MODEL=artifacts/run007c_eyeD512_fp32.tflite IMG=512 MANIFEST=artifacts/val_manifest_run009.csv REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy LABELS=D_eye,G,C,A,H N=300 CONFIGS="tfl_fp32_t4 tfl_fp32_t1" ./run_pi.sh > run009/lat_512.log 2>&1
OUT=run009/lat_224 MODEL=artifacts/run002_mnv3s_fp32.tflite IMG=224 MANIFEST=artifacts/val_manifest.csv REF=artifacts/torch_cpu_val_probs.npy LABELS=D,G,C,A,H N=300 CONFIGS="tfl_fp32_t4 tfl_fp32_t1" ./run_pi.sh > run009/lat_224.log 2>&1
echo done > run009/lat_all.done
