#!/bin/bash
cd ~/eye-edge
OUT=run009/lat_512 ONNX=artifacts/run007c_eyeD512_fp32.onnx IMG=512 MANIFEST=artifacts/val_manifest_run009.csv REF=artifacts/run007c_eyeD512_torch_cpu_val_probs.npy LABELS=D_eye,G,C,A,H ./run_n100.sh > run009/lat_512.log 2>&1
OUT=run009/lat_224 ONNX=artifacts/run002_mnv3s_fp32.onnx IMG=224 MANIFEST=artifacts/val_manifest.csv REF=artifacts/torch_cpu_val_probs.npy LABELS=D,G,C,A,H ./run_n100.sh > run009/lat_224.log 2>&1
echo ALL_LAT_DONE > run009/lat_all.done
