#!/bin/bash
cd ~/eye-edge
. ~/neo/env.sh
export PYTHONPATH=~/ovtest
.venv/bin/python -c "import openvino;print("openvino from", openvino.__file__)"
for f in /sys/class/powercap/intel-rapl:0/energy_uj /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj /sys/class/powercap/intel-rapl:0/intel-rapl:0:1/energy_uj; do
  cat $f > /dev/null || { echo "RAPL NOT READABLE: $f $(date -Is)"; exit 1; }
done
echo "RAPL readable $(date -Is)"; ps aux --sort=-%cpu | head -5; uptime
CF=ort_fp32_cpu,ov_fp32_cpu,ov_fp32_gpu
ENERGY_OUT=run012/energy_512 .venv/bin/python energy_n100.py all --configs $CF --onnx artifacts/run007c_eyeD512_fp32.onnx --img 512 --manifest artifacts/val_manifest_run009.csv --ref artifacts/run007c_eyeD512_torch_cpu_val_probs.npy --label-cols D_eye,G,C,A,H > run012/energy_512.log 2>&1
echo "512 done rc=$? $(date -Is)"
ENERGY_OUT=run012/energy_224 .venv/bin/python energy_n100.py all --configs $CF --onnx artifacts/run002_mnv3s_fp32.onnx --img 224 --manifest artifacts/val_manifest.csv --ref artifacts/torch_cpu_val_probs.npy --label-cols D,G,C,A,H > run012/energy_224.log 2>&1
echo "224 done rc=$? $(date -Is)"
echo ENERGY_ALL_DONE
