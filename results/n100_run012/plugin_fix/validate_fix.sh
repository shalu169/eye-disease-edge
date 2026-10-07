#!/bin/bash
# Run 012 fix validation: original wheel GPU plugin vs patched plugin (same OpenVINO 2026.4.1 tag + patch).
# Patched plugin is swapped into a private copy of the openvino package; the real venv is untouched.
set -u
cd ~/eye-edge
. ~/neo/env.sh
SP=.venv/lib/python3.13/site-packages
rm -rf ~/ovtest && mkdir -p ~/ovtest && cp -r $SP/openvino ~/ovtest/openvino
cp ~/ovbuild/src/bin/intel64/Release/libopenvino_intel_gpu_plugin.so ~/ovtest/openvino/libs/libopenvino_intel_gpu_plugin.so
sha256sum $SP/openvino/libs/libopenvino_intel_gpu_plugin.so ~/ovtest/openvino/libs/libopenvino_intel_gpu_plugin.so
mkdir -p run012/fix
for v in original patched; do
  if [ $v = patched ]; then export PYTHONPATH=~/ovtest; else unset PYTHONPATH; fi
  echo "===== $v plugin"
  .venv/bin/python -c "import openvino,os;print(openvino.__file__)"
  echo "-- standalone reproducer"; .venv/bin/python run012/ov_gpu_groupconv_fq_repro.py 2>&1 | grep -vi warn
  echo "-- FakeQuantize alone"; .venv/bin/python run012/ov_fq_only.py 2>&1 | grep -vi warn
  echo "-- full model (224 px, val 1270)"
  for cfg in ov_int8_gpu ov_fp32_gpu; do
    .venv/bin/python bench_n100.py run $cfg --out-dir run012/fix/${v}_$cfg > run012/fix/${v}_$cfg.log 2>&1
    python3 -c "import json,glob;r=json.load(open(glob.glob('run012/fix/${v}_$cfg/results_*.json')[0]));print('$cfg', 'mean_auc', r['mean_auc'], 'D', r['val_auc']['D'], 'maxdiff %.2e'%r['max_abs_prob_diff_vs_torch'], 'lat', r['latency_ms']['mean'])"
  done
done
echo VALIDATE_DONE
