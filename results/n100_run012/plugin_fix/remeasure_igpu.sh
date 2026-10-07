#!/bin/bash
# Run 012b: re-measure N100 iGPU rows for the paper with the patched GPU plugin (2026.4.1 + quantize fix)
# and Intel compute-runtime 26.35. Same protocol as runs 004/009. Run only on an idle machine.
cd ~/eye-edge
. ~/neo/env.sh
export PYTHONPATH=~/ovtest
uptime
mkdir -p run012/remeasure
R512="--onnx artifacts/run008final_fp32.onnx --img 512 --manifest artifacts/test_manifest_run008.csv --ref artifacts/run008final_torch_cpu_test_probs.npy --label-cols D_eye,G,C,A,H"
for cfg in ov_fp32_gpu ov_int8_gpu; do
  .venv/bin/python bench_n100.py run $cfg --out-dir run012/remeasure/224_$cfg > run012/remeasure/224_$cfg.log 2>&1; echo "224 $cfg rc=$?"
done
.venv/bin/python bench_n100.py run ov_fp32_gpu --out-dir run012/remeasure/512_ov_fp32_gpu $R512 > run012/remeasure/512_ov_fp32_gpu.log 2>&1; echo "512 ov_fp32_gpu rc=$?"
uptime
echo REMEASURE_DONE
