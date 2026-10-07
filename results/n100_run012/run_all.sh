#!/bin/bash
# Run 012: iGPU driver fix. Old Debian OpenCL driver 22.43 vs Intel compute-runtime 26.35 (user-space, ~/neo).
cd ~/eye-edge
R512="--onnx artifacts/run008final_fp32.onnx --img 512 --manifest artifacts/test_manifest_run008.csv --ref artifacts/run008final_torch_cpu_test_probs.npy --label-cols D_eye,G,C,A,H"
run() { # name driver precision config extra...
  name=$1; drv=$2; prec=$3; cfg=$4; shift 4
  ( [ "$drv" = new ] && . ~/neo/env.sh; [ "$prec" != default ] && export OV_GPU_PRECISION=$prec
    .venv/bin/python bench_n100.py run $cfg --out-dir run012/$name "$@" > run012/$name.log 2>&1 )
  echo "$(date +%T) $name rc=$?"
}
for drv in old new; do
  run ${drv}_224_fp32_gpu        $drv default ov_fp32_gpu
  run ${drv}_224_int8_gpu        $drv default ov_int8_gpu
  run ${drv}_224_int8_gpu_f32    $drv f32     ov_int8_gpu
  run ${drv}_224_fp32_gpu_f32    $drv f32     ov_fp32_gpu
  run ${drv}_512_fp32_gpu        $drv default ov_fp32_gpu $R512
  run ${drv}_512_fp32_gpu_f32    $drv f32     ov_fp32_gpu $R512
done
run new_224_int8_cpu_check new default ov_int8_cpu
echo ALL_DONE
