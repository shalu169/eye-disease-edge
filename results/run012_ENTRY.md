## Run 012 — N100 iGPU: driver upgrade, INT8 collapse diagnosed and fixed

- **Date**: 2026-10-03
- **Purpose**: run 004 found OpenVINO INT8 on the N100 iGPU collapsing to
  mean AUC 0.6226 (CPU INT8 0.805) and attributed it, without evidence, to
  the old Debian OpenCL driver (intel-opencl-icd 22.43). The paper footnote
  repeated that. Test the driver hypothesis, re-measure iGPU latency with a
  current driver, find the real cause, and fix it.
- **Driver setup (no sudo, nothing system-wide changed)**: Intel
  compute-runtime 26.35.39758.10 (2026-09-17) with IGC 2.41.5 and gmmlib
  22.10.0, official .debs downloaded from GitHub, all five sha256 sums
  matched Intel's published sums. Unpacked with `dpkg-deb -x` into
  `~/neo/root` on the N100. A private ICD file (`~/neo/vendors/intel-neo.icd`)
  plus `OCL_ICD_VENDORS` and `LD_LIBRARY_PATH` (`~/neo/env.sh`) make the
  ocl-icd loader use it. Each result JSON records the libigdrcl.so path
  actually mapped into the process, so old and new runs are verified to use
  different drivers. With the new driver OpenVINO reports the device as
  "Intel(R) UHD Graphics" instead of the raw id "Intel(R) Graphics [0x46d1]".
  Old Debian stack: intel-opencl-icd 22.43.24595.41, libigc1
  1.0.12504.6 (bookworm), libigdgmm12 22.7.2.
- **Env**: as run 004/009 (OpenVINO 2026.4.1, NNCF 3.4.0, ONNX Runtime
  1.30.0, Python 3.13.5). `bench_n100.py` gained an optional
  `OV_GPU_PRECISION` override (`INFERENCE_PRECISION_HINT`) and logs the GPU
  name, inference precision and driver path. Defaults unchanged.
- **Method**:
  1. Same protocol as run 004 (224 px run002 model, 1270-image validation
     set, batch 1, 50 warm-up + 500 timed) and run 009 (512 px final model,
     1025-image test set) for the iGPU configs under old and new driver, each
     at the plugin default (f16) and forced f32 inference precision.
  2. Re-quantize with NNCF variants (MIXED/PERFORMANCE preset × target
     device ANY/GPU), same 300 calibration images as run 004, each on CPU and
     GPU (`results/run012_int8_variants.py`).
  3. Layer-wise localisation (`results/run012_layer_diff.py`): every
     Convolution/GroupConvolution/Add/Multiply/activation/ReduceMean/MatMul
     output exposed as a model output; CPU vs GPU at f32 on 3 images,
     relative error per node. FP32 ONNX as control.
  4. NNCF workarounds targeted at the localised layer type.
- **Results**:

  Driver and precision (224 px on validation, 512 px on test; latency
  model-only mean ms):

  | config | old driver 22.43: mean AUC / latency | new driver 26.35: mean AUC / latency |
  |---|---|---|
  | 224 FP32 iGPU, default f16 | 0.8215 / 3.02 | 0.8215 / 2.75 |
  | 224 FP32 iGPU, f32 | 0.8214 / 3.62 | 0.8214 / 3.24 |
  | 224 INT8 iGPU, default | 0.6226 / 3.25 | 0.6226 / 3.06 |
  | 224 INT8 iGPU, f32 | 0.6124 / 3.48 | 0.6124 / 3.08 |
  | 512 FP32 iGPU, default f16 (test) | 0.8546 / 6.94 | 0.8546 / 7.04 |
  | 512 FP32 iGPU, f32 (test) | 0.8545 / 9.70 | 0.8545 / 9.17 |

  At f32 the iGPU matches the PyTorch reference (max |Δp| 7.7e-6 at 224,
  1.2e-5 at 512); the 0.06 probability shift seen in runs 004/009 is the
  plugin's default f16 precision. CPU INT8 re-check: 0.8050, 1.94 ms
  (= run 004).

  NNCF variants, new driver (mean val AUC / latency ms):

  | variant | CPU | iGPU |
  |---|---|---|
  | MIXED, target ANY (= run 004) | 0.8050 / 1.94 | 0.6226 / 2.99 |
  | MIXED, target GPU | 0.8050 / 1.93 | 0.6226 / 3.01 |
  | PERFORMANCE, target ANY | 0.7831 / 1.92 | 0.6078 / 2.93 |
  | PERFORMANCE, target GPU | 0.7831 / 1.93 | 0.6078 / 2.88 |
  | per-tensor activations | 0.5969 / 1.91 | 0.5238 / 2.88 |
  | first depthwise conv kept in float | 0.8038 / 1.92 | 0.7382 / 2.99 |
  | **all depthwise convs kept in float** | **0.8043 / 1.86** | **0.8037 / 2.83** |

  Localisation: in the INT8 IR, CPU and GPU agree to <1e-7 through the
  stem and its hard-swish. The first divergence is the first depthwise
  convolution (`blocks.0.0.conv_dw`, 3×3, stride 2, 16 channels). On the GPU
  its output contains NaN in exactly 2 of 16 channels (channels 14 and 15,
  all 56×56 positions, at f16 and f32 alike), on every image tested; the CPU
  output is finite. The NaNs propagate to the classifier (max relative error
  2.05 downstream). The NaN channels are not the ones with degenerate
  quantization ranges (channels 3, 4, 12 have [0, ≤0.29]), which points to
  the GPU plugin's INT8 depthwise kernel rather than the quantization
  parameters. FP32 control: no node above 5 % except the fused classifier
  MatMul, an artefact of exposing an output inside a fused op.
- **Minimal reproduction (added 2026-10-03, `results/run012_min_repro.py`,
  `results/n100_run012/min_repro.json`)**: the failing layer rebuilt in
  isolation with the original constants (activation FakeQuantize ->
  GroupConvolution 3x3 with weight FakeQuantize), CPU vs GPU at f32, new driver.

  | variant (one change at a time) | GPU NaN channels | rel. error on finite values |
  |---|---|---|
  | as produced by NNCF (weight ranges shape [16,1,1,1,3]) | 14, 15 | 4.7e-2 |
  | weight ranges collapsed to per-channel [16,1,1,1,1] | none | 0 |
  | weight ranges per-tensor | none | 0 |
  | weight FakeQuantize removed | none | 0 |
  | activation FakeQuantize removed | 14, 15 | 4.7e-2 |
  | stride 1 instead of 2 | 14, 15 | 4.7e-2 |
  | layer tiled to 32 channels | 30, 31 | 4.4e-2 |

  **The trigger is the weight FakeQuantize range shape.** NNCF quantized the
  depthwise weights ([16,1,1,3,3] after a Reshape) with ranges of shape
  [16,1,1,1,3]: one range per output channel *and* per kernel column, instead
  of the usual per-output-channel [16,1,1,1,1]. This is valid under
  FakeQuantize's numpy broadcasting, and the CPU plugin computes it
  correctly. The GPU plugin computes it wrongly: every channel is off (about
  5 % relative error) and the last two output channels are NaN, whatever the
  channel count. That pattern is consistent with the GPU kernel indexing the
  48-value range tensor as if it held one value per channel, reading wrong
  values everywhere and past the end for the last channels. This last part
  is an inference from the symptoms, not verified in the OpenVINO source.
  The unusual per-column range shape is itself likely an NNCF quirk with
  5-D group-convolution weights.
- **Plugin fix (added 2026-10-04)**:
  - **Root cause in source**: `src/plugins/intel_gpu/src/graph/graph_optimizer/prepare_quantization.cpp`,
    `prepare_scale_shift_opt`. Its helper `get_offset_safe` addressed the
    FakeQuantize range tensors as `b*p[0] + f*p[1] + y*p[2] + x*p[3]`. For a
    5D (bfzyx) range layout `p[2]` and `p[3]` are the z and y pitches, so x was
    multiplied by the y pitch. The loop also never visited z or w. For ranges
    `[16,1,1,1,3]` the read offset was `3b+3x` instead of `3b+x`, so channels
    14-15 read past the 48-element buffer: NaN, exactly as observed. The code
    is identical in the 2026.4.1 tag and current master.
  - **Patch** (`plugin_fix/ov_quantize_5d_fix.patch`, +113/−30):
    `get_offset_safe` wraps each coordinate to the range tensor's size and
    calls `layout::get_linear_offset`; the loop covers b, f, w, z, y, x. New
    unit test `quantize_gpu.quantize_levels_255_5d_ranges_vary_along_x`.
  - **Build**: OpenVINO tag 2026.4.1 (commit e213a147257, same as the
    installed wheel) plus the patch, built on the N100 (GPU plugin and
    `ov_gpu_unit_tests`; CPU/NPU plugins, oneDNN, Python and frontends off).
    The patched `libopenvino_intel_gpu_plugin.so` (sha256 b95e2c53…) was
    placed in a private copy of the installed openvino package (`~/ovtest`,
    used via PYTHONPATH). The real environment is unchanged.
  - **Validation**:
    - Standalone reproducer: NaN channels [14, 15] → none, error vs CPU 1.0e-7.
    - FakeQuantize-only tests: all channels wrong → all correct.
    - Full INT8 model (224 px, val): mean AUC 0.6226 → 0.8012 (CPU INT8 0.8050).
    - FP32 on the iGPU unchanged (0.8215).
    - Unit tests: all 109 `quantize_gpu*` tests pass, including the new one.
      Broader filter (quantize/fusings/prepare_quantization; 3,181 tests run
      before the run was stopped as too slow, plus the remaining 364 enabled
      `*quant*` tests): 12 failures.
    - All 12 failures also fail with the unpatched build, so they are
      pre-existing: a dynamic-shape `get_tensor()` exception in several
      dynamic quantized convolution tests, plus one fused conv and one
      pooling test. Likely due to this GPU or the reduced build without oneDNN.
    - The new test fails without the fix (value error at x=2) and passes
      with it. After restoring the patch, the rebuilt plugin is byte-identical
      to the validated one.
  - **Re-measured iGPU rows** (patched plugin, compute-runtime 26.35, idle
    N100, same protocol as runs 004/009; `n100_run012/remeasure/`):

    | config | mean AUC | latency mean / p95 (ms) | preprocess (ms) | peak RSS (MB) |
    |---|---|---|---|---|
    | 224 FP32 iGPU (val) | 0.8215 | 2.74 / 2.78 | 4.34 | 687 |
    | 224 INT8 iGPU (val) | 0.8012 | 3.01 / 3.04 | 4.38 | 711 |
    | 512 FP32 iGPU (test) | 0.8546 | 7.02 / 7.07 | 6.80 | 672 |

    The paper's Tables 4 (latency) and 6 (224 edge benchmark) now use these
    iGPU numbers. The energy table (run 009) was measured with the stock
    plugin and the Debian driver and is labelled as such.
- **Energy re-measured (added 2026-10-04, `n100_run012/energy/`)**: run 009's
  exact protocol (`energy_n100.py`, idle baseline, 3 × 60 s, model-only and
  end-to-end), with compute-runtime 26.35 and the patched plugin package
  (PYTHONPATH=~/ovtest). The GPU runs' runtime strings record the neo driver
  path. RAPL readable throughout. Idle package 2.66 ± 0.03 W (224 pass),
  2.60 ± 0.11 W (512 pass).

  | input | runtime | mode | W | inf/s | mJ/inf | mJ above idle | run 009 mJ/inf (old driver) |
  |---|---|---|---|---|---|---|---|
  | 224 | ORT CPU | model | 11.33 ± 0.17 | 426.5 ± 14.4 | 26.6 ± 0.5 | 20.3 ± 0.3 | 26.4 |
  | 224 | OV CPU | model | 11.31 ± 0.03 | 343.6 ± 1.2 | 32.9 ± 0.0 | 25.2 ± 0.0 | 32.7 |
  | 224 | OV iGPU | model | 8.54 ± 0.04 | 370.2 ± 0.4 | 23.1 ± 0.1 | 15.9 ± 0.1 | 28.0 |
  | 512 | ORT CPU | model | 11.53 ± 0.21 | 96.0 ± 3.8 | 120.2 ± 2.5 | 93.1 ± 1.4 | 120.7 |
  | 512 | OV CPU | model | 11.50 ± 0.01 | 89.4 ± 0.1 | 128.6 ± 0.1 | 99.5 ± 0.1 | 127.7 |
  | 512 | OV iGPU | model | 9.18 ± 0.02 | 143.2 ± 0.1 | 64.1 ± 0.2 | 46.0 ± 0.2 | 69.6 |
  | 224 | ORT CPU | e2e | 9.77 | 137.8 | 70.9 | 51.6 | 69.6 |
  | 224 | OV CPU | e2e | 10.74 | 139.7 | 76.9 | 57.9 | 75.3 |
  | 224 | OV iGPU | e2e | 8.50 | 119.0 | 71.4 | 49.1 | 70.2 |
  | 512 | ORT CPU | e2e | 11.00 | 59.3 | 185.6 | 141.8 | 186.1 |
  | 512 | OV CPU | e2e | 11.09 | 54.0 | 205.3 | 157.2 | 202.9 |
  | 512 | OV iGPU | e2e | 9.27 | 73.0 | 126.9 | 91.3 | 132.6 |

  The new driver lowers model-only iGPU energy by 8-18 % (power 9.3 → 8.5 W
  at 224, 10.1 → 9.2 W at 512). End-to-end iGPU and all CPU figures change by
  less than 5 %. The paper's energy table (Table 7) and the graphical
  abstract now use these numbers.
- **Artifacts**: `results/n100_run012/` (per-config JSON/logs for the driver
  matrix, `int8_variants/results.json`, `int8_variants/results_workarounds.json`,
  `layer_diff_{base,fp32}.{csv,json}`, `setup_and_driver.txt` with the
  driver env, ICD file and package checksums); scripts
  `results/run012_{bench_n100,int8_variants,layer_diff}.py` (working copies
  in `edge/`). Quantized IRs remain on the N100 in
  `~/eye-edge/run012/int8_variants/`.
- **Verdict**:
  - **The driver was not the cause.** The INT8 iGPU collapse is identical
    under Intel's current compute-runtime and under the 2022 Debian driver,
    and identical at f16 and f32 precision. The run 004 attribution and the
    paper footnote were wrong and are corrected.
  - **Cause**: the OpenVINO 2026.4.1 GPU plugin mis-computes a quantized
    depthwise GroupConvolution whose weight FakeQuantize has ranges varying
    along the kernel-width axis (shape [C,1,1,1,3], as NNCF emitted here):
    about 5 % error in every channel and NaN in the last two. The CPU plugin
    is correct on the same graph. Reproduced in a single isolated layer;
    independent of driver, precision, stride, activation quantization, NNCF
    preset and target device. Worth reporting to OpenVINO (GPU plugin) and
    possibly NNCF (range shape).
  - **Fix**: excluding depthwise convolutions from quantization
    (`ignored_scope=IgnoredScope(types=["GroupConvolution"])`) restores iGPU
    INT8 to CPU-INT8 accuracy (0.8037 vs 0.8043). It is still not useful:
    at 2.83 ms it is no faster than the iGPU in f16 (2.75 ms) and slower than
    the CPU in FP32 (2.06 ms), and it keeps INT8's accuracy and calibration
    cost relative to FP32 (−0.018 mean AUC).
  - **iGPU latency is not a driver fault either.** The new driver shaves
    about 9 % at 224 px (3.02 → 2.75 ms) and changes 512 px by +1 % (6.94 →
    7.04 ms) at the default precision. At 224 px the iGPU remains slower than
    the CPU because per-inference dispatch overhead dominates a model this
    small. At 512 px it is the fastest backend, as in run 009.
  - Paper/run 009 numbers measured with the old driver stand: the new
    driver changes no AUC and moves iGPU latency by −9 % to +1 %.
- **Limitations**: one OpenVINO version (2026.4.1); an older OpenVINO was not
  tried, so whether this is a regression is unknown. Layer diff on 3 images.
  Single latency run per config.
