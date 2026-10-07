<!-- DRAFT PR for openvinotoolkit/openvino, from a fork, base branch master. Not opened.
     Title: -->
**Title:** [GPU] Fix FakeQuantize range offsets for 5D/6D ranges in prepare_scale_shift_opt

### Details:
 - **Bug:** `prepare_quantization::prepare_scale_shift_opt` precomputes the scale/shift buffers of a `quantize` node from its constant input/output ranges. Its helper `get_offset_safe` addressed the range tensors with a hard-coded 4D formula: `b*pitches[0] + f*pitches[1] + y*pitches[2] + x*pitches[3]`. For a 5D (bfzyx) range tensor, `pitches[2]` and `pitches[3]` are the z and y pitches, so x was multiplied by the y pitch. In addition, the surrounding loop never iterated over z or w.
 - **Effect:** for ranges that vary along x of a 5D tensor, e.g. weight FakeQuantize ranges of shape `[C,1,1,1,KW]` on a depthwise GroupConvolution (what NNCF emits for ONNX-imported depthwise convolutions), every channel read the wrong ranges and the last channels read past the end of the buffer.
   - With `[16,1,1,1,3]` ranges the computed offset is `3b+3x` instead of `3b+x`, so channels 14 and 15 read beyond the 48-element buffer and produce NaN.
   - In an NNCF-quantized MobileNetV3-small this collapsed GPU INT8 accuracy (mean AUC 0.62 vs 0.805 on CPU for the same IR).
 - **Fix:**
   - `get_offset_safe` wraps each coordinate to the range tensor's own size (broadcasting) and delegates to `layout::get_linear_offset`, which is format- and rank-aware.
   - The loop now iterates b, f, w, z, y, x, building the index with `format::bfwzyx`.
   - 4D behaviour is unchanged.
 - **Test:** added `quantize_gpu.quantize_levels_255_5d_ranges_vary_along_x`, a 5D bfzyx quantize with constant ranges varying per batch and per x, compared against a reference FakeQuantize computed in the test.

### Validation (Intel N100 iGPU, Debian 13, Intel compute-runtime 26.35; OpenVINO 2026.4.1 tag + this patch; the changed code is identical on master):
 - `ov_gpu_unit_tests --gtest_filter=quantize_gpu*`: 109/109 pass. The new test fails without the fix (wrong value at x=2) and passes with it.
 - Broader `*quantize*:*fusings*:*prepare_quantization*` / `*quant*` subset: 12 failures. All 12 fail identically on the unpatched build, so they are pre-existing in this environment. They are dynamic-shape `get_tensor()` exceptions in `conv_dyn_*quantized*` and `quantized_convolution_*_dynamic`, plus `conv_fp32_multi_eltwise_quantization.basic/1` and `pooling_f32_scale.fp16_scale_out/1`. The build has oneDNN disabled.
 - Standalone Python reproducer (linked issue), original vs patched plugin: NaN in channels 14-15 → no NaN, relative error vs CPU 1.0e-7.
 - FakeQuantize alone on a 5D tensor with `[C,1,1,1,KW]` ranges: all channels wrong → all correct.
 - Full NNCF INT8 MobileNetV3-small on GPU, mean AUC on 1270 validation images: 0.6226 → 0.8012 (CPU: 0.8050).
 - FP32 model on GPU: unchanged (0.8215 before and after).

### Tickets:
 - Closes #<issue number>

### AI Assistance:
 - AI assistance used: yes
 - How AI was used: an AI assistant (Claude, Anthropic) helped localise the bug (CPU/GPU layer-by-layer comparison, single-layer reproducers), draft the fix and the unit test, and write this description.
 - Human validation performed: <AUTHOR TO COMPLETE: who reviewed the change and confirms understanding of it>. The patched plugin was built from the 2026.4.1 tag plus this patch and checked on real hardware: unit tests, the Python reproducer, and the full INT8 model accuracy on the GPU, with results as listed above.
