## Run 010 — Grad-CAM: reference vs edge implementation, sanity checks, faithfulness, lesion localisation, on-device cost

- **Date**: 2026-10-03
- **Status**: DONE on the final model. The development pass on the run007c
  checkpoint is kept in `results/run010/tables_dev.md` and the `*_dev*` files.
  It gave the same qualitative picture, except that faithfulness against the
  centre-bias baseline was weaker (see Verdict).
- **Purpose**: the paper claims an on-device Grad-CAM path that needs no
  automatic differentiation on the device. Literature (Adebayo 2018, Arun 2021,
  Saporta 2022, Sayres 2019, Ghassemi 2021) warns that saliency maps can be
  independent of the model, unfaithful, and poor at small lesions. This run
  tests (1) that the edge implementation is numerically the same Grad-CAM as
  the PyTorch reference, (2) Adebayo's randomisation sanity checks, (3)
  deletion/insertion faithfulness against random and centre-bias baselines,
  (4) localisation against expert pixel-level DR lesion masks (IDRiD), and
  (5) its latency/memory cost on the N100 and the Pi 3B.
- **Model**: the run 008 final checkpoint from `results/run008_FINAL_CHECKPOINT.json`
  (`updated_after_all_seeds: true`, median-validation seed):
  `results/checkpoints/run008A_joint5_eyeD_512_s2_best_epoch21.pt` (md5
  `c94c9a7b1067e18b8aaeec64b61953d3`). It is a 512 px joint 5-label model with
  per-eye DR, and it never saw the run 008 TEST split. The full pipeline ran
  unattended via `results/run010_run_final.sh` (12:59-15:27 CEST, Mac CPU only).
  Development work used `run007c_joint5_eyeD_512_s1_best_epoch22.pt`, whose
  training set contains the TEST images, so its test-set numbers are not
  reported as results.
- **Grad-CAM definition** (`results/run010_common.py`): target layer = output
  of `forward_features` (= `blocks[5]`, 1x1 ConvBnAct 96→576, 16x16 at 512 px),
  i.e. the last conv feature map before GAP. Per label c: weights = spatial mean
  of d logit_c / dA (pre-sigmoid logit), map = ReLU(Σ_k w_k A_k), bilinear
  upsample to 512 (align_corners=False), divided by the per-map maximum for
  display. All metrics are invariant to that scaling.
- **Edge path** (`edge/gradcam_numpy.py`): the backbone (ONNX or TFLite) outputs
  only the 576x16x16 feature map. NumPy computes GAP → conv_head (1x1 = dense
  1024x576) → hardswish → classifier (5x1024) → sigmoid, and the Grad-CAM weights
  analytically: d logit_c/d pooled = (W2[c] ⊙ hardswish'(z)) W1, divided by HW
  (the gradient is the same at every position because GAP is linear). Bilinear
  upsampling uses two interpolation matrices that reproduce torch exactly. The
  idea is ported from the Flutter app's `lib/inference/gradcam.dart` in the
  `fundus-app-local-mode` worktree (read only; nothing there was modified).
  Exports: `results/run010_export_edge.py` (torch 2.14 ONNX opset 17: backbone,
  full model, head weights `.npz`) and `results/run010_export_tflite.py`
  (onnx2tf 1.26.3 + TF 2.18.0 from the worktree's `.venv-export`, read only,
  CPU). Final artifacts: `edge/artifacts/run010_final_backbone_fp32.onnx` (3.7 MB,
  md5 `2f1573abbfef08f3f5a17beed9e1ecc3`), `run010_final_full_fp32.onnx` (6.1 MB,
  `ccb496758103a0ec337ff04c103030a2`), `run010_final_backbone_fp32.tflite`
  (3.7 MB, `b448d16361063b27e1fda50b4b6387c3`, output NHWC [1,16,16,576]),
  `run010_final_full_fp32.tflite` (6.1 MB, `aedea11f938de6d1c6075184403c8c1a`),
  `run010_final_head.npz` (`9829700c114722c134275d3eb3555d06`). **Note:** onnx2tf
  rewrites its input ONNX in place (the simplified graph), so the md5s in
  `run010_final_export.json` (taken right after the torch export) differ from the
  files on disk. Every ONNX number in this entry (Mac parity, faithfulness, N100)
  was measured on the rewritten files listed here, and they pass parity.
- **Environments (Mac, CPU only; MPS not used)**: `.venv` (Python 3.14,
  torch 2.14.0, timm 1.0.30, grad-cam 1.5.7, numpy 2.5.3, scipy 1.18.1,
  scikit-learn 1.9.1, matplotlib 3.11.2, opencv 5.0.0) and a new
  `.venv-run010` (Python 3.12.14, onnxruntime 1.30.0 = the N100 version,
  numpy 1.26.4 = the Pi version, torch 2.14.1, timm 1.0.30; freeze in
  `results/run010/env_venv_run010_freeze.txt`). onnxruntime was not in `.venv`,
  and the shared training venv was left untouched.
- **IDRiD provenance**: Indian Diabetic Retinopathy Image Dataset (Porwal et al.
  2018, Data 3(3):25, doi 10.3390/data3030025), "A. Segmentation" part: 81
  images (54 train + 27 test, 4288x2848) with pixel masks for microaneurysms (81
  images), haemorrhages (80), hard exudates (81), soft exudates (40) and the
  optic disc (81). Source: Kaggle mirror
  `aaryapatel98/indian-diabetic-retinopathy-image-dataset` (dataset id 472549,
  uploaded 2020-01-11, 7967 downloads), downloaded 2026-10-03 with the
  kaggle CLI and the existing token. Zip sha256
  `68c20658a41c695828377bfc347b97b0d9d658f0dd1ce72609d4d0cec7330059`
  (`data/idrid/zip.sha256`, Kaggle metadata in
  `data/idrid/kaggle_dataset-metadata.json`). **Licence: CC BY 4.0.** The zip
  contains the original `LICENSE.txt` ("IDRiD dataset (c) by Prasanna Porwal,
  Samiksha Pachade and Manesh Kokare … licensed under a Creative Commons
  Attribution 4.0 International License") and the full CC-BY-4.0 text, and the
  Kaggle page lists the same licence. No login-gated source was used. Cite
  porwal2018indian, not the mirror.
- **IDRiD preprocessing** (`results/run010_idrid_prep.py`, ODIR-like): FOV =
  grey > 15. The disc is cut at the top and bottom in every IDRiD image (81/81),
  so the square crop is taken around the disc: side = horizontal FOV extent
  (3410-3802 px), vertical centre = mean row of the widest FOV rows. The parts
  outside the image are padded black (like ODIR's inscribed-circle crop, but
  with black bands where IDRiD's disc is truncated). The image is then resized to
  512 (PIL bilinear). Masks get the identical crop/pad. Each output pixel's
  lesion fraction is computed with the PIL BOX filter, and the pixel counts as
  lesion if that fraction is > 0, so microaneurysms (~7x downscale) are kept.
  Mean mask area in the 512 FOV: MA 0.27 %, HE 1.74 %, EX 1.89 %, SE 0.62 %. The
  mask alignment was checked visually (`paper/figs/run010_idrid_overlays.png`
  top row).
- **Methods**:
  - *Reference cross-check* (`results/run010_check_reference.py`): 64 seeded test
    images × 5 labels against `pytorch_grad_cam.GradCAM(target_layers=[model.blocks[-1]])`.
  - *Parity* (`results/run010_parity_edge.py`): 300 seeded test images × 5
    labels, ONNX Runtime 1.30.0 and TFLite (TF 2.18 interpreter), FP32. The
    NumPy head is also run on the torch feature map, which isolates the head
    maths.
  - *Sanity checks* (`results/run010_sanity.py`): 100 seeded test images × 5
    labels. Cascading re-initialisation from the top (classifier → conv_head →
    blocks.5 … blocks.0 → stem) to the architecture's own init: weights of a
    fresh `timm.create_model(..., pretrained=False)`, seed 0, BN reset to
    γ=1, β=0, μ=0, σ²=1. Independent re-initialisation of blocks.5 only.
    Similarity: Spearman ρ (128x128 subsample) and SSIM (σ=1.5) of the 512 maps.
    Reference levels: the same label's map of a different image (similarity
    from generic fundus layout alone) and the centre-bias map.
    *Data randomisation* (model trained on permuted labels,
    `results/run010_train_randlabels.py`, written but **not run**): at 512 px it
    needs about 30 min on MPS, which run 008 B/C were using throughout, or about
    11 h on CPU (measured 0.32 s/image per training step).
  - *Faithfulness* (`results/run010_faithfulness.py`): every positive TEST image
    per label (D = per-eye DR, 285; G 62; C 62; A 42; H 34; 485 pairs on 448
    images). Pixels are ranked by the saliency map. Deletion/insertion is done at
    33 fractions (0, 1/32 … 1), with blur fill (Gaussian σ=10 px, primary) and
    ImageNet-mean fill. The target label's sigmoid probability is tracked, and the
    AUC is a trapezoid over the fraction. Baselines: smooth random (U(0,1) on
    16x16, upsampled; 1 draw per image) and centre bias (Gaussian, σ=0.25·512).
    95 % bootstrap CIs (2000 resamples over images) and paired Wilcoxon tests.
    Perturbed images were scored with the exported full-model ONNX in ORT
    (checked against torch, max logit diff < 1e-3, asserted). It is about 4x
    faster than torch CPU.
  - *Localisation* (`results/run010_localisation.py`): DR Grad-CAM on the 81
    IDRiD images, scored inside the FOV. Pointing game, strict and with ±15 px
    tolerance. Energy pointing game (EPG = share of the map's mass inside
    lesions; chance = lesion area fraction). Pixel AUROC and AP (every 2nd
    row/column). Baselines: smooth random (mean over 50 draws per image), centre
    bias, and the *glaucoma* Grad-CAM as a wrong-class control. Results are
    reported for the union of the four lesion types and for each type.
  - *Figures* (`results/run010_figures.py`): selection is a seeded uniform random
    draw within each category. The operating threshold per label = 90 %
    sensitivity on VALIDATION (`run008_common.thr_at_sens`). Selections are
    logged in `results/run010/figure_selection_final.json` and
    `figure_idrid_selection_final.json`.
  - *On-device cost* (`edge/gradcam_bench.py`, driver `edge/run010_device.sh`,
    bundle `edge/artifacts/run010_bundle_final/` sent over ssh with tar,
    `COPYFILE_DISABLE=1`, md5s checked on arrival). This ran only after
    `results/run009_ENTRY.md` existed. One process per config:
    `plain` = full model; `head` = backbone + NumPy head (probabilities only);
    `gradcam1` = + Grad-CAM for DR (512 px, normalised); `gradcam5` = all 5
    labels. Timing is batch 1 on one fixed preprocessed image, excluding JPEG
    decode/resize, which run 009 measures. Peak RSS = `ru_maxrss` (Pi
    cross-checked with `/usr/bin/time -v`). Before timing, each config checks
    its output against the Mac PyTorch-autograd probabilities and maps of 3
    test images in `run010_final_fixture.npz` (maps stored as float16).
    N100 (`n100m`, ~/eye-edge venv: Python 3.13.5, ORT 1.30.0, OpenVINO 2026.4.1,
    numpy 2.4.6; governor performance): 50 warmup + 500 timed runs. Runtimes:
    ORT default, ORT with `session.intra_op.allow_spinning=0`, and OpenVINO CPU
    (`PERFORMANCE_HINT=LATENCY`). Pi 3B (~/eye-edge venv: Python 3.9.2,
    tflite-runtime 2.13.0, numpy 1.26.4): 4 threads, 10 warmup + 100 timed runs.
    `get_throttled`, clock and temperature were recorded before/after each
    config, and the 30 s idle gap between configs is the same on both devices.
    Nothing was installed on either device.

- **Results (final model; every table generated by `results/run010_tables.py` from the JSONs)**:

  ### Reference Grad-CAM vs pytorch-grad-cam
  
  64 test images x 5 labels = 320 maps. Low-res (16x16, min-max) max |diff| 1.5e-06; 512 px library-style post-processing max |diff| 1.4e-06; our display map vs library output max |diff| 1.5e-06, min Spearman 0.9999999999. Grad-CAM forward vs plain forward prob max |diff| 1.2e-06.
  
  ### Edge Grad-CAM parity vs PyTorch autograd (Mac)
  
  | runtime | images / maps | feature map max abs diff | prob max diff, plain full model | prob max diff, backbone + NumPy head | prob max diff, NumPy head on torch features | heatmap max abs diff | mean of per-map max diff | heatmap max diff, NumPy head on torch features | min Spearman | argmax identical |
  |---|---|---|---|---|---|---|---|---|---|---|
  | onnx (1.30.0) | 300 / 1500 | 1.4e-03 (1.8e-05 rel) | 1.1e-05 | 1.1e-05 | 3.9e-07 | 2.3e-04 | 8.4e-06 | 1.4e-05 | 0.99999997 | 0.999 |
  | tflite (tensorflow 2.18.0) | 300 / 1500 | 3.9e-03 (5.4e-05 rel) | 6.8e-05 | 6.8e-05 | 3.9e-07 | 7.2e-04 | 2.9e-05 | 1.3e-05 | 0.99983390 | 0.999 |
  
  ### Sanity checks (Adebayo 2018): Spearman rho / SSIM of randomised vs original map
  
  100 test images, all 5 labels. Median over images; per-label Spearman medians.
  
  | condition | mean abs Δp | Spearman (all) | SSIM (all) | D | G | C | A | H | all-zero maps |
  |---|---|---|---|---|---|---|---|---|---|
  | cascade → classifier | 0.322 | 0.075 | 0.483 | 0.15 | 0.04 | 0.04 | 0.12 | 0.10 | 0 |
  | cascade → conv_head | 0.644 | 0.022 | 0.406 | -0.07 | -0.15 | 0.13 | 0.14 | 0.05 | 0 |
  | cascade → blocks.5 | 0.704 | -0.038 | 0.335 | -0.29 | -0.10 | 0.08 | 0.06 | 0.06 | 0 |
  | cascade → blocks.4 | 0.401 | -0.002 | 0.308 | 0.27 | -0.06 | -0.04 | -0.07 | 0.01 | 0 |
  | cascade → blocks.3 | 0.359 | -0.013 | 0.374 | -0.04 | -0.02 | -0.05 | 0.02 | 0.01 | 0 |
  | cascade → blocks.2 | 0.413 | 0.002 | 0.390 | 0.07 | -0.07 | -0.07 | 0.04 | 0.03 | 0 |
  | cascade → blocks.1 | 0.754 | 0.009 | 0.215 | 0.28 | -0.08 | -0.07 | 0.10 | -0.05 | 0 |
  | cascade → blocks.0 | 0.431 | 0.007 | 0.266 | 0.10 | -0.12 | -0.05 | 0.14 | 0.00 | 0 |
  | cascade → stem | 0.424 | -0.045 | 0.322 | -0.11 | -0.07 | -0.00 | -0.03 | -0.02 | 0 |
  | independent: blocks.5 only | 0.377 | 0.067 | 0.432 | -0.08 | 0.11 | 0.13 | 0.12 | 0.05 | 0 |
  | reference: different image, same label | – | 0.144 | 0.531 | 0.21 | 0.16 | 0.21 | 0.07 | 0.09 | 0 |
  | reference: centre-bias map | – | 0.176 | 0.139 | 0.39 | 0.22 | 0.05 | 0.14 | 0.12 | 0 |
  
  Class specificity (same image, other label's map), Spearman median: D_vs_G -0.006, D_vs_C 0.104, D_vs_A 0.168, D_vs_H 0.124; all label pairs median 0.087 [IQR -0.085, 0.220].
  
  ### Deletion / insertion AUC, blur fill (test positives; deletion lower = better, insertion higher = better)
  
  | label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | D | 285 | 0.69 | 0.331 | 0.470 | 0.294 | -0.139 [-0.156, -0.122] (p=7.0e-36, win 0.85) | +0.037 [0.021, 0.051] (p=6.0e-07, win 0.39) | 0.769 | 0.474 | 0.606 | +0.295 [0.281, 0.310] (p=1.7e-48, win 1.00) | +0.163 [0.151, 0.176] (p=4.4e-48, win 0.98) |
  | G | 62 | 0.72 | 0.238 | 0.522 | 0.270 | -0.284 [-0.329, -0.240] (p=1.9e-11, win 0.94) | -0.031 [-0.077, 0.014] (p=5.1e-01, win 0.48) | 0.843 | 0.532 | 0.625 | +0.312 [0.267, 0.357] (p=7.6e-12, win 1.00) | +0.218 [0.182, 0.258] (p=7.6e-12, win 1.00) |
  | C | 62 | 0.73 | 0.737 | 0.754 | 0.906 | -0.018 [-0.046, 0.009] (p=2.9e-01, win 0.56) | -0.170 [-0.215, -0.123] (p=7.6e-12, win 1.00) | 0.877 | 0.755 | 0.813 | +0.122 [0.086, 0.160] (p=7.3e-10, win 0.95) | +0.064 [0.030, 0.100] (p=3.5e-02, win 0.55) |
  | A | 42 | 0.67 | 0.041 | 0.187 | 0.112 | -0.147 [-0.190, -0.108] (p=4.5e-13, win 1.00) | -0.071 [-0.095, -0.049] (p=4.5e-13, win 1.00) | 0.406 | 0.175 | 0.446 | +0.231 [0.185, 0.282] (p=2.3e-12, win 0.98) | -0.040 [-0.080, 0.002] (p=1.9e-01, win 0.50) |
  | H | 34 | 0.38 | 0.045 | 0.163 | 0.116 | -0.117 [-0.157, -0.080] (p=1.6e-09, win 0.97) | -0.070 [-0.092, -0.048] (p=1.7e-07, win 0.88) | 0.467 | 0.177 | 0.197 | +0.290 [0.228, 0.353] (p=1.2e-10, win 1.00) | +0.270 [0.207, 0.334] (p=1.6e-09, win 0.94) |
  | all | 485 | 0.68 | 0.326 | 0.467 | 0.341 | -0.141 [-0.155, -0.128] (p=5.0e-57, win 0.85) | -0.015 [-0.030, -0.001] (p=2.0e-01, win 0.56) | 0.740 | 0.471 | 0.593 | +0.269 [0.256, 0.284] (p=1.2e-80, win 0.99) | +0.147 [0.134, 0.160] (p=1.1e-63, win 0.88) |
  
  ### Deletion / insertion AUC, mean fill (test positives; deletion lower = better, insertion higher = better)
  
  | label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | D | 285 | 0.69 | 0.196 | 0.373 | 0.204 | -0.178 [-0.192, -0.164] (p=3.2e-46, win 0.96) | -0.009 [-0.022, 0.005] (p=4.0e-01, win 0.50) | 0.618 | 0.376 | 0.452 | +0.242 [0.222, 0.261] (p=1.3e-43, win 0.91) | +0.167 [0.149, 0.184] (p=3.7e-39, win 0.88) |
  | G | 62 | 0.72 | 0.080 | 0.123 | 0.145 | -0.043 [-0.064, -0.024] (p=1.0e-04, win 0.69) | -0.066 [-0.094, -0.039] (p=3.4e-05, win 0.68) | 0.532 | 0.121 | 0.611 | +0.412 [0.367, 0.457] (p=7.6e-12, win 1.00) | -0.079 [-0.130, -0.029] (p=6.5e-03, win 0.37) |
  | C | 62 | 0.73 | 0.198 | 0.120 | 0.768 | +0.078 [0.048, 0.111] (p=2.8e-05, win 0.34) | -0.570 [-0.607, -0.533] (p=7.6e-12, win 1.00) | 0.301 | 0.124 | 0.606 | +0.178 [0.140, 0.219] (p=8.3e-11, win 0.89) | -0.305 [-0.357, -0.253] (p=2.5e-11, win 0.05) |
  | A | 42 | 0.67 | 0.069 | 0.131 | 0.096 | -0.063 [-0.108, -0.023] (p=4.2e-02, win 0.62) | -0.027 [-0.050, -0.006] (p=2.0e-01, win 0.52) | 0.380 | 0.118 | 0.435 | +0.262 [0.219, 0.307] (p=1.5e-11, win 0.93) | -0.055 [-0.106, 0.001] (p=7.2e-02, win 0.43) |
  | H | 34 | 0.38 | 0.011 | 0.024 | 0.041 | -0.012 [-0.020, -0.006] (p=1.9e-06, win 0.88) | -0.029 [-0.041, -0.018] (p=1.2e-09, win 0.97) | 0.141 | 0.023 | 0.098 | +0.118 [0.087, 0.150] (p=5.8e-10, win 0.94) | +0.043 [0.013, 0.072] (p=3.0e-03, win 0.76) |
  | all | 485 | 0.68 | 0.157 | 0.264 | 0.248 | -0.106 [-0.120, -0.092] (p=2.6e-45, win 0.81) | -0.091 [-0.109, -0.072] (p=2.6e-13, win 0.62) | 0.513 | 0.264 | 0.466 | +0.249 [0.233, 0.264] (p=5.4e-75, win 0.92) | +0.047 [0.027, 0.069] (p=5.8e-09, win 0.66) |
  
  ### IDRiD localisation of the DR Grad-CAM (81 images, FOV only)
  
  Model on IDRiD: mean p(DR) 0.948, median 0.998, 97.5% with p >= 0.5. Grad-CAM energy on the optic disc 0.019 (centre map 0.029, OD area 0.026). Spearman(p(DR), EPG any-lesion) = 0.258 (p=2.0e-02).
  
  | lesions | n img | area frac (chance EPG) | area frac +15px (≈chance PG-tol) | method | PG strict | PG ±15 px | EPG | pixel AUROC | AP |
  |---|---|---|---|---|---|---|---|---|---|
  | any (MA∪HE∪EX∪SE) | 81 | 0.0418 | 0.352 | Grad-CAM DR | 0.14 | 0.79 | 0.0704 | 0.657 | 0.0811 |
  |  |  |  |  | random | 0.04 | 0.32 | 0.0417 | 0.499 | 0.0432 |
  |  |  |  |  | centre | 0.09 | 0.58 | 0.0482 | 0.575 | 0.0657 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.10 | 0.30 | 0.0475 | 0.518 | 0.0591 |
  | microaneurysms | 81 | 0.0027 | 0.137 | Grad-CAM DR | 0.01 | 0.46 | 0.0057 | 0.622 | 0.0078 |
  |  |  |  |  | random | 0.00 | 0.13 | 0.0027 | 0.499 | 0.0030 |
  |  |  |  |  | centre | 0.00 | 0.20 | 0.0030 | 0.580 | 0.0038 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.02 | 0.11 | 0.0025 | 0.515 | 0.0036 |
  | haemorrhages | 80 | 0.0174 | 0.132 | Grad-CAM DR | 0.10 | 0.59 | 0.0362 | 0.708 | 0.0629 |
  |  |  |  |  | random | 0.02 | 0.12 | 0.0174 | 0.502 | 0.0191 |
  |  |  |  |  | centre | 0.01 | 0.14 | 0.0162 | 0.477 | 0.0195 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.00 | 0.12 | 0.0140 | 0.505 | 0.0218 |
  | hard exudates | 81 | 0.0189 | 0.184 | Grad-CAM DR | 0.02 | 0.20 | 0.0247 | 0.640 | 0.0303 |
  |  |  |  |  | random | 0.01 | 0.16 | 0.0189 | 0.499 | 0.0201 |
  |  |  |  |  | centre | 0.06 | 0.42 | 0.0268 | 0.711 | 0.0517 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.06 | 0.15 | 0.0273 | 0.535 | 0.0340 |
  | soft exudates | 40 | 0.0062 | 0.035 | Grad-CAM DR | 0.00 | 0.03 | 0.0087 | 0.569 | 0.0113 |
  |  |  |  |  | random | 0.00 | 0.03 | 0.0062 | 0.502 | 0.0083 |
  |  |  |  |  | centre | 0.03 | 0.05 | 0.0050 | 0.397 | 0.0081 |
  |  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.03 | 0.10 | 0.0080 | 0.498 | 0.0180 |
  
  Paired differences Grad-CAM DR minus baseline (mean [95% bootstrap CI], Wilcoxon p):
  
  - any (MA∪HE∪EX∪SE): EPG vs random +0.029 [0.023, 0.034] p=2.7e-12; AUROC vs random +0.158 [0.134, 0.182] p=1.8e-13; EPG vs centre +0.022 [0.015, 0.029] p=8.2e-09; AUROC vs centre +0.082 [0.046, 0.118] p=3.5e-05; EPG vs gradcam_G +0.023 [0.003, 0.039] p=7.7e-08; AUROC vs gradcam_G +0.139 [0.107, 0.169] p=2.0e-10
  - microaneurysms: EPG vs random +0.003 [0.002, 0.004] p=5.6e-14; AUROC vs random +0.122 [0.102, 0.142] p=1.7e-13; EPG vs centre +0.003 [0.002, 0.003] p=1.5e-12; AUROC vs centre +0.042 [0.014, 0.068] p=2.3e-03; EPG vs gradcam_G +0.003 [0.002, 0.004] p=2.6e-12; AUROC vs gradcam_G +0.107 [0.084, 0.132] p=8.3e-11
  - haemorrhages: EPG vs random +0.019 [0.015, 0.023] p=5.6e-13; AUROC vs random +0.206 [0.175, 0.236] p=1.3e-13; EPG vs centre +0.020 [0.016, 0.024] p=1.1e-13; AUROC vs centre +0.231 [0.192, 0.272] p=7.5e-13; EPG vs gradcam_G +0.022 [0.017, 0.028] p=4.6e-13; AUROC vs gradcam_G +0.203 [0.163, 0.245] p=4.6e-12
  - hard exudates: EPG vs random +0.006 [0.002, 0.009] p=9.0e-05; AUROC vs random +0.141 [0.108, 0.172] p=3.6e-10; EPG vs centre -0.002 [-0.008, 0.003] p=4.1e-01; AUROC vs centre -0.071 [-0.117, -0.027] p=2.9e-03; EPG vs gradcam_G -0.003 [-0.024, 0.011] p=1.2e-03; AUROC vs gradcam_G +0.104 [0.057, 0.151] p=7.4e-05
  - soft exudates: EPG vs random +0.003 [0.000, 0.005] p=4.2e-01; AUROC vs random +0.067 [-0.010, 0.139] p=9.7e-02; EPG vs centre +0.004 [0.001, 0.006] p=1.8e-02; AUROC vs centre +0.173 [0.078, 0.269] p=9.4e-04; EPG vs gradcam_G +0.001 [-0.005, 0.005] p=1.1e-01; AUROC vs gradcam_G +0.071 [-0.037, 0.181] p=1.7e-01
  
  ### On-device cost: n100, OPENBLAS_NUM_THREADS=1 (primary)
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | ort (onnxruntime 1.30.0) | plain | 7.5 / 7.5 / 7.5 | – | – | 108.4 | 9.1e-06 | – |
  | ort (onnxruntime 1.30.0) | head | 8.1 / 8.1 / 8.2 | 0.8 | 0.7 | 107.9 | 9.2e-06 | – |
  | ort (onnxruntime 1.30.0) | gradcam1 | 9.2 / 9.2 / 9.2 | 1.8 | 1.7 | 117.5 | 9.2e-06 | 2.4e-04 |
  | ort (onnxruntime 1.30.0) | gradcam5 | 12.4 / 12.3 / 12.4 | 4.8 | 4.9 | 127.8 | 9.2e-06 | 2.5e-04 |
  | ort_nospin (onnxruntime 1.30.0) | plain | 7.9 / 7.9 / 7.9 | – | – | 106.6 | 9.1e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | head | 9.3 / 9.3 / 9.3 | 0.9 | 1.4 | 109.6 | 9.2e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | gradcam1 | 9.7 / 9.7 / 9.7 | 1.8 | 1.8 | 115.5 | 9.2e-06 | 2.4e-04 |
  | ort_nospin (onnxruntime 1.30.0) | gradcam5 | 12.9 / 12.8 / 12.9 | 4.8 | 5.0 | 130.6 | 9.2e-06 | 2.5e-04 |
  | ov (openvino 2026.4.1) | plain | 8.2 / 8.2 / 8.2 | – | – | 147.9 | 2.1e-06 | – |
  | ov (openvino 2026.4.1) | head | 9.0 / 9.0 / 9.0 | 0.8 | 0.8 | 151.4 | 1.9e-06 | – |
  | ov (openvino 2026.4.1) | gradcam1 | 10.1 / 10.1 / 10.2 | 1.8 | 1.9 | 160.5 | 1.9e-06 | 2.4e-04 |
  | ov (openvino 2026.4.1) | gradcam5 | 13.0 / 12.9 / 13.0 | 4.4 | 4.8 | 174.6 | 1.9e-06 | 2.5e-04 |
  
  ### On-device cost: n100, OpenBLAS default threads (first pass)
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | ort (onnxruntime 1.30.0) | plain | 7.4 / 7.4 / 7.5 | – | – | 107.2 | 9.1e-06 | – |
  | ort (onnxruntime 1.30.0) | head | 32.6 / 32.0 / 48.0 | 5.8 | 25.2 | 116.7 | 9.2e-06 | – |
  | ort (onnxruntime 1.30.0) | gradcam1 | 48.5 / 48.0 / 72.0 | 23.9 | 41.1 | 125.6 | 9.2e-06 | 2.4e-04 |
  | ort (onnxruntime 1.30.0) | gradcam5 | 86.3 / 96.0 / 112.0 | 57.5 | 78.9 | 139.0 | 9.2e-06 | 2.5e-04 |
  | ort_nospin (onnxruntime 1.30.0) | plain | 8.0 / 7.9 / 7.9 | – | – | 108.2 | 9.1e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | head | 13.2 / 13.2 / 14.7 | 0.7 | 5.3 | 115.1 | 9.2e-06 | – |
  | ort_nospin (onnxruntime 1.30.0) | gradcam1 | 15.6 / 15.7 / 16.8 | 1.5 | 7.6 | 123.4 | 9.2e-06 | 2.4e-04 |
  | ort_nospin (onnxruntime 1.30.0) | gradcam5 | 17.4 / 17.5 / 19.1 | 4.0 | 9.4 | 137.6 | 9.2e-06 | 2.5e-04 |
  | ov (openvino 2026.4.1) | plain | 8.3 / 8.2 / 8.3 | – | – | 147.8 | 2.1e-06 | – |
  | ov (openvino 2026.4.1) | head | 22.7 / 21.5 / 34.1 | 1.0 | 14.4 | 157.5 | 1.9e-06 | – |
  | ov (openvino 2026.4.1) | gradcam1 | 24.1 / 21.9 / 38.1 | 2.0 | 15.8 | 166.8 | 1.9e-06 | 2.4e-04 |
  | ov (openvino 2026.4.1) | gradcam5 | 26.6 / 25.1 / 38.8 | 4.2 | 18.3 | 180.7 | 1.9e-06 | 2.5e-04 |
  
  ### On-device cost: pi, OPENBLAS_NUM_THREADS=1
  
  | runtime | mode | latency mean / p50 / p95 (ms) | NumPy part mean (ms) | extra vs plain (ms) | peak RSS (MB) | prob diff vs Mac torch | map max diff vs Mac torch |
  |---|---|---|---|---|---|---|---|
  | tflite (tflite-runtime 2.13.0) | plain | 547.0 / 546.9 / 569.6 | – | – | 63.3 | 1.2e-04 | – |
  | tflite (tflite-runtime 2.13.0) | head | 553.5 / 553.6 / 573.4 | 13.5 | 6.5 | 63.1 | 1.2e-04 | – |
  | tflite (tflite-runtime 2.13.0) | gradcam1 | 597.4 / 599.6 / 621.9 | 61.5 | 50.4 | 70.3 | 1.2e-04 | 2.5e-04 |
  | tflite (tflite-runtime 2.13.0) | gradcam5 | 693.0 / 693.8 / 720.6 | 155.8 | 146.0 | 81.2 | 1.2e-04 | 3.9e-04 |

- **Device conditions**:
  - **N100**: idle before every config (load avg ≤ 0.26, 3.2 GHz). The suite was
    run twice. Pass 1 used numpy's default OpenBLAS threading
    (`results/n100_run010/openblas_default/`). Pass 2, the primary one, used
    `OPENBLAS_NUM_THREADS=1` (`results/n100_run010/`). A first attempt failed
    before timing because `/usr/bin/time` is not installed on the N100 (rc 127,
    `out_attempt1_no_usr_bin_time/`); the driver now skips it there.
  - **Pi**: `get_throttled` was 0x50005 or 0x50000 and the ARM clock was
    600 MHz at every before/after sample (14:03-14:09 CEST), 48-55 °C. As in
    runs 006/009, all Pi latencies are under the ~600 MHz under-voltage cap,
    roughly 2x pessimistic.
  - **Pi first attempt** (`results/pi_run010/out_attempt1_openblas_default/`),
    default OpenBLAS threads: `plain` and `head` completed. `gradcam1` died with
    SIGSEGV after 74 s. `gradcam5` was still running after 16 min at ~390 % CPU
    (about 70 s expected) and was killed. 10-iteration probes with 1 and 4
    OpenBLAS threads both completed and gave the same maps, so the failure
    appears only in long runs with 4 OpenBLAS threads competing with the 4
    TFLite/XNNPACK threads on 4 cores. The cause was not pinned down. The final
    Pi suite used `OPENBLAS_NUM_THREADS=1`, and all 4 configs completed.

- **Artifacts**:
  - Scripts: `results/run010_common.py`, `run010_check_reference.py`,
    `run010_export_edge.py`, `run010_export_tflite.py`, `run010_parity_edge.py`,
    `run010_sanity.py`, `run010_train_randlabels.py` (not run),
    `run010_faithfulness.py`, `run010_idrid_prep.py`, `run010_localisation.py`,
    `run010_make_device_fixture.py`, `run010_figures.py`, `run010_tables.py`
    (generates every table above from the JSONs), `run010_run_final.sh`. Edge
    code: `edge/gradcam_numpy.py`, `edge/gradcam_bench.py`, `edge/run010_device.sh`.
  - Results (`results/run010/`, `*_final*`; development pass `*_dev*`):
    `reference_vs_pytorch_grad_cam_final.json`, `parity_{onnx,tflite}_final.json`,
    `sanity_final.json`, `faithfulness_final.json` (+ `_per_image.json`,
    `_curves.npz`), `localisation_final.json` (+ `_per_image.csv`),
    `idrid_manifest.csv`, `figure_selection_final.json`,
    `figure_idrid_selection_final.json`, `probs_final_{val,test}.npy`,
    `tables_final.md`, `log_*_final.txt`, `log_final_driver.txt`,
    `env_venv_run010_freeze.txt`.
  - Device results: `results/n100_run010/` and `results/pi_run010/`
    (`results_<runtime>_<mode>.json`, `log_*.txt`, `suite.log`, failed
    attempts in sub-directories). On the devices: `~/eye-edge/run010/`.
  - Edge artifacts: `edge/artifacts/run010_final_*`,
    `edge/artifacts/run010_bundle_final/`.
  - Figures (PDF + PNG preview): `paper/figs/run010_gradcam_examples`,
    `run010_idrid_overlays`, `run010_deletion_insertion`,
    `run010_sanity_cascade`. Colour map viridis (perceptually uniform, CVD-safe),
    each with a colourbar. Colour = positive Grad-CAM evidence for the named
    label scaled to that image's maximum: 0 (dark purple) = none, 1 (yellow) =
    strongest. It is not a probability. Example selection (seed 2026, uniform
    within each category; candidates: TP D 253, G 55, C 50, A 34, H 34; DR FP 360;
    other-disease FP 1451; mild-NPDR misses 22): `4354_left` (DR TP, moderate
    NPDR), `1294_right` (G), `726_right` (C), `608_left` (A), `4214_right` (H),
    `3431_right` (DR FP), `3110_right` (glaucoma FP), `4542_right` (mild NPDR
    missed). IDRiD overlays (seed 2027): IDRiD_01, 12, 32, 50.
  - IDRiD data: `data/idrid/` (zip, sha256, Kaggle metadata, `prep512/`).

- **Verdict (final model)**:
  - **The edge Grad-CAM is the same Grad-CAM, and no autodiff is needed on the
    device.**
    - The reference matches pytorch-grad-cam to 1.5e-6.
    - Mac, 300 test images x 5 labels: the ONNX backbone + NumPy head gives
      heatmaps within 2.3e-4 of PyTorch autograd (max abs diff, maps in [0,1])
      and probabilities within 1.1e-5. TFLite: 7.2e-4 (99th percentile 1.6e-4)
      and 6.8e-5. The hottest pixel is identical in 1499 of 1500 maps for both.
    - The NumPy maths alone (on torch features) adds ≤ 1.4e-5. The remaining
      error is the backbone runtime's.
    - On the devices, against the stored Mac reference: N100 maps ≤ 2.5e-4,
      probabilities ≤ 9.2e-6 (ORT) / 1.9e-6 (OpenVINO); Pi maps ≤ 3.9e-4,
      probabilities ≤ 1.2e-4. These diffs include the float16 storage of the
      reference maps.
  - **Passes the model-randomisation sanity check (Adebayo).**
    - Re-initialising only the classifier drops the median Spearman ρ between
      the randomised and original map to 0.075. That is already below the 0.144
      between the same label's maps of two *different* images.
    - Every deeper cascade stage stays between −0.045 and 0.022. Randomising only
      the Grad-CAM layer (blocks.5) gives 0.067.
    - SSIM stays at 0.22-0.48, below the different-image level of 0.53. On these
      smooth, mostly empty maps SSIM is dominated by background, so ρ is the
      informative metric.
    - The maps are class-specific: the median ρ between two labels' maps of the
      same image is 0.087.
    - The data-randomisation test was **not run** (compute).
  - **Faithful: clearly better than random, mostly better than a centre prior.**
    Blur fill, all 485 positive (image, label) pairs:
    - Insertion AUC: 0.740 vs 0.471 (random) and 0.593 (centre). Δ vs centre
      +0.147 [0.134, 0.160], Grad-CAM wins on 88 % of pairs.
    - Deletion AUC: 0.326 vs 0.467 (random) and 0.341 (centre). Δ vs centre
      −0.015 [−0.030, −0.001], paired Wilcoxon p = 0.20, so deletion does not
      separate Grad-CAM from the centre prior.
    - Per label:
      - **DR deletion is worse than centre** (0.331 vs 0.294, Δ +0.037 [0.021, 0.051]).
      - For cataract, deletion does not separate Grad-CAM from random (Δ −0.018,
        p = 0.29). This fits cataract being a global media opacity.
      - AMD insertion ties with centre (Δ −0.040 [−0.080, 0.002]).
    - Mean-colour fill is confounded for cataract: a grey disc looks like an
      opaque lens, so deleting the centre keeps p(cataract) high (centre
      deletion AUC 0.768). Blur is the primary fill.
    - The development checkpoint (trained on these test images) had *not* beaten
      the centre prior on insertion (Δ −0.009 [−0.024, 0.006]). Faithfulness
      depends on the model as well as the method.
  - **DR lesion localisation: above every baseline overall, weak in absolute
    terms, worst for microaneurysms.** 81 IDRiD images, external data; the model
    gives 97.5 % of them p(DR) ≥ 0.5.
    - Any-lesion pixel AUROC is 0.657, against 0.499 (random), 0.575 (centre)
      and 0.518 (wrong-class glaucoma map).
    - EPG is 0.070, 1.7x the lesion area fraction of 0.042.
    - The map's maximum hits a lesion in 11/81 images (0.14), against 0.04
      (random), 0.09 (centre) and 0.10 (glaucoma map).
    - The ±15 px pointing game (0.79) must be read against its chance level: the
      dilated lesions cover 35 % of the FOV.
    - Haemorrhages localise best: AUROC 0.708, pointing game 8/81 (0.10).
    - Hard exudates: AUROC 0.640, *below* the centre map (0.711, Δ −0.071
      [−0.117, −0.027]). Exudates cluster around the macula near the image
      centre.
    - Soft exudates (n = 40): AUROC 0.569, not different from random (Δ +0.067
      [−0.010, 0.139]).
    - **Microaneurysms**: the pointing game hits 1/81, EPG 0.0057 (2.1x the
      0.0027 chance level), AUROC 0.622. Grad-CAM at the 16x16 feature resolution
      (32 px cells at 512 px) cannot point at lesions a few pixels wide, as
      Saporta 2022 found for small pathologies.
    - Localisation correlates weakly with confidence (Spearman ρ between p(DR)
      and EPG = 0.258, p = 0.020).
    - The DR map puts 1.9 % of its mass on the optic disc, against a disc area of
      2.6 %, so there is no optic-disc shortcut on IDRiD. On ODIR, several
      sampled maps peak at the optic disc (DR FP, glaucoma FP, mild-NPDR miss)
      or at the fovea (DR TP).
  - **On-device cost is small on the N100 and moderate on the Pi**, with
    `OPENBLAS_NUM_THREADS=1`:
    - **N100, ORT** (plain 7.5 ms, 108 MB): Grad-CAM for DR adds +1.7 ms
      (9.2 ms) and all 5 labels add +4.9 ms (12.4 ms). Peak RSS +9 / +19 MB.
      The probability-only backbone + NumPy head path costs +0.7 ms.
    - **N100, OpenVINO**: +1.9 / +4.8 ms on 8.2 ms.
    - **Pi 3B, TFLite, 4 threads, 600 MHz cap** (plain 547 ms, 63 MB): +50 ms for
      DR (597 ms, +9 %) and +146 ms for 5 labels (693 ms, +27 %). The NumPy part
      itself is 62 / 156 ms. Peak RSS +7 / +18 MB (70 / 81 MB, well inside the
      870 MB board).
    - **Deployment caveat**: with numpy's default multi-threaded OpenBLAS next to
      the inference runtime, the N100 overhead grew to +79 ms (ORT, 5 labels;
      +9.4 ms with ORT thread spinning off; +18 ms OpenVINO). On the Pi it gave a
      crash and a stall. The NumPy Grad-CAM must be single-threaded
      (`OPENBLAS_NUM_THREADS=1`) when it runs next to an inference runtime.
  - **Paper framing**: Grad-CAM can be offered on both devices at little extra
    cost and is verified to be the reference Grad-CAM. It passes the randomisation
    check, and its relevance ranking is faithful to the model, beating random and,
    on insertion, a centre prior. It is **not** a lesion detector. It overlaps
    haemorrhages above chance, does no better than a centre prior on exudates,
    and does not show microaneurysms, the lesions that define mild NPDR, the
    model's weakest grade. With Sayres 2019 (heatmaps hurt readers on no-DR
    cases) and Ghassemi 2021, the map should be presented as a coarse "where the
    model looked" aid next to a positive prediction, not as evidence that the
    prediction is right.

- **Limitations**:
  - No data-randomisation test. One randomisation seed. One random-map draw per
    image in faithfulness (50 per image in localisation).
  - IDRiD is one Indian clinic and one camera, with the disc truncated at
    top/bottom (black bands after padding). Every image has DR, so the DR map is
    never evaluated on negatives. Masks are binarised with an any-coverage rule
    at 512 px, which slightly enlarges small lesions.
  - Pixel-wise deletion/insertion creates out-of-distribution images (the ROAR
    argument, Hooker 2019). Two fills are reported, with no retraining. Whole
    images are ranked, including the black corners.
  - The figure's operating thresholds (90 % validation sensitivity) are very low
    for H (0.00054) and A (0.039), because pos_weight training compresses those
    probabilities. So the "correct positive" H/A examples are correct only at
    those thresholds.
  - Grad-CAM only (no Grad-CAM++, integrated gradients or Score-CAM), and only
    the last-conv target layer (the only one available on the device without a
    second backbone output).
  - Pi latencies are under the under-voltage cap. Single run per config. The Pi
    clock was sampled before/after each config only.
