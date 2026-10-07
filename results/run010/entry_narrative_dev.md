## Run 010 — Grad-CAM: reference vs edge implementation, sanity checks, faithfulness, lesion localisation, on-device cost

- **Date**: 2026-10-03
- **Status**: DEVELOPMENT NUMBERS (see "Model" below). The final-checkpoint
  pipeline is scripted (`results/run010_run_final.sh`) and has to be re-run
  once `results/run008_FINAL_CHECKPOINT.json` says `"updated_after_all_seeds": true`.
- **Purpose**: the paper claims an on-device Grad-CAM path that needs no
  automatic differentiation on the device. Literature (Adebayo 2018, Arun 2021,
  Saporta 2022, Sayres 2019, Ghassemi 2021) warns that saliency maps can be
  independent of the model, unfaithful, and poor at small lesions. This run
  tests (1) that the edge implementation is numerically the same Grad-CAM as
  the PyTorch reference, (2) Adebayo's randomisation sanity checks, (3)
  deletion/insertion faithfulness against random and centre-bias baselines,
  (4) localisation against expert pixel-level DR lesion masks (IDRiD), and
  (5) its latency/memory cost on the N100 and the Pi 3B.
- **Model**: development checkpoint
  `results/checkpoints/run007c_joint5_eyeD_512_s1_best_epoch22.pt` (512 px,
  joint 5-label, per-eye DR; same architecture and size as the run 008
  model). **Caveat**: this checkpoint was trained on the original 5122-image
  train split, which contains the run 008 TEST images, so every TEST-set
  number below (faithfulness, example figure) is computed on images the dev
  model was trained on. The IDRiD localisation numbers are external data for
  either model. (The run 008 driver stopped after starting A s0 at 01:40 and was
  restarted at 11:51; the final checkpoint is expected around 13:30.)
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
  CPU). Dev artifacts: `edge/artifacts/run010_dev007c_{backbone,full}_fp32.{onnx,tflite}`,
  `run010_dev007c_head.npz`, `run010_dev007c_export.json`,
  `run010_dev007c_tflite_export.json`.
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
    `results/run010_train_randlabels.py`) was **not run**. At 512 px it needs about
    30 min on MPS, which belongs to run 008, and about 11 h on CPU (measured 0.32
    s/image per training step).
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
    logged in `results/run010/figure_selection_dev.json` and
    `figure_idrid_selection_dev.json`.
