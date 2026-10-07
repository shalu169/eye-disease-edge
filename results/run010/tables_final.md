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
