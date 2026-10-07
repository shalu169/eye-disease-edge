### Reference Grad-CAM vs pytorch-grad-cam

64 test images x 5 labels = 320 maps. Low-res (16x16, min-max) max |diff| 1.0e-06; 512 px library-style post-processing max |diff| 1.0e-06; our display map vs library output max |diff| 1.2e-06, min Spearman 0.9999999999. Grad-CAM forward vs plain forward prob max |diff| 8.9e-07.

### Edge Grad-CAM parity vs PyTorch autograd (Mac)

| runtime | images / maps | feature map max abs diff | prob max diff, plain full model | prob max diff, backbone + NumPy head | prob max diff, NumPy head on torch features | heatmap max abs diff | mean of per-map max diff | heatmap max diff, NumPy head on torch features | min Spearman | argmax identical |
|---|---|---|---|---|---|---|---|---|---|---|
| onnx (1.30.0) | 300 / 1500 | 2.1e-03 (2.5e-05 rel) | 1.4e-05 | 1.5e-05 | 4.8e-07 | 1.1e-04 | 8.1e-06 | 2.9e-06 | 0.99999999 | 1.000 |
| tflite (tensorflow 2.18.0) | 300 / 1500 | 1.9e-03 (2.8e-05 rel) | 2.3e-05 | 2.4e-05 | 4.8e-07 | 1.0e-04 | 1.3e-05 | 2.7e-06 | 0.99999998 | 1.000 |

### Sanity checks (Adebayo 2018): Spearman rho / SSIM of randomised vs original map

100 test images, all 5 labels. Median over images; per-label Spearman medians.

| condition | mean abs Δp | Spearman (all) | SSIM (all) | D | G | C | A | H | all-zero maps |
|---|---|---|---|---|---|---|---|---|---|
| cascade → classifier | 0.437 | 0.080 | 0.421 | 0.24 | -0.07 | 0.01 | 0.12 | 0.20 | 0 |
| cascade → conv_head | 0.502 | 0.137 | 0.422 | 0.06 | 0.02 | 0.13 | 0.20 | 0.28 | 0 |
| cascade → blocks.5 | 0.692 | 0.124 | 0.387 | -0.05 | 0.00 | 0.15 | 0.29 | 0.31 | 0 |
| cascade → blocks.4 | 0.425 | 0.069 | 0.319 | 0.16 | 0.07 | 0.07 | -0.04 | 0.10 | 0 |
| cascade → blocks.3 | 0.396 | 0.038 | 0.365 | 0.03 | 0.03 | 0.02 | 0.12 | -0.02 | 0 |
| cascade → blocks.2 | 0.417 | 0.046 | 0.389 | 0.05 | 0.01 | 0.06 | 0.06 | 0.05 | 0 |
| cascade → blocks.1 | 0.776 | 0.100 | 0.260 | 0.25 | 0.04 | 0.01 | 0.10 | 0.13 | 0 |
| cascade → blocks.0 | 0.440 | 0.032 | 0.320 | 0.20 | -0.07 | 0.04 | 0.14 | -0.09 | 0 |
| cascade → stem | 0.431 | 0.011 | 0.352 | -0.03 | -0.00 | 0.05 | -0.03 | 0.07 | 0 |
| independent: blocks.5 only | 0.175 | 0.143 | 0.433 | 0.15 | 0.07 | -0.10 | 0.32 | 0.33 | 0 |
| reference: different image, same label | – | 0.112 | 0.471 | 0.17 | 0.10 | 0.09 | 0.10 | 0.12 | 0 |
| reference: centre-bias map | – | 0.158 | 0.205 | 0.37 | 0.13 | -0.02 | 0.14 | 0.22 | 0 |

Class specificity (same image, other label's map), Spearman median: D_vs_G 0.009, D_vs_C 0.067, D_vs_A 0.131, D_vs_H 0.267; all label pairs median 0.155 [IQR -0.010, 0.307].

### Deletion / insertion AUC, blur fill (test positives; deletion lower = better, insertion higher = better)

| label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 285 | 0.74 | 0.368 | 0.471 | 0.435 | -0.103 [-0.123, -0.081] (p=4.3e-18, win 0.72) | -0.067 [-0.089, -0.044] (p=7.4e-09, win 0.66) | 0.553 | 0.470 | 0.548 | +0.083 [0.066, 0.101] (p=1.9e-18, win 0.73) | +0.005 [-0.016, 0.025] (p=6.4e-01, win 0.49) |
| G | 62 | 0.97 | 0.345 | 0.543 | 0.348 | -0.198 [-0.247, -0.149] (p=1.1e-08, win 0.81) | -0.003 [-0.051, 0.045] (p=6.0e-01, win 0.61) | 0.726 | 0.564 | 0.702 | +0.163 [0.122, 0.205] (p=4.0e-09, win 0.85) | +0.024 [-0.009, 0.056] (p=2.0e-01, win 0.56) |
| C | 62 | 0.97 | 0.542 | 0.608 | 0.571 | -0.066 [-0.101, -0.031] (p=3.0e-04, win 0.73) | -0.029 [-0.077, 0.022] (p=9.3e-02, win 0.68) | 0.663 | 0.612 | 0.750 | +0.050 [0.017, 0.089] (p=5.4e-03, win 0.73) | -0.087 [-0.133, -0.043] (p=1.3e-03, win 0.34) |
| A | 42 | 0.97 | 0.300 | 0.474 | 0.369 | -0.173 [-0.221, -0.123] (p=3.3e-08, win 0.83) | -0.069 [-0.122, -0.015] (p=2.7e-02, win 0.64) | 0.658 | 0.473 | 0.672 | +0.185 [0.121, 0.249] (p=1.2e-06, win 0.86) | -0.015 [-0.057, 0.031] (p=2.1e-01, win 0.40) |
| H | 34 | 0.96 | 0.064 | 0.241 | 0.127 | -0.176 [-0.215, -0.143] (p=1.2e-10, win 1.00) | -0.062 [-0.089, -0.035] (p=1.3e-04, win 0.74) | 0.444 | 0.255 | 0.482 | +0.189 [0.144, 0.236] (p=5.8e-10, win 0.97) | -0.038 [-0.091, 0.017] (p=1.6e-01, win 0.41) |
| all | 485 | 0.83 | 0.360 | 0.482 | 0.414 | -0.121 [-0.137, -0.107] (p=1.5e-38, win 0.76) | -0.054 [-0.070, -0.036] (p=2.4e-11, win 0.66) | 0.591 | 0.485 | 0.600 | +0.105 [0.092, 0.120] (p=3.2e-37, win 0.78) | -0.009 [-0.024, 0.006] (p=2.3e-01, win 0.47) |

### Deletion / insertion AUC, mean fill (test positives; deletion lower = better, insertion higher = better)

| label | n | mean p | del Grad-CAM | del random | del centre | Δ vs random [95% CI] | Δ vs centre [95% CI] | ins Grad-CAM | ins random | ins centre | Δ vs random [95% CI] | Δ vs centre [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | 285 | 0.74 | 0.206 | 0.292 | 0.221 | -0.087 [-0.099, -0.074] (p=8.3e-30, win 0.81) | -0.015 [-0.031, 0.002] (p=1.4e-01, win 0.53) | 0.478 | 0.287 | 0.430 | +0.191 [0.171, 0.211] (p=2.0e-41, win 0.90) | +0.048 [0.031, 0.065] (p=7.9e-09, win 0.67) |
| G | 62 | 0.97 | 0.168 | 0.210 | 0.208 | -0.042 [-0.072, -0.013] (p=1.5e-02, win 0.65) | -0.040 [-0.076, -0.006] (p=7.9e-02, win 0.58) | 0.769 | 0.207 | 0.703 | +0.563 [0.518, 0.605] (p=7.6e-12, win 1.00) | +0.067 [0.025, 0.109] (p=3.1e-03, win 0.69) |
| C | 62 | 0.97 | 0.143 | 0.102 | 0.837 | +0.040 [0.025, 0.058] (p=4.6e-06, win 0.29) | -0.695 [-0.726, -0.659] (p=7.6e-12, win 1.00) | 0.247 | 0.103 | 0.615 | +0.145 [0.116, 0.177] (p=9.7e-12, win 0.97) | -0.368 [-0.422, -0.316] (p=1.4e-11, win 0.03) |
| A | 42 | 0.97 | 0.152 | 0.142 | 0.208 | +0.011 [-0.028, 0.050] (p=2.3e-01, win 0.38) | -0.055 [-0.094, -0.015] (p=2.3e-02, win 0.64) | 0.390 | 0.147 | 0.498 | +0.243 [0.207, 0.283] (p=4.5e-13, win 1.00) | -0.108 [-0.139, -0.078] (p=7.1e-08, win 0.17) |
| H | 34 | 0.96 | 0.041 | 0.064 | 0.167 | -0.023 [-0.039, -0.009] (p=4.3e-03, win 0.68) | -0.126 [-0.147, -0.106] (p=1.2e-10, win 1.00) | 0.287 | 0.058 | 0.310 | +0.230 [0.196, 0.266] (p=1.2e-10, win 1.00) | -0.023 [-0.053, 0.009] (p=5.7e-02, win 0.35) |
| all | 485 | 0.83 | 0.177 | 0.228 | 0.293 | -0.052 [-0.062, -0.041] (p=1.7e-19, win 0.68) | -0.116 [-0.139, -0.094] (p=9.1e-16, win 0.64) | 0.465 | 0.225 | 0.486 | +0.240 [0.222, 0.257] (p=8.6e-75, win 0.94) | -0.021 [-0.040, -0.002] (p=7.1e-01, win 0.53) |

### IDRiD localisation of the DR Grad-CAM (81 images, FOV only)

Model on IDRiD: mean p(DR) 0.939, median 0.996, 96.3% with p >= 0.5. Grad-CAM energy on the optic disc 0.008 (centre map 0.029, OD area 0.026). Spearman(p(DR), EPG any-lesion) = 0.385 (p=3.8e-04).

| lesions | n img | area frac (chance EPG) | area frac +15px (≈chance PG-tol) | method | PG strict | PG ±15 px | EPG | pixel AUROC | AP |
|---|---|---|---|---|---|---|---|---|---|
| any (MA∪HE∪EX∪SE) | 81 | 0.0418 | 0.352 | Grad-CAM DR | 0.16 | 0.83 | 0.0777 | 0.692 | 0.0907 |
|  |  |  |  | random | 0.04 | 0.32 | 0.0417 | 0.499 | 0.0432 |
|  |  |  |  | centre | 0.09 | 0.58 | 0.0482 | 0.575 | 0.0657 |
|  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.11 | 0.27 | 0.0572 | 0.561 | 0.0704 |
| microaneurysms | 81 | 0.0027 | 0.137 | Grad-CAM DR | 0.00 | 0.43 | 0.0055 | 0.650 | 0.0079 |
|  |  |  |  | random | 0.00 | 0.13 | 0.0027 | 0.499 | 0.0030 |
|  |  |  |  | centre | 0.00 | 0.20 | 0.0030 | 0.580 | 0.0038 |
|  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.00 | 0.04 | 0.0024 | 0.533 | 0.0033 |
| haemorrhages | 80 | 0.0174 | 0.132 | Grad-CAM DR | 0.11 | 0.55 | 0.0353 | 0.696 | 0.0584 |
|  |  |  |  | random | 0.02 | 0.12 | 0.0174 | 0.502 | 0.0191 |
|  |  |  |  | centre | 0.01 | 0.14 | 0.0162 | 0.477 | 0.0195 |
|  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.03 | 0.14 | 0.0226 | 0.570 | 0.0297 |
| hard exudates | 81 | 0.0189 | 0.184 | Grad-CAM DR | 0.02 | 0.26 | 0.0321 | 0.708 | 0.0439 |
|  |  |  |  | random | 0.01 | 0.16 | 0.0189 | 0.499 | 0.0201 |
|  |  |  |  | centre | 0.06 | 0.42 | 0.0268 | 0.711 | 0.0517 |
|  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.05 | 0.12 | 0.0263 | 0.556 | 0.0355 |
| soft exudates | 40 | 0.0062 | 0.035 | Grad-CAM DR | 0.05 | 0.07 | 0.0111 | 0.647 | 0.0166 |
|  |  |  |  | random | 0.00 | 0.03 | 0.0062 | 0.502 | 0.0083 |
|  |  |  |  | centre | 0.03 | 0.05 | 0.0050 | 0.397 | 0.0081 |
|  |  |  |  | Grad-CAM glaucoma (wrong class) | 0.07 | 0.12 | 0.0130 | 0.514 | 0.0352 |

Paired differences Grad-CAM DR minus baseline (mean [95% bootstrap CI], Wilcoxon p):

- any (MA∪HE∪EX∪SE): EPG vs random +0.036 [0.030, 0.042] p=4.4e-13; AUROC vs random +0.193 [0.169, 0.217] p=2.7e-14; EPG vs centre +0.030 [0.023, 0.037] p=4.4e-11; AUROC vs centre +0.117 [0.082, 0.152] p=2.1e-08; EPG vs gradcam_G +0.020 [0.001, 0.035] p=6.7e-07; AUROC vs gradcam_G +0.130 [0.096, 0.163] p=9.6e-09
- microaneurysms: EPG vs random +0.003 [0.002, 0.003] p=5.6e-14; AUROC vs random +0.150 [0.130, 0.170] p=3.6e-14; EPG vs centre +0.002 [0.002, 0.003] p=3.3e-13; AUROC vs centre +0.070 [0.042, 0.096] p=1.2e-06; EPG vs gradcam_G +0.003 [0.002, 0.004] p=3.0e-13; AUROC vs gradcam_G +0.117 [0.089, 0.145] p=2.9e-10
- haemorrhages: EPG vs random +0.018 [0.014, 0.022] p=2.4e-12; AUROC vs random +0.193 [0.162, 0.225] p=1.8e-12; EPG vs centre +0.019 [0.015, 0.023] p=2.5e-13; AUROC vs centre +0.218 [0.179, 0.256] p=4.2e-12; EPG vs gradcam_G +0.013 [0.006, 0.018] p=2.1e-08; AUROC vs gradcam_G +0.125 [0.085, 0.164] p=5.3e-07
- hard exudates: EPG vs random +0.013 [0.009, 0.017] p=4.8e-10; AUROC vs random +0.209 [0.178, 0.240] p=3.3e-13; EPG vs centre +0.005 [0.001, 0.009] p=5.9e-05; AUROC vs centre -0.002 [-0.041, 0.035] p=9.3e-01; EPG vs gradcam_G +0.006 [-0.013, 0.018] p=1.5e-06; AUROC vs gradcam_G +0.152 [0.106, 0.194] p=1.2e-08
- soft exudates: EPG vs random +0.005 [0.002, 0.009] p=2.4e-02; AUROC vs random +0.145 [0.082, 0.204] p=1.7e-04; EPG vs centre +0.006 [0.003, 0.010] p=3.9e-03; AUROC vs centre +0.250 [0.169, 0.331] p=1.4e-06; EPG vs gradcam_G -0.002 [-0.008, 0.003] p=8.2e-02; AUROC vs gradcam_G +0.133 [0.051, 0.223] p=4.1e-03
