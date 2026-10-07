## Run 011 — Confidence calibration: temperature vs Platt scaling, on device, under INT8

- **Date**: 2026-10-03
- **Purpose**: check whether the run 008 models' probabilities can be
  shown to a health worker as confidence, whether a post-hoc fix works,
  whether the fix survives export to the N100/Pi, and whether INT8 breaks it.
- **Method** (`results/run011_calibration.py`):
  - Inputs: run 008 per-image probabilities on validation and test
    (`results/run008_probs/`), all 9 models.
  - Per label, fitted on VALIDATION negative log-likelihood, applied
    unchanged to TEST: (i) temperature scaling σ(z/T) (Guo et al. 2017);
    (ii) Platt scaling σ(a·z + b). Platt was added after temperature
    scaling failed (see verdict): the loss's positive-class weight w shifts
    logits by roughly log w, which needs a bias term.
  - Test metrics: ECE (15 equal-width bins), adaptive ECE (15 equal-mass
    bins), Brier, NLL, mean predicted probability vs prevalence. AUC is
    unchanged by both maps (monotonic).
  - Final model (A s2): patient-clustered bootstrap (1000) of ECE and of the
    ECE change. The bootstrap distribution of ECE is biased upward
    (resampling duplicates add binning noise), so for DR the Platt point
    estimate falls below its own percentile interval. Report the CI of the
    change, which is unaffected.
  - Device check: validation-fitted Platt maps applied to the final model's
    test probabilities from the N100 (ONNX Runtime CPU, OpenVINO CPU/iGPU)
    and the Pi (TFLite FP32, 4 threads) from run 009.
  - INT8 (development, `results/run011_int8_transfer.py`): 224 px run002
    model, Pi run 006 TFLite probabilities on the 1270-image validation set.
    A Platt map fitted to FP32 outputs is applied unchanged to the
    dynamic-range and full-integer INT8 outputs of the same images.
- **Results — final model (A s2), test set**:

  | label | prevalence | mean p raw → Platt | T | Platt (a, b) | ECE raw | ECE temp. | ECE Platt | ΔECE Platt−raw, 95 % CI | Brier raw → Platt | NLL raw → Platt |
  |---|---|---|---|---|---|---|---|---|---|---|
  | DR (per-eye) | 0.278 | 0.429 → 0.283 | 1.32 | (0.68, −1.02) | 0.163 | 0.163 | 0.027 | [−0.143, −0.094] | 0.184 → 0.144 | 0.558 → 0.447 |
  | Glaucoma | 0.060 | 0.153 → 0.069 | 1.58 | (0.51, −1.75) | 0.097 | 0.114 | 0.021 | [−0.086, −0.054] | 0.078 → 0.038 | 0.277 → 0.148 |
  | Cataract | 0.060 | 0.110 → 0.055 | 1.12 | (0.68, −2.02) | 0.055 | 0.056 | 0.016 | [−0.049, −0.024] | 0.044 → 0.027 | 0.181 → 0.121 |
  | AMD | 0.041 | 0.099 → 0.048 | 1.46 | (0.45, −2.01) | 0.058 | 0.084 | 0.012 | [−0.055, −0.028] | 0.049 → 0.024 | 0.197 → 0.106 |
  | Hypertensive ret. | 0.033 | 0.075 → 0.031 | 1.40 | (0.33, −2.21) | 0.049 | 0.063 | 0.009 | [−0.054, −0.018] | 0.045 → 0.029 | 0.174 → 0.120 |

  Adaptive ECE raw → Platt: DR 0.162 → 0.021, G 0.104 → 0.020, C 0.062 →
  0.012, A 0.067 → 0.014, H 0.059 → 0.016.

  Across seeds (test ECE, mean ± s.d. of 3 seeds), raw → Platt:
  A: DR 0.148 ± 0.047 → 0.034 ± 0.006; G 0.161 → 0.020; C 0.129 → 0.017;
  A 0.174 → 0.017; H 0.129 → 0.011. B (224 px): DR 0.114 → 0.032; G 0.135
  → 0.018; C 0.072 → 0.013; A 0.093 → 0.022; H 0.099 → 0.009. C (512
  patient): DR 0.116 → 0.032; G 0.111 → 0.021; C 0.116 → 0.017; A 0.079 →
  0.014; H 0.075 → 0.009. Temperature scaling left ECE unchanged or worse
  for every group and label.

  Device check (Platt-calibrated test ECE; max |Δ calibrated p| vs reference):
  N100 ONNX Runtime CPU and OpenVINO CPU and Pi TFLite FP32 give identical
  ECE to four decimals (DR 0.0266, G 0.0206, C 0.0164, A 0.0125, H 0.0095);
  max difference 2.5e-5 (N100) and 1.4e-4 (Pi). N100 iGPU: DR 0.0281, others
  within 0.0012; max difference 0.032.

  INT8 (development, validation, Platt map fitted on FP32, ECE):

  | variant | DR | G | C | A | H |
  |---|---|---|---|---|---|
  | FP32 (in-sample) | 0.025 | 0.025 | 0.014 | 0.009 | 0.012 |
  | dynamic-range INT8 | 0.029 | 0.015 | 0.015 | 0.015 | 0.006 |
  | full-integer INT8 | 0.080 | 0.045 | 0.051 | 0.018 | 0.009 |

- **Artifacts**: `results/run011/calibration.json` (all models, all
  metrics, temperatures, Platt parameters, bootstrap, device check, raw INT8
  ECE), `results/run011/int8_platt_transfer_dev.json`, scripts
  `results/run011_calibration.py`, `results/run011_int8_transfer.py`, log
  `results/logs/run011_calibration.log`, figure
  `paper/figs/run011_reliability.{pdf,png}` (final model, test,
  reliability curves raw / temperature / Platt per label).
- **Verdict**:
  - Raw outputs are badly over-confident towards disease: mean predicted
    probability is 1.5-2.6× the prevalence, as expected from the
    positive-class weights in the loss.
  - **Temperature scaling does not fix this** and makes it worse for
    glaucoma, AMD and hypertensive retinopathy: one temperature cannot
    remove a constant logit offset, and the NLL-optimal T > 1 pushes
    probabilities further towards 0.5.
  - **Platt scaling fixes it**: test ECE falls by 0.04-0.14 per label, with
    CIs excluding zero for every label, and mean predicted probability
    lands within 0.002-0.009 of prevalence. It costs two operations per
    label on the device.
  - The calibrated outputs survive deployment exactly on every CPU runtime
    (identical ECE to four decimals on N100 and Pi).
  - Full-integer INT8 breaks the calibration map: DR ECE triples (0.025 →
    0.080) and cataract nearly quadruples. Dynamic-range INT8 roughly
    preserves it. This reinforces the FP32 recommendation.
- **Limitations**: one dataset; calibration fitted on the 1270-image
  validation set, which also selected checkpoints; INT8 analysis is on the
  development model and in-sample for FP32.
