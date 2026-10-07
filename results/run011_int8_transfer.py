"""
Run 011 add-on (development, 224 px run002 model, VALIDATION set, Pi run 006
TFLite probabilities): fit a Platt map on the FP32 outputs and apply it
unchanged to the dynamic-range and full-integer INT8 outputs of the same
images. In-sample for FP32, so only the relative degradation is meaningful.
Usage (repo root): .venv/bin/python results/run011_int8_transfer.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, "results")
from run011_calibration import ece, fit_platt, platt  # noqa: E402
from run008_common import NAMES, load_df, split3  # noqa: E402

_, val, _ = split3(load_df())
val = val.reset_index(drop=True)
pf = np.load("results/pi_run006/probs_tfl_fp32_t4.npy")
out = {}
for k in ["fp32_t4", "drq_t4", "int8_t4"]:
    pq = np.load(f"results/pi_run006/probs_tfl_{k}.npy")
    out[k] = {}
    for i, n in enumerate(NAMES):
        y = val[n].values.astype(float)
        out[k][n] = round(ece(y, platt(pq[:, i], fit_platt(y, pf[:, i]))), 4)
json.dump({"note": "Platt map fitted on FP32 validation probabilities, applied to each variant on the same "
                   "images; in-sample for FP32, so only the relative degradation is meaningful",
           "ece_after_fp32_platt": out}, open("results/run011/int8_platt_transfer_dev.json", "w"), indent=1)
print(json.dumps(out))
