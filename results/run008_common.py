"""
Run 008 shared code: three-way patient-level split.
  - original split (GroupShuffleSplit test_size=0.2, random_state=42 on full_df by ID)
    -> original train (5122) / original val (1270). Val is kept as VALIDATION
    (checkpoint selection, threshold choice) only.
  - NEW held-out TEST = GroupShuffleSplit(test_size=0.2, random_state=2026) by ID
    applied to the ORIGINAL train dataframe; the rest is the new training set.
The test images were only ever *trained on* in runs 001-007, never evaluated on,
so no earlier design decision (resolution, per-eye DR labels, checkpointing)
used them.
Labels / per-eye DR derivation reused unchanged from run007_common.py.
Run all run008 scripts from the repo root.
"""
import sys

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "results")
from run007_common import MEAN, STD, OdirDataset, _auc, dr_metrics, load_df, split  # noqa: E402,F401

NAMES = ["D", "G", "C", "A", "H"]
MANIFEST_COLS = ["filename", "ID", "eye", "D", "G", "C", "A", "H", "D_eye", "dr_grade_eye",
                 "dr_suspected_eye", "dr_adjacent_eye", "N", "M", "O"]
SPLIT_DIR = "results/run008_splits"


def split3(df):
    train_orig, val = split(df)
    assert len(val) == 1270 and len(train_orig) == 5122
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    tr_idx, te_idx = next(gss.split(train_orig, groups=train_orig["ID"]))
    train = train_orig.iloc[tr_idx].reset_index(drop=True)
    test = train_orig.iloc[te_idx].reset_index(drop=True)
    sets = {"train": train, "val": val, "test": test}
    ids = {k: set(v["ID"]) for k, v in sets.items()}
    files = {k: set(v["filename"]) for k, v in sets.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (ids[a] & ids[b]), f"patient overlap {a}/{b}"
        assert not (files[a] & files[b]), f"file overlap {a}/{b}"
    assert sum(len(v) for v in sets.values()) == len(df)
    return train, val, test


def thr_at_sens(y, p, target=0.90):
    """Largest threshold t with sensitivity(p >= t) >= target (chosen on VALIDATION)."""
    y = np.asarray(y).astype(bool)
    pos = np.sort(np.asarray(p)[y])[::-1]
    k = int(np.ceil(target * len(pos)))  # need at least k positives >= t
    return float(pos[k - 1])
