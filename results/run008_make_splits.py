"""Write run008 train/val/test manifests and print their composition.
Usage (repo root): .venv/bin/python results/run008_make_splits.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, "results")
from run008_common import MANIFEST_COLS, SPLIT_DIR, load_df, split3  # noqa: E402


def main():
    df = load_df()
    train, val, test = split3(df)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d[MANIFEST_COLS].to_csv(f"{SPLIT_DIR}/{name}.csv", index=False)
    # re-read and re-check from disk
    r = {n: pd.read_csv(f"{SPLIT_DIR}/{n}.csv") for n in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (set(r[a]["ID"]) & set(r[b]["ID"]))
        assert not (set(r[a]["filename"]) & set(r[b]["filename"]))
    print("zero patient / file overlap between train, val, test: OK")
    for n, d in r.items():
        g = d["dr_grade_eye"].value_counts().reindex([0, 1, 2, 3, 4, -1], fill_value=0).tolist()
        print(f"{n}: images={len(d)} patients={d['ID'].nunique()} "
              + " ".join(f"{c}={int(d[c].sum())}" for c in ["D", "D_eye", "G", "C", "A", "H"])
              + f" | eye grades none/mild/mod/sevNPDR/PDR/ungraded={g}")
    # edge INT8 calibration images (runs 004/006) were drawn from the original train split:
    cal = "edge/artifacts/calib_manifest.csv"
    if os.path.exists(cal):
        c = pd.read_csv(cal)
        col = "filename" if "filename" in c.columns else c.columns[0]
        print(f"edge calib manifest: {len(c)} rows, overlap with test={len(set(c[col]) & set(r['test']['filename']))}, "
              f"with val={len(set(c[col]) & set(r['val']['filename']))}")


if __name__ == "__main__":
    main()
