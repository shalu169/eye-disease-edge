"""
Run 010 step 5a: preprocess the 81 IDRiD segmentation images (Porwal et al.
2018, Data 3(3):25, doi 10.3390/data3030025; CC BY 4.0) like the ODIR images
the model was trained on (ODIR "preprocessed_images": fundus disc cropped to
a square that inscribes the circle, resized to 512x512), applying the identical
geometric transform to the lesion masks.

Source copy: Kaggle mirror aaryapatel98/indian-diabetic-retinopathy-image-dataset
(zip sha256 in data/idrid/zip.sha256; licence files inside the zip: CC BY 4.0).

Geometry per image (4288x2848, the disc is cut at top/bottom):
  FOV = grey > 15. Horizontal extent [x0, x1] of the FOV gives the disc diameter
  D = x1 - x0 + 1 and centre cx; the vertical centre cy is the mean row index
  of the rows whose FOV width is >= 99.5 % of the maximum width. Square box
  [cx - D/2, cy - D/2, cx + D/2, cy + D/2] (out-of-image parts padded black),
  then resize to 512x512 (PIL bilinear for the image; for masks the fraction of
  lesion pixels per output pixel via PIL BOX filter, and an output pixel is
  "lesion" if that fraction > 0, so microaneurysms of a few source pixels are
  not lost at the ~6.7x downscale; the fraction is stored too).
Output: data/idrid/prep512/<id>.png, data/idrid/prep512/<id>_masks.npz
        (MA, HE, EX, SE, OD as uint8 0/1 and *_frac float16), manifest
        results/run010/idrid_manifest.csv.
Usage: .venv/bin/python results/run010_idrid_prep.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

ROOT = "data/idrid/raw/A.%20Segmentation/A. Segmentation"
OUTD = "data/idrid/prep512"
TYPES = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates",
         "OD": "5. Optic Disc"}
S = 512


def box_for(rgb):
    g = np.asarray(rgb.convert("L"), dtype=np.float32) > 15
    cols = np.where(g.any(0))[0]
    x0, x1 = cols.min(), cols.max()
    widths = g.sum(1)
    rows = np.where(widths >= 0.995 * widths.max())[0]
    cy = rows.mean()
    cx = (x0 + x1) / 2
    D = x1 - x0 + 1
    return (int(round(cx - D / 2)), int(round(cy - D / 2)), int(round(cx - D / 2)) + D,
            int(round(cy - D / 2)) + D), {"x0": int(x0), "x1": int(x1), "cy": float(cy), "D": int(D),
                                          "touches_top": bool(g[0].any()), "touches_bottom": bool(g[-1].any())}


def main():
    os.makedirs(OUTD, exist_ok=True)
    rows = []
    for split, sub in [("train", "a. Training Set"), ("test", "b. Testing Set")]:
        for p in sorted(glob.glob(f"{ROOT}/1. Original Images/{sub}/*.jpg")):
            iid = os.path.basename(p)[:-4]
            im = Image.open(p).convert("RGB")
            box, geo = box_for(im)
            im.crop(box).resize((S, S), Image.BILINEAR).save(f"{OUTD}/{iid}.png")
            masks = {}
            for t, d in TYPES.items():
                mp = f"{ROOT}/2. All Segmentation Groundtruths/{sub}/{d}/{iid}_{t}.tif"
                if os.path.exists(mp):
                    m = np.asarray(Image.open(mp))
                    if m.ndim == 3:
                        m = m[..., 0]
                    assert m.shape == (im.size[1], im.size[0])
                    mi = Image.fromarray(((m > 0) * 255).astype(np.uint8)).crop(box)
                    frac = np.asarray(mi.resize((S, S), Image.BOX), dtype=np.float32) / 255.0
                    src_px = int((m > 0).sum())
                else:
                    frac = np.zeros((S, S), np.float32)
                    src_px = 0
                masks[t] = (frac > 0).astype(np.uint8)
                masks[t + "_frac"] = frac.astype(np.float16)
                geo[f"{t}_src_pixels"] = src_px
                geo[f"{t}_px512"] = int(masks[t].sum())
            np.savez_compressed(f"{OUTD}/{iid}_masks.npz", **masks)
            rows.append({"id": iid, "split": split, "orig_w": im.size[0], "orig_h": im.size[1],
                         "box": json.dumps([int(v) for v in box]), **geo})
    df = pd.DataFrame(rows)
    df.to_csv("results/run010/idrid_manifest.csv", index=False)
    print(df.describe().T.to_string())
    print(len(df), "images;", {t: int((df[f"{t}_px512"] > 0).sum()) for t in TYPES})


if __name__ == "__main__":
    main()
