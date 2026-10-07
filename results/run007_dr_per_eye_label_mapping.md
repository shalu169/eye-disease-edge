# Run 007 — per-eye DR label mapping (ODIR-5K keywords)

Implemented in `results/run007_common.py` (`load_df`). Source: `data/full_df.csv`.

## Row -> eye -> keywords

`full_df.csv` has one row per image (6392 rows, 3358 patients). Eye is parsed
from `filename` (`<ID>_left.jpg` / `<ID>_right.jpg`; matches `Left-Fundus` /
`Right-Fundus` for every row). That eye's keyword string is
`Left-Diagnostic Keywords` or `Right-Diagnostic Keywords`. Keywords are split on
both ASCII `,` and full-width `，` (both occur), stripped.

## Keyword -> per-eye DR grade

Vocabulary was inspected in full (94 distinct keywords). DR keywords:

| keyword (exact) | per-eye D | grade |
|---|---|---|
| `mild nonproliferative retinopathy` | 1 | 1 mild NPDR |
| `moderate non proliferative retinopathy` | 1 | 2 moderate NPDR |
| `suspected moderate non proliferative retinopathy` | 1 | 2 (flagged suspected) |
| `severe nonproliferative retinopathy` | 1 | 3 severe NPDR |
| `proliferative diabetic retinopathy` | 1 | 4 PDR |
| `severe proliferative diabetic retinopathy` | 1 | 4 PDR |
| `diabetic retinopathy` | 1 | ungraded (-1) |
| `suspected diabetic retinopathy` | 1 | ungraded (flagged suspected) |
| `suspicious diabetic retinopathy` | 1 | ungraded (flagged suspected) |

Everything else -> per-eye D = 0. Multiple DR keywords on one eye (4 images,
all repeats of the same grade) -> max grade.

Explicitly NOT DR: `hypertensive retinopathy`, `myopia retinopathy`,
`central serous chorioretinopathy`, `old chorioretinopathy`, vein/artery
occlusions.

DR-adjacent but not DR keywords (label stays 0; such negatives are dropped,
together with "suspected" positives, only in the `eyeD_strict` evaluation):
`laser spot`, `post laser photocoagulation`, `fundus laser photocoagulation spots`,
`post retinal laser surgery`, `intraretinal hemorrhage`,
`intraretinal microvascular abnormality`, `suspected microvascular anomalies`,
`maculopathy`.

## Validation of the mapping against ODIR's own D flag

At patient level, ODIR `D == 1` exactly when at least one of the patient's two
keyword strings contains one of the 9 DR keywords above: 1105/1105 D-positive
patients matched, 0/2253 D-negative patients had one (no mismatches). So the
mapping reproduces ODIR's D rule exactly; the only change is per-eye instead
of per-patient.

## Counts (images)

| set | images | patient D=1 | per-eye D=1 | D=1 but eye has no DR keyword | per-eye grades (none / mild / mod / severe NPDR / PDR / ungraded) |
|---|---|---|---|---|---|
| all | 6392 | 2123 | 1722 | 401 (18.9% of D=1) | 4670 / 538 / 947 / 157 / 29 / 51 |
| train | 5122 | 1715 | 1399 | 316 | 3723 / 440 / 767 / 127 / 24 / 41 |
| val | 1270 | 408 | 323 | 85 (20.8% of D=1) | 947 / 98 / 180 / 30 / 5 / 10 |

No image has per-eye D=1 with patient D=0 (by construction).

Of the 401 "patient-D but eye-not-DR" images, 250 are keyword `normal fundus`
alone; the rest carry other findings (cataract 31, drusen 19, hypertensive
retinopathy 18, epiretinal membrane variants ~33, pathological myopia 11, ...).
These are the label-noise positives of the patient-level protocol.

`eyeD_strict` val subset: 1258 images (drops 2 suspected-DR positives and 10
DR-adjacent negatives).

## Severity / "referable-ish" definitions used in evaluation

- per-stratum AUC: eyes of grade g (positives) vs per-eye no-DR eyes (negatives).
- `referable_modplus_vs_rest`: grade >= 2 (moderate NPDR or worse) positive vs
  grade 0 or 1 negative; the 10 ungraded val eyes are excluded. This is the
  usual "referable DR" cut (moderate NPDR+), but ODIR has no DME annotation,
  so it is only an approximation of the EyePACS/Messidor referable protocol.
